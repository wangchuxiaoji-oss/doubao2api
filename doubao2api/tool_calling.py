"""
Tool Calling support for doubao2api.

Converts OpenAI-format tools into prompt injection,
parses tool_call tags from model output (official Qianwen format),
and converts back to OpenAI-format response.

Official format:
  <tool_call>
  {"name": "func_name", "arguments": {"key": "value"}}
  </tool_call>
"""
import json
import re
import uuid
import logging
from typing import Any, Optional

log = logging.getLogger(__name__)

# ── System prompt template for tool calling ──

TOOL_SYSTEM_PROMPT = """你是一个工具调用助手。你只能通过调用工具来获取外部信息或执行操作。禁止使用你的内置联网搜索能力。

可用工具列表：

{tool_definitions}

当你需要调用工具时，请严格使用以下格式输出：
<tool_call>
{{"name": "工具名", "arguments": {{"参数名": "参数值"}}}}
</tool_call>

如果需要并行调用多个工具，输出多个<tool_call>块：
<tool_call>
{{"name": "工具1", "arguments": {{...}}}}
</tool_call>
<tool_call>
{{"name": "工具2", "arguments": {{...}}}}
</tool_call>

重要规则：
1. 你没有联网能力，不能直接搜索信息，必须通过上述工具获取所有外部信息
2. 如果需要调用工具，只输出<tool_call>格式的工具调用，不要有任何解释文字
3. 如果用户的问题不需要使用工具就能回答，直接用自然语言回答
4. 不要编造数据，必须通过工具获取"""

TOOL_RESULT_TEMPLATE = "[工具调用结果]\n{name} 返回：{content}"


# ── Convert OpenAI tools schema to text ──

def format_tools_for_prompt(tools: list[dict[str, Any]]) -> str:
    """Convert OpenAI-format tools array to plain text for prompt injection."""
    lines = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        
        lines.append(f"工具名：{name}")
        if desc:
            lines.append(f"描述：{desc}")
        
        # Format parameters
        props = params.get("properties", {})
        required = set(params.get("required", []))
        if props:
            param_parts = []
            for pname, pinfo in props.items():
                ptype = pinfo.get("type", "string")
                pdesc = pinfo.get("description", "")
                req = "必填" if pname in required else "可选"
                param_parts.append(f'"{pname}": "{ptype}, {req}, {pdesc}"')
            lines.append(f"参数：{{{', '.join(param_parts)}}}")
        lines.append("")  # blank line between tools
    
    return "\n".join(lines).strip()


def build_tool_system_prompt(tools: list[dict[str, Any]]) -> str:
    """Build the full system prompt with tool definitions injected."""
    tool_defs = format_tools_for_prompt(tools)
    return TOOL_SYSTEM_PROMPT.format(tool_definitions=tool_defs)


# ── Parser for <tool_call> blocks (official Qianwen format) ──

# Matches individual <tool_call>...</tool_call> blocks
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)

# Matches JSON object followed by </tool_call> (missing opening tag - thinking mode)
_TOOL_CALL_PARTIAL_RE = re.compile(
    r"(\{[^{}]*\"name\"[^{}]*\"arguments\"[^{}]*\{[^}]*\}[^{}]*\})\s*</tool_call>",
    re.DOTALL,
)

# Even more lenient: just find JSON with "name" and "arguments" keys near </tool_call>
_TOOL_CALL_JSON_RE = re.compile(
    r"(\{[^{}]*\"name\"\s*:\s*[^,}]+[^{}]*\"arguments\"\s*:\s*\{[^}]*\}[^{}]*\})",
    re.DOTALL,
)

# Legacy format support (for backward compat with old model outputs)
_TOOL_CALLS_RE = re.compile(
    r"<tool_calls>\s*(.*?)\s*</tool_calls>",
    re.DOTALL,
)
_INVOKE_RE = re.compile(
    r'<invoke\s+name="([^"]+)">\s*(.*?)\s*</invoke>',
    re.DOTALL,
)
_PARAM_RE = re.compile(
    r'<parameter\s+name="([^"]+)">(.*?)</parameter>',
    re.DOTALL,
)


def _try_parse_json(json_str: str) -> Optional[dict]:
    """Try to parse JSON, with fallback fixes for common model output issues."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    # Fix unquoted values
    fixed = re.sub(
        r'(?<=[{,:])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?=[,}])',
        r' "\1"', json_str,
    )
    fixed = re.sub(
        r'(?<=[{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:',
        r' "\1":', fixed,
    )
    try:
        return json.loads(fixed)
    except (json.JSONDecodeError, TypeError):
        return None


def parse_tool_calls_xml(text: str) -> Optional[list[dict[str, Any]]]:
    """Parse tool_call blocks from model output.
    
    Supports both:
    - Official format: <tool_call>{"name":..., "arguments":...}</tool_call>
    - Legacy format: <tool_calls><invoke name="...">...</invoke></tool_calls>
    
    Returns list of OpenAI-format tool_call dicts, or None if not found.
    """
    tool_calls = []
    
    # Try official format first
    for m in _TOOL_CALL_RE.finditer(text):
        json_str = m.group(1).strip()
        try:
            obj = json.loads(json_str)
        except json.JSONDecodeError:
            # Try to fix common model output issues:
            # 1. Unquoted values like {name: get_weather} -> {"name": "get_weather"}
            fixed = re.sub(
                r'(?<=[{,:])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?=[,}])',
                r' "\1"',
                json_str,
            )
            # 2. Unquoted keys
            fixed = re.sub(
                r'(?<=[{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:',
                r' "\1":',
                fixed,
            )
            try:
                obj = json.loads(fixed)
            except (json.JSONDecodeError, TypeError) as e:
                log.warning("Failed to parse tool_call JSON even after fix: %s | raw: %s", e, json_str[:200])
                continue
        try:
            name = obj.get("name", "")
            arguments = obj.get("arguments", {})
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            elif not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            })
        except (AttributeError, TypeError) as e:
            log.warning("Failed to extract tool_call fields: %s", e)
            continue
    
    if tool_calls:
        return tool_calls
    
    # Fallback 2: partial format (missing opening <tool_call> tag, common in thinking mode)
    if "</tool_call>" in text:
        for m in _TOOL_CALL_PARTIAL_RE.finditer(text):
            json_str = m.group(1).strip()
            obj = _try_parse_json(json_str)
            if obj and isinstance(obj, dict) and "name" in obj:
                arguments = obj.get("arguments", {})
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                elif not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {"name": obj["name"], "arguments": arguments},
                })
        if tool_calls:
            return tool_calls

    # Fallback 3: just find JSON with name+arguments pattern (no tags at all)
    if not tool_calls:
        for m in _TOOL_CALL_JSON_RE.finditer(text):
            json_str = m.group(1).strip()
            obj = _try_parse_json(json_str)
            if obj and isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                arguments = obj.get("arguments", {})
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                elif not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {"name": obj["name"], "arguments": arguments},
                })
        if tool_calls:
            return tool_calls

    # Fallback 4: legacy <tool_calls><invoke> format
    match = _TOOL_CALLS_RE.search(text)
    if not match:
        return None
    inner = match.group(1)
    for invoke_match in _INVOKE_RE.finditer(inner):
        func_name = invoke_match.group(1)
        params_text = invoke_match.group(2)
        arguments = {}
        for param_match in _PARAM_RE.finditer(params_text):
            param_name = param_match.group(1)
            param_value = param_match.group(2).strip()
            arguments[param_name] = param_value
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        })
    
    return tool_calls if tool_calls else None


def is_tool_call_start(text: str) -> bool:
    """Check if accumulated text looks like the start of a tool call."""
    stripped = text.strip()
    if stripped.startswith("<tool_call>") or stripped.startswith("<tool_call\n"):
        return True
    if stripped.startswith("<tool_calls>"):
        return True
    # Thinking mode: tool call may start with JSON directly (no opening tag)
    # Check if it looks like {"name": ...} or starts with { followed by "name"
    if stripped.startswith("{") and '"name"' in stripped[:100]:
        return True
    return False


def has_complete_tool_calls(text: str) -> bool:
    """Check if text contains at least one complete tool_call block or parseable JSON."""
    if "</tool_call>" in text or "</tool_calls>" in text:
        return True
    # In thinking mode, might just be bare JSON with name+arguments
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        if '"name"' in stripped and '"arguments"' in stripped:
            return True
    return False


# ── Message conversion for multi-turn tool use ──

# Max chars for the flattened prompt (~28K tokens safe zone for Qwen web)
MAX_PROMPT_CHARS = 50000
# Max chars per individual tool result
MAX_TOOL_RESULT_CHARS = 8000


def _truncate_tool_result(content: str, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
    """Truncate a tool result, keeping head and tail for context."""
    if not content or len(content) <= max_chars:
        return content
    half = max_chars // 2 - 50
    return (content[:half] +
            f"\n\n... [truncated {len(content) - max_chars} chars] ...\n\n" +
            content[-half:])


def convert_messages_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_chars: int = MAX_PROMPT_CHARS,
) -> str:
    """Convert OpenAI-format messages (including role:tool) to plain text prompt.
    
    Handles:
    - Injects tool system prompt
    - Converts role:assistant with tool_calls to official format
    - Converts role:tool results to readable text
    - Smart truncation to stay within token limits
    """
    tool_system = build_tool_system_prompt(tools)
    parts = []
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "system":
            if content:
                parts.append(f"[system]: {content}\n\n{tool_system}")
            continue
        
        elif role == "tool":
            name = msg.get("name", "unknown_tool")
            tool_content = _truncate_tool_result(content or "")
            parts.append(TOOL_RESULT_TEMPLATE.format(
                name=name, content=tool_content
            ))
        
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                xml = _reconstruct_tool_calls_xml(tool_calls)
                parts.append(f"[assistant]: {xml}")
            elif content:
                parts.append(f"[assistant]: {content}")
        
        elif role == "user":
            if isinstance(content, str):
                parts.append(f"[user]: {content}")
            elif isinstance(content, list):
                text_parts = [p.get("text", "") for p in content 
                             if isinstance(p, dict) and p.get("type") == "text"]
                if text_parts:
                    parts.append(f"[user]: {''.join(text_parts)}")
    
    # If no system message was found, prepend tool system prompt
    if not any(m.get("role") == "system" for m in messages):
        parts.insert(0, f"[system]: {tool_system}")
    
    result = "\n\n".join(parts)
    
    # If still over limit after per-result truncation, drop oldest tool rounds
    if len(result) > max_chars:
        result = _drop_old_rounds(parts, max_chars)
    
    return result


def _drop_old_rounds(parts: list[str], max_chars: int) -> str:
    """Drop oldest tool call/result pairs until under the limit.
    
    Strategy: keep first part (system) and last few parts (recent context),
    drop middle parts (old tool results) with a summary marker.
    """
    if not parts:
        return ""
    
    # Always keep: first part (system+tools) and last part
    # Drop from position 1 onwards until we fit
    header = parts[0]  # system prompt
    
    # Find how many parts we can keep from the end
    budget = max_chars - len(header) - 200  # reserve for join separators + marker
    kept_tail = []
    tail_size = 0
    
    for part in reversed(parts[1:]):
        part_size = len(part) + 4  # +4 for "\n\n" separator
        if tail_size + part_size <= budget:
            kept_tail.insert(0, part)
            tail_size += part_size
        else:
            break
    
    dropped_count = len(parts) - 1 - len(kept_tail)
    if dropped_count > 0:
        marker = f"[... {dropped_count} earlier messages truncated to fit context limit ...]"
        return "\n\n".join([header, marker] + kept_tail)
    else:
        return "\n\n".join([header] + kept_tail)


def _reconstruct_tool_calls_xml(tool_calls: list[dict[str, Any]]) -> str:
    """Reconstruct <tool_call> blocks from OpenAI-format tool_calls for context."""
    parts = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        args_str = func.get("arguments", "{}")
        try:
            args = json.loads(args_str)
        except (json.JSONDecodeError, TypeError):
            args = {}
        obj = {"name": name, "arguments": args}
        parts.append(f"<tool_call>\n{json.dumps(obj, ensure_ascii=False)}\n</tool_call>")
    return "\n".join(parts)
