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

TOOL_SYSTEM_PROMPT = """你是一个工具调用助手。禁止使用内置联网搜索。

## 可用工具
{tool_definitions}

## 调用格式
<tool_call>
{{"name": "工具名", "arguments": {{"参数名": "参数值"}}}}
</tool_call>

多个工具可并行调用（输出多个<tool_call>块）。

## 规则
1. 必须使用上面列出的**精确工具名**，不要用别名或猜测的名字
2. 参数必须严格匹配工具签名中的字段名（!为必填，?为可选）
3. 需要调用工具时只输出<tool_call>块，不加解释文字
4. 不需要工具时直接用自然语言回答"""

TOOL_RESULT_TEMPLATE = "[工具调用结果]\n{name} 返回：{content}"


# ── Convert OpenAI tools schema to text ──

def _compress_type(pinfo: dict) -> str:
    """Compress a JSON Schema property to TS-like type notation."""
    ptype = pinfo.get("type", "string")
    enum = pinfo.get("enum")
    if enum:
        return "|".join(json.dumps(v, ensure_ascii=False) for v in enum)
    if ptype == "array":
        items = pinfo.get("items", {})
        item_type = items.get("type", "any")
        return f"{item_type}[]"
    if ptype == "object":
        # Nested object - just use 'object'
        return "object"
    return ptype


def _compress_params(params: dict) -> str:
    """Compress JSON Schema parameters to TS-like signature.
    
    Example output: {file_path!: string, encoding?: "utf-8"|"base64"}
    '!' = required, '?' = optional
    """
    props = params.get("properties", {})
    required = set(params.get("required", []))
    if not props:
        return "{}"
    
    parts = []
    for pname, pinfo in props.items():
        marker = "!" if pname in required else "?"
        ptype = _compress_type(pinfo)
        pdesc = pinfo.get("description", "")
        # Only include short descriptions (< 60 chars) inline
        if pdesc and len(pdesc) < 60:
            parts.append(f"{pname}{marker}: {ptype} /* {pdesc} */")
        else:
            parts.append(f"{pname}{marker}: {ptype}")
    return "{" + ", ".join(parts) + "}"


# Priority tools get full param expansion; others get name+desc only
PRIORITY_TOOLS = {
    "Read", "Write", "Edit", "Shell", "Bash", "Glob", "Grep",
    "WebFetch", "Task", "TodoWrite",
    # Common OpenCode tools
    "read", "write", "edit", "bash", "glob", "grep",
    "webfetch", "task", "todowrite",
}


def format_tools_for_prompt(tools: list[dict[str, Any]]) -> str:
    """Convert OpenAI-format tools array to compressed TS-like signatures.
    
    Uses compact notation for ~90% space savings vs verbose JSON Schema.
    Priority tools get full param expansion; others get name+desc only when >12 tools.
    """
    lines = []
    expand_all = len(tools) <= 12
    
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        
        is_priority = expand_all or name in PRIORITY_TOOLS
        
        if is_priority:
            sig = _compress_params(params)
            # Truncate description to 120 chars for prompt space
            short_desc = desc[:120] + "..." if len(desc) > 120 else desc
            lines.append(f"- {name}{sig}: {short_desc}")
        else:
            # Minimal: just name and short description
            short_desc = desc[:80] + "..." if len(desc) > 80 else desc
            lines.append(f"- {name}: {short_desc}")
    
    return "\n".join(lines)


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

# Qianwen web model token limit is ~32K tokens.
# We use a character-based heuristic: typical code = ~3 chars/token.
# Safe budget: 30K tokens × 3 chars = 90K chars.
# But we must account for JSON escaping in the POST body (~1.3x for code with newlines).
# Effective safe limit: 90K / 1.3 ≈ 70K chars.
MAX_PROMPT_CHARS = 70000
# Max chars per individual tool result (keep enough context per file)
MAX_TOOL_RESULT_CHARS = 12000


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
    - Smart truncation: preserves first user message (original task)
    - CURRENT TASK injection for multi-step tool chains
    """
    tool_system = build_tool_system_prompt(tools)
    parts = []
    first_user_msg = None  # Track the original task
    
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
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text_parts = [p.get("text", "") for p in content 
                             if isinstance(p, dict) and p.get("type") == "text"]
                text = "".join(text_parts)
            if text:
                if first_user_msg is None:
                    first_user_msg = text
                parts.append(f"[user]: {text}")
    
    # If no system message was found, prepend tool system prompt
    if not any(m.get("role") == "system" for m in messages):
        parts.insert(0, f"[system]: {tool_system}")
    
    # Clean refusals from history to prevent cascade
    parts = clean_refusals_from_history(parts)
    
    result = "\n\n".join(parts)
    
    # If still over limit after per-result truncation, drop oldest tool rounds
    if len(result) > max_chars:
        result = _drop_old_rounds(parts, max_chars, first_user_msg)
    
    return result


def _drop_old_rounds(parts: list[str], max_chars: int, first_user_msg: Optional[str] = None) -> str:
    """Drop oldest tool call/result pairs until under the limit.
    
    Strategy:
    1. If system prompt is too large, truncate it
    2. Always preserve first user message (original task) for context
    3. Keep most recent tool rounds
    4. Inject CURRENT TASK reminder if rounds were dropped
    """
    if not parts:
        return ""
    
    header = parts[0]  # system prompt (may be very large)
    
    # If header alone exceeds 50% of budget, truncate it aggressively
    max_header = int(max_chars * 0.5)
    if len(header) > max_header:
        # Keep the tool instruction part and truncate the system prompt
        tool_marker = "## 可用工具"
        marker_pos = header.find(tool_marker)
        if marker_pos > 0:
            # Keep: first 1500 chars of system + all tool instructions
            sys_prefix = header[:1500]
            tool_instructions = header[marker_pos:]
            header = sys_prefix + "\n\n[... system prompt truncated ...]\n\n" + tool_instructions
        else:
            header = header[:max_header] + "\n[... truncated ...]"
        # If still too large, truncate tool definitions too
        if len(header) > max_header:
            header = header[:max_header] + "\n[... truncated ...]"
    
    # Build CURRENT TASK reminder from first user message
    task_reminder = ""
    if first_user_msg:
        # Truncate to 500 chars max
        task_text = first_user_msg[:500]
        if len(first_user_msg) > 500:
            task_text += "..."
        task_reminder = f"\n\n[CURRENT TASK]: {task_text}"
    
    # Calculate budget for conversation history
    budget = max_chars - len(header) - len(task_reminder) - 200
    
    # Ensure minimum budget for at least some history
    if budget < 5000:
        budget = 5000
    
    # Keep parts from the end (most recent first)
    kept_tail = []
    tail_size = 0
    
    for part in reversed(parts[1:]):
        part_size = len(part) + 4  # +4 for "\n\n" separator
        if tail_size + part_size <= budget:
            kept_tail.insert(0, part)
            tail_size += part_size
        else:
            break
    
    # If we couldn't keep anything, force-keep at least the last 2 parts
    if not kept_tail and len(parts) > 1:
        kept_tail = parts[-2:]
        kept_tail = [p[:3000] if len(p) > 3000 else p for p in kept_tail]
    
    dropped_count = len(parts) - 1 - len(kept_tail)
    if dropped_count > 0:
        marker = f"[... {dropped_count} earlier messages omitted ...]"
        result_parts = [header, marker]
        if task_reminder:
            result_parts.append(task_reminder)
        result_parts.extend(kept_tail)
        return "\n\n".join(result_parts)
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


# ── Refusal Cleaning ──
# Patterns that indicate model refusal (cascade prevention)
_REFUSAL_PATTERNS = [
    re.compile(r"(?i)i('m| am) (sorry|unable|not able|cannot)", re.DOTALL),
    re.compile(r"(?i)i can'?t (help|assist|do|provide|fulfill)", re.DOTALL),
    re.compile(r"(?i)as an ai (language model|assistant)", re.DOTALL),
    re.compile(r"(?i)i('m| am) not (allowed|permitted|able)", re.DOTALL),
    re.compile(r"(?i)(抱歉|对不起|很遗憾).{0,20}(无法|不能|做不到)", re.DOTALL),
    re.compile(r"(?i)作为(一个)?(AI|人工智能|语言模型)", re.DOTALL),
    re.compile(r"(?i)我(无法|不能|没有能力)(帮助|协助|完成|执行)", re.DOTALL),
    re.compile(r"(?i)this (request|task) (is|goes) (beyond|against)", re.DOTALL),
    re.compile(r"(?i)i (don'?t|do not) have (the ability|access|permission)", re.DOTALL),
    re.compile(r"(?i)违反.{0,10}(政策|规定|准则)", re.DOTALL),
]

_REFUSAL_REPLACEMENT = "[earlier assistant turn omitted by proxy]"


def clean_refusals_from_history(parts: list[str]) -> list[str]:
    """Remove refusal text from assistant messages in history.
    
    Prevents cascade refusals where model sees its own prior refusal
    and repeats it. Only cleans assistant messages, preserves tool_call blocks.
    """
    cleaned = []
    for part in parts:
        if not part.startswith("[assistant]:"):
            cleaned.append(part)
            continue
        # Don't clean if it contains tool_call (useful context)
        if "<tool_call>" in part:
            cleaned.append(part)
            continue
        # Check for refusal patterns
        content = part[len("[assistant]: "):]
        is_refusal = any(p.search(content) for p in _REFUSAL_PATTERNS)
        if is_refusal:
            cleaned.append(f"[assistant]: {_REFUSAL_REPLACEMENT}")
        else:
            cleaned.append(part)
    return cleaned


# ── Streaming Guard ──
class StreamingGuard:
    """Buffer for streaming output to detect incomplete tool_call tags.
    
    Accumulates initial chars (warmup) before emitting, and keeps a guard
    window at the tail to detect cross-chunk tool_call boundaries.
    """
    
    WARMUP_CHARS = 80       # Buffer before first emit
    GUARD_WINDOW = 200      # Tail buffer for cross-chunk detection
    
    def __init__(self):
        self._buffer = ""
        self._emitted = 0
        self._warmed_up = False
    
    def feed(self, chunk: str) -> str:
        """Feed a chunk, return text safe to emit (may be empty during warmup)."""
        self._buffer += chunk
        
        if not self._warmed_up:
            if len(self._buffer) < self.WARMUP_CHARS:
                return ""
            self._warmed_up = True
        
        # Keep guard window at tail
        safe_end = len(self._buffer) - self.GUARD_WINDOW
        if safe_end <= self._emitted:
            return ""
        
        emit = self._buffer[self._emitted:safe_end]
        self._emitted = safe_end
        return emit
    
    def flush(self) -> str:
        """Flush remaining buffer (call at stream end)."""
        if self._emitted < len(self._buffer):
            emit = self._buffer[self._emitted:]
            self._emitted = len(self._buffer)
            return emit
        return ""
    
    @property
    def full_buffer(self) -> str:
        """Get the complete accumulated buffer."""
        return self._buffer
    
    def has_incomplete_tool_call(self) -> bool:
        """Check if buffer has an opening <tool_call> without matching close."""
        opens = self._buffer.count("<tool_call>")
        closes = self._buffer.count("</tool_call>")
        return opens > closes


# ── Truncation Auto-Continue Detection ──

def detect_truncated_tool_call(text: str) -> bool:
    """Detect if model output was truncated mid-tool-call.
    
    Returns True if there's an unclosed <tool_call> tag or incomplete JSON.
    """
    opens = text.count("<tool_call>")
    closes = text.count("</tool_call>")
    if opens > closes:
        return True
    # Check for trailing incomplete JSON (starts with { but no matching })
    stripped = text.rstrip()
    if stripped.endswith(("{", '{"', '"name"')):
        return True
    return False


def build_continuation_prompt(original_output: str, max_anchor: int = 2000) -> str:
    """Build a prompt to continue from where the model was truncated.
    
    Takes the tail of the original output as anchor context.
    """
    anchor = original_output[-max_anchor:] if len(original_output) > max_anchor else original_output
    return (
        f"你的上一次回复在输出过程中被截断了。以下是你上次输出的末尾部分：\n"
        f"---\n{anchor}\n---\n"
        f"请从中断点继续输出，不要重复已输出的内容。"
    )


# ── Topic Isolation ──
# Detects when a new task begins in the same session, allowing us to
# discard irrelevant history and free up context budget.

def _extract_tokens(text: str) -> set:
    """Extract meaningful tokens from text for similarity comparison.
    
    Handles both Latin (word-based) and CJK (character bigram) text.
    """
    tokens = set()
    # Split on whitespace/punctuation for Latin words
    words = re.split(r'[\s\W_]+', text.lower())
    tokens.update(t for t in words if len(t) >= 2)
    
    # For CJK: extract character bigrams (2-char sliding window)
    # CJK Unified Ideographs range: \u4e00-\u9fff
    cjk_chars = re.findall(r'[\u4e00-\u9fff]', text)
    if len(cjk_chars) >= 2:
        for i in range(len(cjk_chars) - 1):
            tokens.add(cjk_chars[i] + cjk_chars[i + 1])
    
    return tokens


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


# Threshold below which we consider messages to be about different topics
TOPIC_SIMILARITY_THRESHOLD = 0.10


def detect_topic_change(prev_user_msg: str, new_user_msg: str) -> bool:
    """Detect if the new user message is about a completely different topic.
    
    Uses Jaccard similarity on token sets. Returns True if the messages
    are dissimilar enough to warrant discarding old history.
    
    Only triggers on substantial messages (>10 chars) to avoid false positives
    on short follow-ups like "yes", "continue", "fix it".
    """
    # Short messages are likely follow-ups, not new topics
    if len(new_user_msg) < 10 or len(prev_user_msg) < 10:
        return False
    
    tokens_prev = _extract_tokens(prev_user_msg)
    tokens_new = _extract_tokens(new_user_msg)
    
    # Need enough tokens to make a meaningful comparison
    if len(tokens_prev) < 3 or len(tokens_new) < 3:
        return False
    
    similarity = _jaccard_similarity(tokens_prev, tokens_new)
    return similarity < TOPIC_SIMILARITY_THRESHOLD


def filter_history_by_topic(
    messages: list[dict[str, Any]],
    max_history_on_change: int = 2,
) -> list[dict[str, Any]]:
    """Filter message history when a topic change is detected.
    
    If the latest user message is about a different topic than the previous
    user message, keep only the system prompt + last N messages.
    This frees up context budget for the new task.
    
    Args:
        messages: Full message list (OpenAI format)
        max_history_on_change: How many recent messages to keep on topic change
    
    Returns:
        Filtered message list (may be shorter if topic changed)
    """
    # Find the last two user messages
    user_msgs = [(i, m) for i, m in enumerate(messages) if m.get("role") == "user"]
    
    if len(user_msgs) < 2:
        return messages  # Not enough history to compare
    
    prev_idx, prev_msg = user_msgs[-2]
    curr_idx, curr_msg = user_msgs[-1]
    
    # Extract text content
    prev_text = prev_msg.get("content", "")
    curr_text = curr_msg.get("content", "")
    if isinstance(prev_text, list):
        prev_text = " ".join(p.get("text", "") for p in prev_text if isinstance(p, dict))
    if isinstance(curr_text, list):
        curr_text = " ".join(p.get("text", "") for p in curr_text if isinstance(p, dict))
    
    if detect_topic_change(prev_text, curr_text):
        log.info("Topic change detected (prev=%s... -> new=%s...), trimming history",
                 prev_text[:30], curr_text[:30])
        # Keep system messages + last N messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        recent = messages[-max_history_on_change:]
        return system_msgs + recent
    
    return messages


# ── Tool Name Obfuscation ──
# Qwen's web platform may reject certain tool names that conflict with
# built-in functions (e.g., "Read", "Write", "Search").
# We obfuscate outgoing names and de-obfuscate incoming names.

# Explicit aliases for commonly conflicting names
_TOOL_NAME_ALIASES = {
    "Read": "fs_read_file",
    "Write": "fs_write_file",
    "Edit": "fs_edit_file",
    "Bash": "exec_command",
    "Shell": "exec_shell",
    "Grep": "text_search",
    "Glob": "file_find",
    "WebFetch": "http_fetch",
    "WebSearch": "web_query",
}

# Reverse mapping for de-obfuscation
_TOOL_NAME_REVERSE = {v: k for k, v in _TOOL_NAME_ALIASES.items()}

# Generic prefix for names not in the explicit alias list
_OBFUSCATION_PREFIX = "u_"


class ToolNameObfuscator:
    """Handles bidirectional tool name obfuscation.
    
    Usage:
        obf = ToolNameObfuscator(enabled=True)
        # Before sending to model:
        obfuscated_tools = obf.obfuscate_tools(tools)
        prompt = obf.obfuscate_prompt_text(prompt)
        # After receiving from model:
        original_name = obf.deobfuscate_name(model_output_name)
    """
    
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._custom_map: dict[str, str] = {}      # original -> obfuscated
        self._custom_reverse: dict[str, str] = {}  # obfuscated -> original
    
    def obfuscate_name(self, name: str) -> str:
        """Convert original tool name to obfuscated name."""
        if not self.enabled:
            return name
        # Check explicit aliases first
        if name in _TOOL_NAME_ALIASES:
            return _TOOL_NAME_ALIASES[name]
        # Check custom map
        if name in self._custom_map:
            return self._custom_map[name]
        # Apply generic prefix
        obf_name = f"{_OBFUSCATION_PREFIX}{name.lower()}"
        self._custom_map[name] = obf_name
        self._custom_reverse[obf_name] = name
        return obf_name
    
    def deobfuscate_name(self, name: str) -> str:
        """Convert obfuscated name back to original."""
        if not self.enabled:
            return name
        # Check reverse aliases
        if name in _TOOL_NAME_REVERSE:
            return _TOOL_NAME_REVERSE[name]
        # Check custom reverse map
        if name in self._custom_reverse:
            return self._custom_reverse[name]
        # Strip prefix if present
        if name.startswith(_OBFUSCATION_PREFIX):
            return name[len(_OBFUSCATION_PREFIX):]
        return name
    
    def obfuscate_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Create a copy of tools with obfuscated names."""
        if not self.enabled:
            return tools
        result = []
        for tool in tools:
            if tool.get("type") != "function":
                result.append(tool)
                continue
            func = tool.get("function", {})
            original_name = func.get("name", "")
            obf_name = self.obfuscate_name(original_name)
            new_tool = {
                "type": "function",
                "function": {
                    **func,
                    "name": obf_name,
                },
            }
            result.append(new_tool)
        return result
    
    def obfuscate_prompt_text(self, text: str) -> str:
        """Replace bare tool names in prompt text with obfuscated versions."""
        if not self.enabled:
            return text
        for original, obfuscated in _TOOL_NAME_ALIASES.items():
            # Only replace whole-word occurrences to avoid partial matches
            text = re.sub(rf'\b{re.escape(original)}\b', obfuscated, text)
        for original, obfuscated in self._custom_map.items():
            text = re.sub(rf'\b{re.escape(original)}\b', obfuscated, text)
        return text
    
    def deobfuscate_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Restore original names in parsed tool_calls."""
        if not self.enabled:
            return tool_calls
        result = []
        for tc in tool_calls:
            func = tc.get("function", {})
            obf_name = func.get("name", "")
            original_name = self.deobfuscate_name(obf_name)
            new_tc = {
                **tc,
                "function": {
                    **func,
                    "name": original_name,
                },
            }
            result.append(new_tc)
        return result
