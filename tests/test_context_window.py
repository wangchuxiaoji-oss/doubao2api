"""
Phase 1: Context Window Capacity Test for doubao-expert with tool calling.

Tests:
1. Input limit: progressively larger prompts with full tool definitions
2. Output limit: ask model to generate long responses
3. Failure behavior: what happens when limit is exceeded

Usage:
    python tests/test_context_window.py --base-url http://103.237.92.203:9090
"""
import argparse
import json
import time
import httpx
import sys

BASE_URL = "http://103.237.92.203:9090"

# Full OpenCode-style tool definitions (English, 8 tools)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the filesystem. Returns file content with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the file"},
                    "offset": {"type": "integer", "description": "Line number to start from (1-indexed)"},
                    "limit": {"type": "integer", "description": "Max lines to read (default 200)"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file by replacing oldString with newString.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the file"},
                    "old_string": {"type": "string", "description": "Exact text to find and replace"},
                    "new_string": {"type": "string", "description": "Replacement text"}
                },
                "required": ["file_path", "old_string", "new_string"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a new file. Overwrites if exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the file"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["file_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents using regex pattern. Returns matching file paths and line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory to search in"},
                    "include": {"type": "string", "description": "File glob pattern filter (e.g. '*.py')"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern. Returns file paths sorted by modification time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g. '**/*.py')"},
                    "path": {"type": "string", "description": "Base directory to search from"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "workdir": {"type": "string", "description": "Working directory"},
                    "timeout": {"type": "integer", "description": "Timeout in milliseconds"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": "Launch a sub-agent to handle complex multi-step tasks autonomously.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Short description of the task"},
                    "prompt": {"type": "string", "description": "Detailed task instructions"},
                    "subagent_type": {"type": "string", "enum": ["explore", "general"], "description": "Agent type"}
                },
                "required": ["description", "prompt", "subagent_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "todowrite",
            "description": "Create and maintain a structured task list. Track progress of multi-step work.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                                "priority": {"type": "string", "enum": ["high", "medium", "low"]}
                            }
                        }
                    }
                },
                "required": ["todos"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "webfetch",
            "description": "Fetch content from a URL and return it in markdown format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "format": {"type": "string", "enum": ["text", "markdown", "html"]}
                },
                "required": ["url"]
            }
        }
    }
]


# Generate realistic code content for padding
def generate_code_padding(char_count: int) -> str:
    """Generate realistic Python code to fill context."""
    snippet = '''def process_data(items: list[dict], config: Config) -> Result:
    """Process a batch of items according to configuration rules."""
    results = []
    for idx, item in enumerate(items):
        if not validate_item(item, config.schema):
            logger.warning(f"Invalid item at index {idx}: {item.get('id')}")
            continue
        transformed = apply_transforms(item, config.transforms)
        if config.dedup_enabled:
            key = compute_hash(transformed, config.dedup_fields)
            if key in seen_hashes:
                metrics.increment("duplicates_skipped")
                continue
            seen_hashes.add(key)
        results.append(transformed)
    return Result(items=results, stats=metrics.snapshot())

'''
    repeats = (char_count // len(snippet)) + 1
    return (snippet * repeats)[:char_count]


def test_input_limit(base_url: str):
    """Test input context limit by sending progressively larger prompts."""
    print("\n" + "=" * 60)
    print("TEST 1: INPUT CONTEXT LIMIT")
    print("=" * 60)

    # Calculate tools overhead
    tools_json = json.dumps(TOOLS)
    print(f"Tools definition size: {len(tools_json):,} chars")

    sizes = [4000, 8000, 16000, 32000, 48000, 64000, 96000, 128000]
    results = []

    for size in sizes:
        code_content = generate_code_padding(size)
        messages = [
            {"role": "user", "content": f"Review this code and identify the main function:\n\n```python\n{code_content}\n```\n\nWhat does process_data do?"}
        ]

        payload = {
            "model": "doubao",
            "stream": False,
            "messages": messages,
            "tools": TOOLS,
        }

        prompt_size = len(json.dumps(payload))
        print(f"\n--- Testing {size:,} chars (payload: {prompt_size:,} chars) ---")

        start = time.time()
        try:
            resp = httpx.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                timeout=120.0,
            )
            elapsed = time.time() - start

            if resp.status_code == 200:
                data = resp.json()
                choice = data["choices"][0]
                content = choice["message"].get("content") or ""
                tool_calls = choice["message"].get("tool_calls")
                finish = choice.get("finish_reason") or choice["finish_reason"]
                print(f"  OK ({elapsed:.1f}s) | finish={finish} | response_len={len(content)} | tool_calls={bool(tool_calls)}")
                results.append({"size": size, "status": "ok", "elapsed": elapsed, "response_len": len(content)})
            else:
                print(f"  HTTP {resp.status_code} ({elapsed:.1f}s): {resp.text[:200]}")
                results.append({"size": size, "status": f"http_{resp.status_code}", "elapsed": elapsed})
        except httpx.TimeoutException:
            elapsed = time.time() - start
            print(f"  TIMEOUT ({elapsed:.1f}s)")
            results.append({"size": size, "status": "timeout", "elapsed": elapsed})
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ERROR ({elapsed:.1f}s): {e}")
            results.append({"size": size, "status": f"error: {e}", "elapsed": elapsed})

    return results


def test_output_limit(base_url: str):
    """Test output generation limit."""
    print("\n" + "=" * 60)
    print("TEST 2: OUTPUT LENGTH LIMIT")
    print("=" * 60)

    prompts = [
        ("short", "Write a hello world function in Python."),
        ("medium", "Write a complete REST API server in Python with FastAPI that has CRUD endpoints for a user management system. Include models, routes, error handling, and authentication middleware. Be thorough and include all code."),
        ("long", "Write a complete implementation of a task queue system in Python. Include: 1) A priority queue with multiple backends (Redis, in-memory, SQLite), 2) Worker pool management with graceful shutdown, 3) Retry logic with exponential backoff, 4) Dead letter queue, 5) Monitoring and metrics collection, 6) Complete type hints and docstrings. Output ALL the code in full, do not abbreviate."),
    ]

    results = []
    for label, prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        payload = {"model": "doubao", "stream": False, "messages": messages, "tools": TOOLS}

        print(f"\n--- Testing output: {label} ---")
        start = time.time()
        try:
            resp = httpx.post(f"{base_url}/v1/chat/completions", json=payload, timeout=180.0)
            elapsed = time.time() - start
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"].get("content") or ""
                print(f"  OK ({elapsed:.1f}s) | output_len={len(content):,} chars | ~{len(content)//4} tokens")
                results.append({"label": label, "output_len": len(content), "elapsed": elapsed})
            else:
                print(f"  HTTP {resp.status_code} ({elapsed:.1f}s)")
                results.append({"label": label, "status": f"http_{resp.status_code}"})
        except httpx.TimeoutException:
            print(f"  TIMEOUT (>{180}s)")
            results.append({"label": label, "status": "timeout"})
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"label": label, "status": f"error: {e}"})

    return results


def test_multi_turn_growth(base_url: str):
    """Test context growth in multi-turn with tool results."""
    print("\n" + "=" * 60)
    print("TEST 3: MULTI-TURN CONTEXT GROWTH")
    print("=" * 60)
    print("Simulates growing message history (like OpenCode would)")

    messages = []
    fake_file_content = generate_code_padding(2000)  # 2K per tool result

    results = []
    for turn in range(1, 21):
        # User asks to read a file
        messages.append({"role": "user", "content": f"Read the file src/module_{turn}.py and summarize it."})
        # Assistant calls read_file
        messages.append({
            "role": "assistant", "content": None,
            "tool_calls": [{"id": f"call_{turn:03d}", "type": "function",
                           "function": {"name": "read_file", "arguments": json.dumps({"file_path": f"/project/src/module_{turn}.py"})}}]
        })
        # Tool result
        messages.append({
            "role": "tool", "tool_call_id": f"call_{turn:03d}",
            "content": f"```python\n# module_{turn}.py\n{fake_file_content}\n```"
        })

        # Now ask model to respond
        payload = {"model": "doubao", "stream": False, "messages": messages, "tools": TOOLS}
        payload_size = len(json.dumps(payload))

        print(f"\n--- Turn {turn} | messages={len(messages)} | payload={payload_size:,} chars ---")

        start = time.time()
        try:
            resp = httpx.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120.0)
            elapsed = time.time() - start
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"].get("content") or ""
                tool_calls = data["choices"][0]["message"].get("tool_calls")
                print(f"  OK ({elapsed:.1f}s) | response_len={len(content)} | tool_calls={bool(tool_calls)}")
                # Add assistant response to history
                messages.append({"role": "assistant", "content": content[:200]})
                results.append({"turn": turn, "payload_size": payload_size, "status": "ok", "elapsed": elapsed})
            else:
                error_text = resp.text[:200]
                print(f"  HTTP {resp.status_code} ({elapsed:.1f}s): {error_text}")
                results.append({"turn": turn, "payload_size": payload_size, "status": f"http_{resp.status_code}", "error": error_text})
                break  # Stop on error
        except httpx.TimeoutException:
            print(f"  TIMEOUT")
            results.append({"turn": turn, "payload_size": payload_size, "status": "timeout"})
            break
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"turn": turn, "payload_size": payload_size, "status": f"error: {e}"})
            break

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Context window capacity test")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--test", choices=["input", "output", "growth", "all"], default="all")
    args = parser.parse_args()

    print(f"Target: {args.base_url}")
    print(f"Tools: {len(TOOLS)} definitions")

    all_results = {}

    if args.test in ("input", "all"):
        all_results["input"] = test_input_limit(args.base_url)

    if args.test in ("output", "all"):
        all_results["output"] = test_output_limit(args.base_url)

    if args.test in ("growth", "all"):
        all_results["growth"] = test_multi_turn_growth(args.base_url)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(json.dumps(all_results, indent=2, default=str))
