#!/usr/bin/env python3
"""Test: Gemma 4 tool call parsing — special chars in args and multi-turn.

Two cases:
1. Single-turn: model generates tool call with angle brackets in args
   (Vec<u32>). Checks for streaming '<' doubling bug.
2. Multi-turn: tool_use → tool_result → follow-up with angle brackets
   and curly braces in prior args. Checks message conversion + response.

Usage:
    python3 tests/test_gemma4_tool_calls.py [--url URL]
"""

import json
import re

from vllm_client import (
    VllmClient, argparser, main_exit, run_matrix,
    openai_tool_format, anthropic_tool_format,
    tool_call, chat_text,
)

# ── Tools ────────────────────────────────────────────────────────────────────

WRITE_PROPS = {"file_path": {"type": "string"}, "content": {"type": "string"}}
WRITE_OAI = openai_tool_format("Write", "Write content to a file", WRITE_PROPS, ["file_path", "content"])
WRITE_ANT = anthropic_tool_format("Write", "Write content to a file", WRITE_PROPS, ["file_path", "content"])

SEARCH_PROPS = {"pattern": {"type": "string"}}
SEARCH_OAI = openai_tool_format("search_files", "Search for files", SEARCH_PROPS, ["pattern"])
SEARCH_ANT = anthropic_tool_format("search_files", "Search for files", SEARCH_PROPS, ["pattern"])

# ── Case 1: angle bracket doubling in tool call args ─────────────────────────

RUST_PROMPT = (
    "Use the Write tool to write this exact Rust code to /tmp/test.rs:\n"
    "fn parse(s: &str) -> Result<Vec<u32>, String> {\n"
    '    s.split(",").map(|x| x.trim().parse::<u32>()'
    ".map_err(|e| e.to_string())).collect()\n"
    "}"
)


def test_angle_brackets(client, endpoint, stream):
    """Tool call args with '<' must not be doubled in streaming."""
    args = tool_call(
        client, endpoint, stream,
        messages=[{"role": "user", "content": RUST_PROMPT}],
        openai_tools=[WRITE_OAI], anthropic_tools=[WRITE_ANT],
    )
    if args is None:
        return False, "no tool call produced"
    content = args.get("content", "")
    doubles = [m.group() for m in re.finditer(r"<<([A-Za-z])\1", content)]
    if doubles:
        return False, f"doubled '<' → {doubles}\n  content: {content[:200]}"
    if "Result<Vec<u32>" not in content:
        return False, f"missing expected code\n  content: {content[:200]}"
    return True, ""


# ── Case 2: multi-turn tool_result with special chars ────────────────────────

def test_multi_turn(client, endpoint, stream):
    """Multi-turn with <, { in prior tool args must not break message conversion."""
    if endpoint == "openai":
        messages = [
            {"role": "user", "content": "Search for HashMap<String, {value}> pattern."},
            {"role": "assistant", "tool_calls": [{
                "id": "call_abc123", "type": "function",
                "function": {
                    "name": "search_files",
                    "arguments": json.dumps({"pattern": "HashMap<String, {value}>"}),
                },
            }]},
            {"role": "tool", "tool_call_id": "call_abc123",
             "content": "Found 3 files: foo.py, bar.ts, baz.go"},
            {"role": "user", "content": "How many files were found? Reply with just the number."},
        ]
    else:
        messages = [
            {"role": "user", "content": "Search for HashMap<String, {value}> pattern."},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "call_abc123", "name": "search_files",
                 "input": {"pattern": "HashMap<String, {value}>"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_abc123",
                 "content": "Found 3 files: foo.py, bar.ts, baz.go"},
            ]},
            {"role": "user", "content": "How many files were found? Reply with just the number."},
        ]

    content = chat_text(
        client, endpoint, stream, messages,
        openai_tools=[SEARCH_OAI], anthropic_tools=[SEARCH_ANT],
    ).strip()
    if not content:
        return False, "empty response"
    if "3" in content or "three" in content.lower():
        return True, repr(content[:80])
    return False, f"expected '3', got: {content[:120]!r}"


# ── Case 3: tool call with thinking enabled ──────────────────────────────────

WEATHER_PROPS = {"city": {"type": "string"}}
WEATHER_OAI = openai_tool_format("get_weather", "Get weather for a city", WEATHER_PROPS, ["city"])
WEATHER_ANT = anthropic_tool_format("get_weather", "Get weather for a city", WEATHER_PROPS, ["city"])


def test_tool_call_with_thinking(client, endpoint, stream):
    """Tool call must be extracted even when model thinks first.

    Uses a prompt that forces longer reasoning before the tool call,
    increasing the chance of <|tool_call> landing inside the thinking block.
    Runs 20 attempts since the bug is intermittent (~7% rate).
    """
    prompt = (
        "I need to compare weather in several cities for a trip. "
        "Consider factors like season, latitude, and typical conditions. "
        "Think carefully about which city would be best, then check the weather "
        "for Paris using the get_weather tool."
    )
    misses = 0
    for attempt in range(20):
        args = tool_call(
            client, endpoint, stream,
            messages=[{"role": "user", "content": prompt}],
            openai_tools=[WEATHER_OAI], anthropic_tools=[WEATHER_ANT],
            thinking=True,
        )
        if args is None:
            misses += 1
    if misses:
        return False, f"{misses}/20 tool calls swallowed by thinking"
    return True, "20/20 tool calls produced"


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = argparser("Test Gemma 4 tool call parsing").parse_args()
    print(f"vLLM: {args.url}\n")

    client = VllmClient(args.url)
    results = []

    print("── angle brackets in tool call args ──")
    results.extend(run_matrix(client, test_angle_brackets))

    print("\n── multi-turn with special chars ──")
    results.extend(run_matrix(client, test_multi_turn))

    print("\n── tool call with thinking enabled ──")
    results.extend(run_matrix(client, test_tool_call_with_thinking))

    main_exit(results)
