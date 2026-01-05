#!/usr/bin/env python3
r"""Test: Gemma 4 streaming tool parser leaks string delimiter fragments.

Gemma 4 uses <|""|> (token 52) as string delimiters in tool call args
instead of regular quotes. When a streaming token boundary splits this
delimiter, partial fragments (<|, "|>) leak into the parsed JSON.

The fix (upstream f53fa26 / PR #38992) adds delimiter chars to the
trailing strip set in _emit_argument_diff().

Non-streaming is unaffected — the bug is purely in the streaming
argument diff logic.

Usage:
    python3 tests/test_gemma4_quote_delim.py [--url URL]
"""

import json

from vllm_client import (
    VllmClient, argparser, main_exit, run_matrix,
    openai_tool_format, anthropic_tool_format,
)

TOOL_PROPS = {
    "file_path": {"type": "string"},
    "content": {"type": "string"},
}
OPENAI_TOOL = openai_tool_format("Write", "Write content to a file", TOOL_PROPS, ["file_path", "content"])
ANTHROPIC_TOOL = anthropic_tool_format("Write", "Write content to a file", TOOL_PROPS, ["file_path", "content"])

PROMPT = 'Use the Write tool to write exactly \'print("Hello")\' to /tmp/t.py'


def get_tool_args(client: VllmClient, endpoint: str, stream: bool) -> dict | None:
    """Extract parsed tool call args dict."""
    msg = [{"role": "user", "content": PROMPT}]

    if endpoint == "openai":
        if stream:
            chunks = []
            for chunk in client.openai.chat.completions.create(
                model="default", messages=msg, max_tokens=1024,
                stream=True, tools=[OPENAI_TOOL],
            ):
                if not chunk.choices:
                    continue
                for tc in chunk.choices[0].delta.tool_calls or []:
                    if tc.function and tc.function.arguments:
                        chunks.append(tc.function.arguments)
            raw = "".join(chunks)
        else:
            r = client.openai.chat.completions.create(
                model="default", messages=msg, max_tokens=1024, tools=[OPENAI_TOOL],
            )
            tcs = r.choices[0].message.tool_calls
            if not tcs:
                return None
            raw = tcs[0].function.arguments
    else:
        kwargs = dict(
            model="default", messages=msg, max_tokens=1024,
            tools=[ANTHROPIC_TOOL], thinking={"type": "disabled"},
        )
        if stream:
            chunks = []
            with client.anthropic.messages.stream(**kwargs) as s:
                for event in s:
                    if event.type == "content_block_delta" and event.delta.type == "input_json_delta":
                        chunks.append(event.delta.partial_json)
            raw = "".join(chunks)
        else:
            r = client.anthropic.messages.create(**kwargs)
            tool_blocks = [b for b in r.content if b.type == "tool_use"]
            if not tool_blocks:
                return None
            return dict(tool_blocks[0].input)

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def test_delimiter_leak(client: VllmClient, endpoint: str, stream: bool) -> tuple[bool, str]:
    args = get_tool_args(client, endpoint, stream)
    if args is None:
        return False, "no tool call produced"
    for key, val in args.items():
        if not isinstance(val, str):
            continue
        for frag in ('<|', '"|>', '<|"', '""|>'):
            if frag in val:
                return False, f"{key}: {frag!r} in {val!r}"
    return True, f"{args}"


if __name__ == "__main__":
    args = argparser("Test Gemma 4 string delimiter leak in streaming tool calls").parse_args()
    print(f"vLLM: {args.url}\n")
    main_exit(run_matrix(VllmClient(args.url), test_delimiter_leak))
