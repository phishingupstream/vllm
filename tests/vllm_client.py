"""Shared vLLM test client — OpenAI + Anthropic, streaming + non-streaming.

Provides a unified client and a test matrix runner for testing all 4 modes
(openai/anthropic × streaming/non-streaming) with minimal boilerplate.

Usage in tests:
    from vllm_client import VllmClient, run_matrix, argparser

    def my_test(client, endpoint, stream):
        # endpoint is "openai" or "anthropic"
        # return (passed: bool, detail: str)
        ...

    c = VllmClient(args.url)
    results = run_matrix(c, my_test)
"""

import argparse
import json
import sys

from anthropic import Anthropic
from openai import OpenAI

DEFAULT_URL = "http://vllm.localhost"

MODES = [
    ("openai", False),
    ("openai", True),
    ("anthropic", False),
    ("anthropic", True),
]


class VllmClient:
    """Wraps both OpenAI and Anthropic clients for a single vLLM instance."""

    def __init__(self, base_url: str = DEFAULT_URL, timeout: int = 120):
        self.base_url = base_url
        self.openai = OpenAI(base_url=f"{base_url}/v1", api_key="x", timeout=timeout)
        self.anthropic = Anthropic(base_url=base_url, api_key="x")

    def models(self) -> list[str]:
        return [m.id for m in self.openai.models.list().data]


def run_matrix(client: VllmClient, test_fn, modes=None) -> list[tuple[str, bool, str]]:
    """Run test_fn across all 4 endpoint/streaming modes.

    test_fn(client, endpoint, stream) -> (passed: bool, detail: str)

    Returns list of (label, passed, detail) and prints results.
    """
    results = []
    for endpoint, stream in (modes or MODES):
        label = f"{endpoint} {'streaming' if stream else 'non-streaming'}"
        try:
            passed, detail = test_fn(client, endpoint, stream)
        except Exception as e:
            passed, detail = False, str(e)
        tag = "PASS" if passed else "FAIL"
        print(f"[{tag}] {label}" + (f": {detail}" if detail else ""))
        results.append((label, passed, detail))
    return results


def argparser(description: str = "") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--url", default=DEFAULT_URL, help="vLLM base URL")
    return p


def main_exit(results: list[tuple[str, bool, str]]):
    """Print summary and exit."""
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nResults: {passed}/{len(results)} passed")
    sys.exit(0 if passed == len(results) else 1)


def tool_call(client: VllmClient, endpoint: str, stream: bool,
              messages: list, openai_tools: list, anthropic_tools: list,
              thinking: bool = False,
              ) -> dict | None:
    """Make a tool-call request and return parsed args dict of the first tool call.

    Returns None if no tool call was produced.
    """
    if endpoint == "openai":
        extra = {}
        if thinking:
            extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
        if stream:
            chunks: list[str] = []
            for chunk in client.openai.chat.completions.create(
                model="default", messages=messages, max_tokens=1024,
                stream=True, tools=openai_tools, **extra,
            ):
                if not chunk.choices:
                    continue
                for tc in chunk.choices[0].delta.tool_calls or []:
                    if tc.function and tc.function.arguments:
                        chunks.append(tc.function.arguments)
            raw = "".join(chunks)
        else:
            r = client.openai.chat.completions.create(
                model="default", messages=messages, max_tokens=1024,
                tools=openai_tools, **extra,
            )
            tcs = r.choices[0].message.tool_calls
            if not tcs:
                return None
            raw = tcs[0].function.arguments
    else:
        thinking_cfg = {"type": "enabled", "budget_tokens": 2048} if thinking else {"type": "disabled"}
        kwargs = dict(
            model="default", messages=messages, max_tokens=1024,
            tools=anthropic_tools, thinking=thinking_cfg,
        )
        if stream:
            chunks: list[str] = []
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


def chat_text(client: VllmClient, endpoint: str, stream: bool,
              messages: list, max_tokens: int = 256,
              openai_tools: list | None = None,
              anthropic_tools: list | None = None,
              ) -> str:
    """Make a chat request and return the text content."""
    if endpoint == "openai":
        kwargs = dict(
            model="default", messages=messages, max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if openai_tools:
            kwargs["tools"] = openai_tools
        if stream:
            content = ""
            for chunk in client.openai.chat.completions.create(stream=True, **kwargs):
                if not chunk.choices:
                    continue
                content += chunk.choices[0].delta.content or ""
            return content
        else:
            r = client.openai.chat.completions.create(**kwargs)
            return r.choices[0].message.content or ""
    else:
        kwargs = dict(
            model="default", messages=messages, max_tokens=max_tokens,
            thinking={"type": "disabled"},
        )
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        if stream:
            content = ""
            with client.anthropic.messages.stream(**kwargs) as s:
                for event in s:
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        content += event.delta.text
            return content
        else:
            r = client.anthropic.messages.create(**kwargs)
            return "".join(b.text for b in r.content if b.type == "text")


def openai_tool_format(name: str, description: str, properties: dict,
                       required: list[str] | None = None) -> dict:
    """Build an OpenAI-format tool definition."""
    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": schema},
    }


def anthropic_tool_format(name: str, description: str, properties: dict,
                          required: list[str] | None = None) -> dict:
    """Build an Anthropic-format tool definition."""
    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return {"name": name, "description": description, "input_schema": schema}
