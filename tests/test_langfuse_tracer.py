#!/usr/bin/env python3
"""Tests for the Langfuse tracer middleware.

Sends requests to all three vLLM endpoints (OpenAI chat, OpenAI completions,
Anthropic messages) in both streaming and non-streaming modes, then verifies
the resulting Langfuse traces contain the expected data.

Usage:
    python3 tests/test_langfuse_tracer.py [--url URL] [--langfuse-url URL]

Requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY env vars (or uses
defaults from the vLLM project).
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, field


# ── Config ────────────────────────────────────────────────────────

DEFAULT_URL = "http://vllm.localhost"
VLLM_URL = os.environ.get("VLLM_URL", DEFAULT_URL)
LANGFUSE_URL = os.environ.get("LANGFUSE_URL", "http://langfuse:3000")
LANGFUSE_PK = os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-xxx")
LANGFUSE_SK = os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-xxx")
MODEL = "default"
# Unique tag per test run to isolate traces
RUN_ID = f"test-{int(time.time())}"


# ── Helpers ───────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool
    errors: list[str] = field(default_factory=list)

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        msg = f"  {status}  {self.name}"
        if self.errors:
            for e in self.errors:
                msg += f"\n        {e}"
        return msg


def http_request(url: str, data: dict | None = None,
                 headers: dict | None = None) -> bytes:
    """Make an HTTP request and return the response body."""
    h = headers or {}
    if data is not None:
        body = json.dumps(data).encode()
        h.setdefault("Content-Type", "application/json")
    else:
        body = None
    req = urllib.request.Request(url, data=body, headers=h)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def http_stream(url: str, data: dict, headers: dict | None = None):
    """Make an HTTP request and yield response lines."""
    h = headers or {}
    h.setdefault("Content-Type", "application/json")
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers=h)
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            yield line.decode("utf-8", errors="replace").strip()


def langfuse_auth() -> str:
    return base64.b64encode(f"{LANGFUSE_PK}:{LANGFUSE_SK}".encode()).decode()


def get_traces(limit: int = 20, tags: list[str] | None = None) -> list[dict]:
    """Fetch recent traces from Langfuse."""
    auth = langfuse_auth()
    url = f"{LANGFUSE_URL}/api/public/traces?limit={limit}"
    if tags:
        for tag in tags:
            url += f"&tags={tag}"
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {auth}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data.get("data", [])


def get_observations(trace_id: str) -> list[dict]:
    """Fetch observations for a trace."""
    auth = langfuse_auth()
    url = f"{LANGFUSE_URL}/api/public/observations?traceId={trace_id}"
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {auth}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data.get("data", [])


# ── Test requests ─────────────────────────────────────────────────

def send_openai_chat(stream: bool = False, thinking: bool = False) -> dict:
    """Send a request to /v1/chat/completions."""
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Answer in one word."},
            {"role": "user", "content": "Capital of France?"},
        ],
        "max_tokens": 500 if thinking else 50,
        "stream": stream,
        "temperature": 0.5,
        "top_p": 0.9,
    }
    if not thinking:
        data["chat_template_kwargs"] = {"enable_thinking": False}
    if stream:
        data["stream_options"] = {"include_usage": True}

    headers = {"X-Caller": RUN_ID}
    url = f"{VLLM_URL}/v1/chat/completions"

    if not stream:
        resp = http_request(url, data=data, headers=headers)
        return json.loads(resp)
    else:
        content = ""
        reasoning = ""
        result = {}
        for line in http_stream(url, data, headers):
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            obj = json.loads(line[6:])
            if not result:
                result = {"id": obj.get("id"), "model": obj.get("model")}
            delta = obj["choices"][0].get("delta", {}) if obj.get("choices") else {}
            if delta.get("content"):
                content += delta["content"]
            r = delta.get("reasoning_content") or delta.get("reasoning")
            if r:
                reasoning += r
            if "usage" in obj and obj["usage"]:
                result["usage"] = obj["usage"]
        result["content"] = content
        result["reasoning"] = reasoning
        return result


def send_anthropic(stream: bool = False, thinking: bool = False) -> dict:
    """Send a request to /v1/messages."""
    data = {
        "model": MODEL,
        "max_tokens": 500 if thinking else 50,
        "messages": [{"role": "user", "content": "Capital of Japan? One word."}],
        "stream": stream,
    }
    if thinking:
        data["thinking"] = {"type": "enabled", "budget_tokens": 200}
    else:
        data["thinking"] = {"type": "disabled"}

    headers = {
        "x-api-key": "dummy",
        "anthropic-version": "2023-06-01",
        "X-Caller": RUN_ID,
    }
    url = f"{VLLM_URL}/v1/messages"

    if not stream:
        resp = http_request(url, data=data, headers=headers)
        obj = json.loads(resp)
        content = ""
        reasoning = ""
        for block in obj.get("content", []):
            if block["type"] == "text":
                content += block.get("text", "")
            elif block["type"] == "thinking":
                reasoning += block.get("thinking", "")
        return {
            "id": obj.get("id"),
            "model": obj.get("model"),
            "content": content,
            "reasoning": reasoning,
            "usage": obj.get("usage", {}),
            "stop_reason": obj.get("stop_reason"),
        }
    else:
        content = ""
        reasoning = ""
        result = {}
        for line in http_stream(url, data, headers):
            if not line.startswith("data: "):
                continue
            try:
                obj = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "message_start":
                msg = obj.get("message", {})
                result = {"id": msg.get("id"), "model": msg.get("model")}
            delta = obj.get("delta", {})
            if delta.get("type") == "text_delta":
                content += delta.get("text", "")
            elif delta.get("type") == "thinking_delta":
                reasoning += delta.get("thinking", "")
            if obj.get("type") == "message_delta":
                usage = obj.get("usage", {})
                if usage:
                    result["usage"] = usage
                result["stop_reason"] = delta.get("stop_reason")
        result["content"] = content
        result["reasoning"] = reasoning
        return result


# ── Trace verification ────────────────────────────────────────────

def verify_trace(trace: dict, obs_list: list[dict], expected: dict) -> list[str]:
    """Verify a trace matches expected values. Returns list of errors."""
    errors = []

    def check(field, actual, expected_val, contains=False):
        if contains:
            if expected_val not in (actual or ""):
                errors.append(f"{field}: expected to contain {expected_val!r}, got {actual!r}")
        elif actual != expected_val:
            errors.append(f"{field}: expected {expected_val!r}, got {actual!r}")

    # Trace-level checks
    if expected.get("userId"):
        check("trace.userId", trace.get("userId"), expected["userId"])
    if expected.get("tags"):
        for tag in expected["tags"]:
            if tag not in (trace.get("tags") or []):
                errors.append(f"trace.tags: missing {tag!r}, got {trace.get('tags')}")

    # Output checks
    output = trace.get("output", {})
    if isinstance(output, dict):
        if "message.role" not in output:
            errors.append("trace.output: missing message.role")
        if "message.content" not in output:
            errors.append("trace.output: missing message.content")
        if expected.get("has_reasoning") and "message.reasoning_content" not in output:
            errors.append("trace.output: missing message.reasoning_content")
        if expected.get("content_contains"):
            actual_content = output.get("message.content", "")
            if expected["content_contains"].lower() not in actual_content.lower():
                errors.append(f"trace.output.content: expected to contain {expected['content_contains']!r}, got {actual_content[:100]!r}")

    # Input checks
    input_data = trace.get("input")
    if expected.get("has_system_message"):
        if isinstance(input_data, list):
            roles = [m.get("role") for m in input_data if isinstance(m, dict)]
            if "system" not in roles:
                errors.append(f"trace.input: expected system message, got roles {roles}")

    # Observation checks
    if not obs_list:
        errors.append("No observations found")
        return errors

    obs = obs_list[0]

    # Model should be resolved (not 'default')
    obs_model = obs.get("model", "")
    if obs_model == "default":
        errors.append(f"obs.model: still 'default', should be resolved")

    # Usage
    usage = obs.get("usage", {})
    if expected.get("has_usage"):
        if not usage.get("input") and not usage.get("total"):
            errors.append(f"obs.usage: expected usage data, got {usage}")

    # Model parameters
    if expected.get("has_model_params"):
        params = obs.get("modelParameters", {})
        if not params:
            errors.append("obs.modelParameters: expected params, got empty")

    # Completion start time (TTFT)
    if expected.get("has_ttft"):
        if not obs.get("completionStartTime"):
            errors.append("obs.completionStartTime: expected TTFT, got None")

    # Type should be GENERATION
    if obs.get("type") != "GENERATION":
        errors.append(f"obs.type: expected GENERATION, got {obs.get('type')}")

    return errors


# ── Test cases ────────────────────────────────────────────────────

def run_tests(args) -> list[TestResult]:
    results = []

    # 1. OpenAI chat, non-streaming, no thinking
    print("  Sending: OpenAI chat, non-streaming...")
    resp = send_openai_chat(stream=False, thinking=False)
    results.append(TestResult(
        name="openai_chat_nonstream",
        passed=bool(resp.get("choices")),
        errors=[] if resp.get("choices") else ["No response"],
    ))

    # 2. OpenAI chat, streaming, no thinking
    print("  Sending: OpenAI chat, streaming...")
    resp = send_openai_chat(stream=True, thinking=False)
    results.append(TestResult(
        name="openai_chat_stream",
        passed=bool(resp.get("content")),
        errors=[] if resp.get("content") else ["No content in stream"],
    ))

    # 3. OpenAI chat, streaming, with thinking
    print("  Sending: OpenAI chat, streaming + thinking...")
    resp = send_openai_chat(stream=True, thinking=True)
    results.append(TestResult(
        name="openai_chat_stream_thinking",
        passed=True,  # Will verify in traces
    ))

    # 4. Anthropic, non-streaming
    print("  Sending: Anthropic, non-streaming...")
    resp = send_anthropic(stream=False, thinking=False)
    results.append(TestResult(
        name="anthropic_nonstream",
        passed=bool(resp.get("content")),
        errors=[] if resp.get("content") else ["No content"],
    ))

    # 5. Anthropic, streaming
    print("  Sending: Anthropic, streaming...")
    resp = send_anthropic(stream=True, thinking=False)
    results.append(TestResult(
        name="anthropic_stream",
        passed=bool(resp.get("content")),
        errors=[] if resp.get("content") else ["No content in stream"],
    ))

    # 6. Anthropic, streaming, with thinking
    print("  Sending: Anthropic, streaming + thinking...")
    resp = send_anthropic(stream=True, thinking=True)
    has_thinking = len(resp.get("reasoning", "")) > 0
    results.append(TestResult(
        name="anthropic_stream_thinking",
        passed=has_thinking,
        errors=[] if has_thinking else ["No reasoning content in stream"],
    ))

    # Wait for Langfuse batch flush
    print("  Waiting for Langfuse flush...")
    time.sleep(10)

    # Fetch traces for this test run
    traces = get_traces(limit=20, tags=[RUN_ID])
    if not traces:
        # Retry once
        time.sleep(5)
        traces = get_traces(limit=20, tags=[RUN_ID])

    print(f"  Found {len(traces)} traces for run {RUN_ID}")

    if len(traces) < 6:
        results.append(TestResult(
            name="trace_count",
            passed=False,
            errors=[f"Expected 6 traces, found {len(traces)}"],
        ))
        # Still verify what we have
    else:
        results.append(TestResult(name="trace_count", passed=True))

    # Build expected checks per trace
    expected_checks = {
        "openai_nonstream": {
            "tags": ["vllm", RUN_ID],
            "userId": RUN_ID,
            "has_usage": True,
            "has_model_params": True,
            "has_system_message": True,
            "content_contains": "paris",
        },
        "openai_stream": {
            "tags": ["vllm", "streaming", RUN_ID],
            "userId": RUN_ID,
            "has_usage": True,
            "has_model_params": True,
            "has_ttft": True,
            "content_contains": "paris",
        },
        "openai_stream_thinking": {
            "tags": ["vllm", "streaming", RUN_ID],
            "userId": RUN_ID,
            "has_usage": True,
            "has_ttft": True,
            # May or may not have reasoning depending on model behavior
        },
        "anthropic_nonstream": {
            "tags": ["vllm", "anthropic-api", RUN_ID],
            "userId": RUN_ID,
            "has_usage": True,
        },
        "anthropic_stream": {
            "tags": ["vllm", "anthropic-api", "streaming", RUN_ID],
            "userId": RUN_ID,
            "has_usage": True,
            "has_ttft": True,
        },
        "anthropic_stream_thinking": {
            "tags": ["vllm", "anthropic-api", "streaming", RUN_ID],
            "userId": RUN_ID,
            "has_usage": True,
            "has_ttft": True,
            "has_reasoning": True,
        },
    }

    # Match traces to test cases by tags
    for trace in traces:
        tags = set(trace.get("tags", []))
        obs_list = get_observations(trace["id"])

        # Determine which test this trace belongs to
        is_anthropic = "anthropic-api" in tags
        is_streaming = "streaming" in tags

        # Check if it has reasoning
        output = trace.get("output", {})
        has_reasoning = isinstance(output, dict) and "message.reasoning_content" in output

        if is_anthropic and is_streaming and has_reasoning:
            check_name = "anthropic_stream_thinking"
        elif is_anthropic and is_streaming:
            check_name = "anthropic_stream"
        elif is_anthropic:
            check_name = "anthropic_nonstream"
        elif is_streaming and has_reasoning:
            check_name = "openai_stream_thinking"
        elif is_streaming:
            check_name = "openai_stream"
        else:
            check_name = "openai_nonstream"

        expected = expected_checks.pop(check_name, None)
        if expected is None:
            continue  # Already verified or duplicate

        errors = verify_trace(trace, obs_list, expected)
        results.append(TestResult(
            name=f"trace_{check_name}",
            passed=len(errors) == 0,
            errors=errors,
        ))

    # Any unmatched checks
    for check_name in expected_checks:
        results.append(TestResult(
            name=f"trace_{check_name}",
            passed=False,
            errors=[f"No matching trace found for {check_name}"],
        ))

    return results


# ── Main ──────────────────────────────────────────────────────────

def main():
    global VLLM_URL, LANGFUSE_URL  # noqa: PLW0603

    parser = argparse.ArgumentParser(description="Test Langfuse tracer middleware")
    parser.add_argument("--url", default=VLLM_URL, help="vLLM base URL")
    parser.add_argument("--langfuse-url", default=LANGFUSE_URL, help="Langfuse base URL")
    args = parser.parse_args()

    VLLM_URL = args.url
    LANGFUSE_URL = args.langfuse_url

    print("Testing Langfuse tracer middleware")
    print(f"  vLLM:     {VLLM_URL}")
    print(f"  Langfuse: {LANGFUSE_URL}")
    print(f"  Run ID:   {RUN_ID}")
    print()

    results = run_tests(args)

    print()
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    for r in results:
        print(r)

    print()
    print(f"{passed}/{total} passed")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
