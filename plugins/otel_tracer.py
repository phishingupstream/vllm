# SPDX-License-Identifier: Apache-2.0
"""ASGI middleware that captures LLM request/response content and exports
to SigNoz via OpenTelemetry.

Uses GenAI semantic convention attribute names for compatibility with
Langfuse and SigNoz trace rendering.

Supports both OpenAI-compatible (/v1/chat/completions, /v1/completions)
and Anthropic-compatible (/v1/messages) endpoints.

Works for both streaming (SSE) and non-streaming responses by wrapping
the ASGI send callable to accumulate response body content.

Usage:
    vllm serve ... --middleware plugins.otel_tracer.OtelTracerMiddleware

Environment variables:
    SIGNOZ_OTLP_ENDPOINT     - OTLP HTTP endpoint for SigNoz (e.g. http://signoz-otel-collector:4318/v1/traces)
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import extract
from opentelemetry.sdk.trace.export import BatchSpanProcessor


# GenAI semantic convention constants (inline — no openlit dependency)
class SC:
    # Event names
    GEN_AI_SYSTEM_MESSAGE = "gen_ai.system.message"
    GEN_AI_USER_MESSAGE = "gen_ai.user.message"
    GEN_AI_ASSISTANT_MESSAGE = "gen_ai.assistant.message"
    GEN_AI_TOOL_MESSAGE = "gen_ai.tool.message"
    GEN_AI_CHOICE = "gen_ai.choice"
    GEN_AI_CONTENT_PROMPT_EVENT = "gen_ai.content.prompt"
    GEN_AI_CONTENT_COMPLETION_EVENT = "gen_ai.content.completion"
    # Span attributes
    GEN_AI_PROVIDER_NAME = "gen_ai.system"
    GEN_AI_SYSTEM_VLLM = "vLLM"
    GEN_AI_OPERATION = "gen_ai.operation.name"
    GEN_AI_OPERATION_TYPE_CHAT = "chat"
    GEN_AI_OPERATION_TYPE_TEXT_COMPLETION = "text_completion"
    GEN_AI_REQUEST_IS_STREAM = "gen_ai.request.is_stream"
    GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    GEN_AI_RESPONSE_ID = "gen_ai.response.id"
    GEN_AI_RESPONSE_FINISH_REASON = "gen_ai.response.finish_reasons"
    GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
    GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
    GEN_AI_OUTPUT_TYPE = "gen_ai.output.type"
    GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
    GEN_AI_CLIENT_TOKEN_USAGE = "gen_ai.client.token.usage"
    GEN_AI_SERVER_TTFT = "gen_ai.server.ttft"

logger = logging.getLogger("vllm.plugins.otel_tracer")

# Paths we intercept
_OPENAI_PATHS = {"/v1/chat/completions", "/v1/completions"}
_ANTHROPIC_PATH = "/v1/messages"
_ALL_PATHS = _OPENAI_PATHS | {_ANTHROPIC_PATH}

# Map message roles to OTEL GenAI event names
_ROLE_TO_EVENT = {
    "system": SC.GEN_AI_SYSTEM_MESSAGE,
    "user": SC.GEN_AI_USER_MESSAGE,
    "assistant": SC.GEN_AI_ASSISTANT_MESSAGE,
    "tool": SC.GEN_AI_TOOL_MESSAGE,
}

# Known caller patterns: (user-agent regex, tag name)
_CALLER_PATTERNS = [
    (re.compile(r"openclaw", re.I), "openclaw"),
    (re.compile(r"claude[_-]?agent", re.I), "claude-agent"),
    (re.compile(r"claude[_-]?code", re.I), "claude-code"),
    (re.compile(r"langchain", re.I), "langchain"),
    (re.compile(r"openai-python", re.I), "openai-sdk"),
    (re.compile(r"anthropic", re.I), "anthropic-sdk"),
    (re.compile(r"httpx", re.I), "httpx"),
    (re.compile(r"curl", re.I), "curl"),
]

_tracer: trace.Tracer | None = None


def _init_tracer() -> trace.Tracer | None:
    global _tracer

    signoz_endpoint = os.environ.get("SIGNOZ_OTLP_ENDPOINT")
    if not signoz_endpoint:
        logger.warning("SIGNOZ_OTLP_ENDPOINT not set — tracer disabled")
        return None

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider(resource=Resource.create({"service.name": "vllm"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=signoz_endpoint)))
    trace.set_tracer_provider(provider)
    logger.info("OTEL exporter: SigNoz → %s", signoz_endpoint)

    _tracer = trace.get_tracer("vllm-llm-tracer")
    return _tracer


# ── Shared helpers ────────────────────────────────────────────────


def _build_input_messages(body: dict[str, Any], is_anthropic: bool) -> list[dict[str, str]]:
    """Build a simplified messages list for the gen_ai.input.messages span attribute."""
    messages: list[dict[str, str]] = []

    # Anthropic system prompt is a top-level field
    if is_anthropic and "system" in body:
        system = body["system"]
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text = " ".join(b.get("text", "") for b in system if b.get("type") == "text")
            if text:
                messages.append({"role": "system", "content": text})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        tc = block.get("content", "")
                        text_parts.append(tc if isinstance(tc, str) else json.dumps(tc, ensure_ascii=False))
                    elif block.get("type") == "tool_use":
                        text_parts.append(f"[tool_use: {block.get('name', '?')}]")
                else:
                    text_parts.append(str(block))
            text = " ".join(text_parts)
        messages.append({"role": role, "content": text})

    # Completions API — single prompt string
    if "prompt" in body and not body.get("messages"):
        prompt = body["prompt"]
        text = prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False)
        messages.append({"role": "user", "content": text})

    return messages


def _add_input_events(span: trace.Span, body: dict[str, Any], is_anthropic: bool) -> None:
    """Add OTEL GenAI events for each input message (Langfuse chat-style rendering)."""
    if is_anthropic and "system" in body:
        system = body["system"]
        if isinstance(system, str):
            span.add_event(SC.GEN_AI_SYSTEM_MESSAGE, attributes={"content": system})
        elif isinstance(system, list):
            text = " ".join(b.get("text", "") for b in system if b.get("type") == "text")
            if text:
                span.add_event(SC.GEN_AI_SYSTEM_MESSAGE, attributes={"content": text})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        event_name = _ROLE_TO_EVENT.get(role, SC.GEN_AI_USER_MESSAGE)
        attrs: dict[str, str] = {}

        content = msg.get("content")
        if isinstance(content, str):
            attrs["content"] = content
        elif isinstance(content, list):
            text_parts = []
            other_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        tc = block.get("content", "")
                        text_parts.append(tc if isinstance(tc, str) else json.dumps(tc, ensure_ascii=False))
                    elif block.get("type") == "tool_use":
                        other_parts.append(f"[tool_use: {block.get('name', '?')}({json.dumps(block.get('input', {}), ensure_ascii=False)[:200]})]")
                    else:
                        other_parts.append(json.dumps(block, ensure_ascii=False))
                else:
                    text_parts.append(str(block))
            combined = " ".join(text_parts)
            if other_parts:
                combined += " " + " ".join(other_parts) if combined else " ".join(other_parts)
            attrs["content"] = combined.strip()

        if "name" in msg:
            attrs["name"] = msg["name"]
        if "tool_call_id" in msg:
            attrs["tool_call_id"] = msg["tool_call_id"]
        if "tool_calls" in msg:
            attrs["tool_calls"] = json.dumps(msg["tool_calls"], ensure_ascii=False)

        span.add_event(event_name, attributes=attrs)

    # Completions API — single prompt string
    if "prompt" in body and not body.get("messages"):
        prompt = body["prompt"]
        content = prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False)
        span.add_event(SC.GEN_AI_USER_MESSAGE, attributes={"content": content})


def _add_output_event(
    span: trace.Span,
    content: str,
    reasoning: str,
    finish_reason: str = "stop",
) -> None:
    """Add OTEL GenAI choice event for the output (Langfuse rendering)."""
    attrs: dict[str, str] = {
        "finish_reason": finish_reason,
        "message.role": "assistant",
        "message.content": content or "",
    }
    if reasoning:
        attrs["message.reasoning_content"] = reasoning
    span.add_event(SC.GEN_AI_CHOICE, attributes=attrs)


def _parse_sse_chunk(line: str) -> dict[str, Any] | None:
    """Parse an SSE data line into its JSON object."""
    if not line.startswith("data: "):
        return None
    data = line[6:]
    if data == "[DONE]":
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def _identify_caller(scope: dict[str, Any]) -> str | None:
    """Identify the calling service from ASGI scope headers."""
    headers: dict[str, str] = {}
    for k, v in scope.get("headers", []):
        name = k.decode("latin-1") if isinstance(k, bytes) else k
        val = v.decode("latin-1") if isinstance(v, bytes) else v
        headers[name.lower()] = val

    for header in ("x-caller", "x-service-name", "x-client-name"):
        if header in headers:
            return headers[header]

    ua = headers.get("user-agent", "")
    for pattern, name in _CALLER_PATTERNS:
        if pattern.search(ua):
            return name

    return None


def _extract_model_params(body: dict[str, Any]) -> dict[str, str]:
    """Extract sampling/model parameters and vLLM-specific fields from request body."""
    params: dict[str, str] = {}

    # Standard sampling parameters
    for key in ("temperature", "top_p", "top_k", "max_tokens", "min_p",
                "presence_penalty", "frequency_penalty", "repetition_penalty",
                "seed", "stop", "n", "logprobs", "top_logprobs",
                "response_format", "tool_choice", "parallel_tool_calls"):
        if key in body:
            val = body[key]
            params[key] = json.dumps(val, ensure_ascii=False) if isinstance(val, (list, dict)) else str(val)

    if body.get("stream"):
        params["stream"] = "true"

    # Anthropic thinking config
    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        if thinking.get("budget_tokens"):
            params["thinking_budget_tokens"] = str(thinking["budget_tokens"])
        if thinking.get("type"):
            params["thinking_type"] = thinking["type"]

    # OpenAI-style reasoning/thinking controls
    for key in ("reasoning_effort", "enable_thinking"):
        if key in body:
            params[key] = str(body[key])

    # vLLM extra args (max_thinking_tokens, guided_*, etc.)
    extra_body = body.get("extra_body", {})
    if isinstance(extra_body, dict):
        for key, val in extra_body.items():
            params[f"extra.{key}"] = json.dumps(val, ensure_ascii=False) if isinstance(val, (list, dict)) else str(val)

    vllm_xargs = body.get("vllm_xargs", {})
    if isinstance(vllm_xargs, dict):
        for key, val in vllm_xargs.items():
            params[f"vllm.{key}"] = json.dumps(val, ensure_ascii=False) if isinstance(val, (list, dict)) else str(val)

    # Tools (count, not full definitions)
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        params["tools_count"] = str(len(tools))
        params["tool_names"] = json.dumps([t.get("function", {}).get("name", "?") for t in tools if isinstance(t, dict)])

    return params


# ── OpenAI format helpers ─────────────────────────────────────────


def _openai_extract_sse_deltas(obj: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract content and reasoning deltas from an OpenAI SSE chunk."""
    content_parts = []
    reasoning_parts = []
    for choice in obj.get("choices", []):
        delta = choice.get("delta", {})
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            reasoning_parts.append(reasoning)
        if delta.get("content"):
            content_parts.append(delta["content"])
        if choice.get("text"):
            content_parts.append(choice["text"])
    content = "".join(content_parts) if content_parts else None
    reasoning = "".join(reasoning_parts) if reasoning_parts else None
    return content, reasoning


def _openai_parse_response(body: bytes) -> dict[str, Any]:
    """Parse a non-streaming OpenAI response body."""
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return {}

    content_parts = []
    reasoning_parts = []
    finish_reason = "stop"
    for choice in obj.get("choices", []):
        msg = choice.get("message", {})
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        if reasoning:
            reasoning_parts.append(reasoning)
        if msg.get("content"):
            content_parts.append(msg["content"])
        if choice.get("text"):
            content_parts.append(choice["text"])
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

    usage = obj.get("usage", {})
    return {
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "finish_reason": finish_reason,
        "model": obj.get("model"),
        "request_id": obj.get("id"),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


# ── Anthropic format helpers ──────────────────────────────────────


def _anthropic_extract_sse_event(obj: dict[str, Any]) -> dict[str, Any]:
    """Extract data from an Anthropic SSE event."""
    result: dict[str, Any] = {}
    event_type = obj.get("type")

    if event_type == "message_start":
        msg = obj.get("message", {})
        result["model"] = msg.get("model")
        result["request_id"] = msg.get("id")
        usage = msg.get("usage", {})
        if usage.get("input_tokens"):
            result["prompt_tokens"] = usage["input_tokens"]

    elif event_type == "content_block_delta":
        delta = obj.get("delta", {})
        delta_type = delta.get("type")
        if delta_type == "text_delta":
            result["content"] = delta.get("text", "")
        elif delta_type == "thinking_delta":
            result["reasoning"] = delta.get("thinking", "")

    elif event_type == "message_delta":
        delta = obj.get("delta", {})
        if delta.get("stop_reason"):
            result["finish_reason"] = delta["stop_reason"]
        usage = obj.get("usage", {})
        if usage.get("output_tokens"):
            result["completion_tokens"] = usage["output_tokens"]
        if usage.get("input_tokens"):
            result["prompt_tokens"] = usage["input_tokens"]

    return result


def _anthropic_parse_response(body: bytes) -> dict[str, Any]:
    """Parse a non-streaming Anthropic response body."""
    try:
        obj = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(obj, dict):
        return {}

    content_parts = []
    reasoning_parts = []
    for block in obj.get("content", []):
        block_type = block.get("type")
        if block_type == "text":
            content_parts.append(block.get("text", ""))
        elif block_type == "thinking":
            reasoning_parts.append(block.get("thinking", ""))
        elif block_type == "tool_use":
            content_parts.append(
                f"[tool_use: {block.get('name', '?')}({json.dumps(block.get('input', {}), ensure_ascii=False)[:500]})]"
            )

    stop_reason = obj.get("stop_reason", "end_turn")
    finish_map = {"end_turn": "stop", "max_tokens": "length", "tool_use": "tool_calls"}
    finish_reason = finish_map.get(stop_reason, stop_reason)

    usage = obj.get("usage", {})
    return {
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "finish_reason": finish_reason,
        "model": obj.get("model"),
        "request_id": obj.get("id"),
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
    }


# ── Middleware ────────────────────────────────────────────────────


class OtelTracerMiddleware:
    """ASGI middleware that captures LLM prompts and completions."""

    def __init__(self, app):
        self.app = app
        self.tracer = _init_tracer()

    async def __call__(self, scope, receive, send):
        if not self.tracer or scope["type"] != "http":
            return await self.app(scope, receive, send)

        path = scope.get("path", "")
        if path not in _ALL_PATHS:
            return await self.app(scope, receive, send)

        is_anthropic = path == _ANTHROPIC_PATH
        caller = _identify_caller(scope)

        # Extract W3C trace context from incoming headers for e2e trace propagation
        carrier: dict[str, str] = {}
        for k, v in scope.get("headers", []):
            name = k.decode("latin-1") if isinstance(k, bytes) else k
            val = v.decode("latin-1") if isinstance(v, bytes) else v
            carrier[name.lower()] = val
        parent_ctx = extract(carrier)

        body_chunks: list[bytes] = []

        async def receive_wrapper():
            message = await receive()
            if message["type"] == "http.request":
                raw = message.get("body", b"")
                # Inject stream_options.include_usage for streaming requests
                if raw and not is_anthropic:
                    try:
                        body = json.loads(raw)
                        if body.get("stream"):
                            opts = body.setdefault("stream_options", {})
                            opts["include_usage"] = True
                            raw = json.dumps(body).encode()
                            message = dict(message)
                            message["body"] = raw
                    except (json.JSONDecodeError, TypeError):
                        pass
                body_chunks.append(raw)
            return message

        # State for response capture
        request_body: dict[str, Any] = {}
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        response_body_chunks: list[bytes] = []
        streaming_usage: dict[str, Any] = {}
        resolved_model: str | None = None
        request_id: str | None = None
        finish_reason: str = "stop"
        first_content_time: float | None = None
        is_streaming = False
        span: trace.Span | None = None
        start_time = time.time()

        def _start_span():
            nonlocal span, request_body
            try:
                raw = b"".join(body_chunks)
                request_body = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                request_body = {}

            requested_model = request_body.get("model", "unknown")
            is_stream = request_body.get("stream", False)
            op_name = (SC.GEN_AI_OPERATION_TYPE_CHAT
                       if path == "/v1/chat/completions" or is_anthropic
                       else SC.GEN_AI_OPERATION_TYPE_TEXT_COMPLETION)

            span = self.tracer.start_span(
                f"{requested_model}",
                context=parent_ctx,
                attributes={
                    SC.GEN_AI_PROVIDER_NAME: SC.GEN_AI_SYSTEM_VLLM,
                    SC.GEN_AI_OPERATION: op_name,
                    SC.GEN_AI_REQUEST_IS_STREAM: is_stream,
                    # Langfuse-specific
                    "langfuse.observation.type": "generation",
                },
                start_time=int(start_time * 1e9),
            )

            # Input messages as span attribute (OpenLIT list view)
            input_msgs = _build_input_messages(request_body, is_anthropic)
            span.set_attribute(SC.GEN_AI_INPUT_MESSAGES,
                               json.dumps(input_msgs, ensure_ascii=False))

            # Model parameters
            model_params = _extract_model_params(request_body)
            for key, val in model_params.items():
                span.set_attribute(f"gen_ai.request.{key}", val)
            if model_params:
                span.set_attribute("langfuse.observation.model.parameters",
                                   json.dumps(model_params))

            # Tags (Langfuse)
            tags = ["vllm"]
            if is_anthropic:
                tags.append("anthropic-api")
            if is_stream:
                tags.append("streaming")
            if caller:
                tags.append(caller)
            span.set_attribute("langfuse.trace.tags", json.dumps(tags))

            if caller:
                span.set_attribute("user.id", caller)

        async def send_wrapper(message):
            nonlocal is_streaming, span

            if message["type"] == "http.response.start":
                if span is None:
                    _start_span()

                headers = dict(
                    (k.decode() if isinstance(k, bytes) else k,
                     v.decode() if isinstance(v, bytes) else v)
                    for k, v in message.get("headers", [])
                )
                content_type = headers.get("content-type", "")
                is_streaming = "text/event-stream" in content_type

            elif message["type"] == "http.response.body":
                body = message.get("body", b"")

                if is_streaming and body:
                    nonlocal resolved_model, request_id
                    nonlocal finish_reason, first_content_time
                    text = body.decode("utf-8", errors="replace")
                    for line in text.split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        obj = _parse_sse_chunk(line)
                        if obj is None:
                            continue

                        if is_anthropic:
                            _process_anthropic_sse(obj)
                        else:
                            _process_openai_sse(obj)
                else:
                    response_body_chunks.append(body)

                more_body = message.get("more_body", False)
                if not more_body and span is not None:
                    _finalize_span()

            await send(message)

        def _process_openai_sse(obj: dict[str, Any]):
            nonlocal resolved_model, request_id, finish_reason, first_content_time
            content, reasoning = _openai_extract_sse_deltas(obj)
            if content:
                content_parts.append(content)
                if first_content_time is None:
                    first_content_time = time.time()
            if reasoning:
                reasoning_parts.append(reasoning)
                if first_content_time is None:
                    first_content_time = time.time()
            if resolved_model is None and obj.get("model"):
                resolved_model = obj["model"]
            if request_id is None and obj.get("id"):
                request_id = obj["id"]
            for choice in obj.get("choices", []):
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
            if "usage" in obj and obj["usage"]:
                u = obj["usage"]
                streaming_usage["prompt_tokens"] = u.get("prompt_tokens", 0)
                streaming_usage["completion_tokens"] = u.get("completion_tokens", 0)

        def _process_anthropic_sse(obj: dict[str, Any]):
            nonlocal resolved_model, request_id, finish_reason, first_content_time
            extracted = _anthropic_extract_sse_event(obj)
            if "content" in extracted:
                content_parts.append(extracted["content"])
                if first_content_time is None:
                    first_content_time = time.time()
            if "reasoning" in extracted:
                reasoning_parts.append(extracted["reasoning"])
                if first_content_time is None:
                    first_content_time = time.time()
            if "model" in extracted:
                resolved_model = extracted["model"]
            if "request_id" in extracted:
                request_id = extracted["request_id"]
            if "finish_reason" in extracted:
                sr = extracted["finish_reason"]
                finish_map = {"end_turn": "stop", "max_tokens": "length", "tool_use": "tool_calls"}
                finish_reason = finish_map.get(sr, sr)
            if "prompt_tokens" in extracted:
                streaming_usage["prompt_tokens"] = extracted["prompt_tokens"]
            if "completion_tokens" in extracted:
                streaming_usage["completion_tokens"] = extracted["completion_tokens"]

        def _finalize_span():
            if span is None:
                return

            nonlocal finish_reason, resolved_model, request_id
            end_time = time.time()
            prompt_tokens = 0
            completion_tokens = 0

            if is_streaming:
                content = "".join(content_parts)
                reasoning = "".join(reasoning_parts)
                prompt_tokens = streaming_usage.get("prompt_tokens", 0)
                completion_tokens = streaming_usage.get("completion_tokens", 0)
            else:
                full_body = b"".join(response_body_chunks)
                if is_anthropic:
                    parsed = _anthropic_parse_response(full_body)
                else:
                    parsed = _openai_parse_response(full_body)
                content = parsed.get("content", "")
                reasoning = parsed.get("reasoning", "")
                finish_reason = parsed.get("finish_reason", finish_reason)
                prompt_tokens = parsed.get("prompt_tokens", 0)
                completion_tokens = parsed.get("completion_tokens", 0)
                if parsed.get("model"):
                    resolved_model = parsed["model"]
                if parsed.get("request_id"):
                    request_id = parsed["request_id"]

            # Token usage
            total_tokens = prompt_tokens + completion_tokens
            span.set_attribute(SC.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens)
            span.set_attribute(SC.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens)
            span.set_attribute(SC.GEN_AI_USAGE_TOTAL_TOKENS, total_tokens)
            span.set_attribute(SC.GEN_AI_CLIENT_TOKEN_USAGE, total_tokens)
            # Langfuse naming
            span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)
            span.set_attribute("gen_ai.usage.completion_tokens", completion_tokens)

            # Output messages span attribute
            output_msg = {"role": "assistant", "content": content}
            if reasoning:
                output_msg["reasoning_content"] = reasoning
            span.set_attribute(SC.GEN_AI_OUTPUT_MESSAGES,
                               json.dumps([output_msg], ensure_ascii=False))
            span.set_attribute(SC.GEN_AI_RESPONSE_FINISH_REASON,
                               json.dumps([finish_reason]))
            span.set_attribute(SC.GEN_AI_OUTPUT_TYPE, "text")

            # Events — OpenLIT reads indices 0 (prompt) and 1 (completion)
            input_msgs = _build_input_messages(request_body, is_anthropic)
            prompt_parts = []
            for msg in input_msgs:
                role = msg.get("role", "user")
                text = msg.get("content", "")
                if role == "system":
                    prompt_parts.append(f"[System] {text}")
                elif role == "assistant":
                    prompt_parts.append(f"[Assistant] {text}")
                else:
                    prompt_parts.append(text)
            span.add_event(SC.GEN_AI_CONTENT_PROMPT_EVENT, attributes={
                "gen_ai.prompt": "\n\n".join(prompt_parts),
            })
            completion_text = content or ""
            if reasoning and content:
                completion_text = f"[Reasoning]\n{reasoning}\n\n[Response]\n{content}"
            elif reasoning:
                completion_text = reasoning
            span.add_event(SC.GEN_AI_CONTENT_COMPLETION_EVENT, attributes={
                "gen_ai.completion": completion_text,
            })
            # Langfuse chat-style GenAI events (indices 2+)
            _add_input_events(span, request_body, is_anthropic)
            _add_output_event(span, content, reasoning, finish_reason)

            # Model name — prefer resolved, fall back to requested
            model_name = resolved_model or request_body.get("model", "unknown")
            span.set_attribute(SC.GEN_AI_REQUEST_MODEL, model_name)
            if resolved_model:
                span.set_attribute(SC.GEN_AI_RESPONSE_MODEL, resolved_model)
            if request_id:
                span.set_attribute(SC.GEN_AI_RESPONSE_ID, request_id)

            # TTFT
            if first_content_time is not None:
                ttft = round(first_content_time - start_time, 4)
                span.set_attribute(SC.GEN_AI_SERVER_TTFT, ttft)
                ttft_iso = datetime.fromtimestamp(
                    first_content_time, tz=timezone.utc
                ).isoformat()
                span.set_attribute(
                    "langfuse.observation.completion_start_time", ttft_iso
                )

            span.end(end_time=int(end_time * 1e9))

        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        except Exception as exc:
            if span is not None:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(exc))
                span.end()
            raise
