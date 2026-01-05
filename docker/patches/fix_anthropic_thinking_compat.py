#!/usr/bin/env python3
"""Patch vLLM v0.19 Anthropic endpoint to accept thinking configuration.

Problem:
    Anthropic clients send thinking control as:
        {"thinking": {"type": "enabled", "budget_tokens": 2048}}

    vLLM v0.19's Anthropic endpoint has no thinking field on
    AnthropicMessagesRequest, and _build_base_request doesn't pass
    thinking params to the internal ChatCompletionRequest.

Fix:
    1. Add a `thinking` field to AnthropicMessagesRequest
    2. In _build_base_request, map thinking config to ChatCompletionRequest:
       - thinking.type="enabled" + budget_tokens → thinking_token_budget (native v0.19)
       - thinking.type="disabled" → reasoning_effort="none"

    The apply_reasoning_effort_defaults validator on ChatCompletionRequest
    then handles enable_thinking mapping for Qwen3.5 templates.
"""

# ---- Part 1: Add thinking field to AnthropicMessagesRequest ----

protocol_path = "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/anthropic/protocol.py"

with open(protocol_path, "r") as f:
    content = f.read()

if "[PATCH] anthropic thinking compat" in content:
    print("Already patched (protocol)")
    exit(0)

old_fields = """    top_k: int | None = None
    top_p: float | None = None"""

new_fields = """    top_k: int | None = None
    top_p: float | None = None
    # [PATCH] anthropic thinking compat: accept thinking configuration
    thinking: dict[str, Any] | None = None"""

if old_fields not in content:
    print("ERROR: AnthropicMessagesRequest fields not found - vLLM version may differ")
    exit(1)

content = content.replace(old_fields, new_fields, 1)

# Ensure Any is imported
if "from typing import" in content and "Any" not in content.split("from typing import")[0]:
    pass  # Any should already be imported

with open(protocol_path, "w") as f:
    f.write(content)

print("PATCHED (1/2): Added thinking field to AnthropicMessagesRequest")

# ---- Part 2: Pass thinking config through in _build_base_request ----

serving_path = "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/anthropic/serving.py"

with open(serving_path, "r") as f:
    serving = f.read()

if "[PATCH] anthropic thinking compat" in serving:
    print("Already patched (serving)")
    exit(0)

old_return = """        return ChatCompletionRequest(
            model=anthropic_request.model,
            messages=openai_messages,
            max_tokens=anthropic_request.max_tokens,
            max_completion_tokens=anthropic_request.max_tokens,
            stop=anthropic_request.stop_sequences,
            temperature=anthropic_request.temperature,
            top_p=anthropic_request.top_p,
            top_k=anthropic_request.top_k,
            kv_transfer_params=anthropic_request.kv_transfer_params,
        )"""

new_return = """        # [PATCH] anthropic thinking compat: map thinking config to
        # native v0.19 thinking_token_budget and reasoning_effort.
        thinking_token_budget = None
        reasoning_effort = None
        thinking = getattr(anthropic_request, 'thinking', None)
        if thinking and isinstance(thinking, dict):
            thinking_type = thinking.get('type')
            if thinking_type == 'enabled':
                budget = thinking.get('budget_tokens')
                max_tok = anthropic_request.max_tokens or 0
                # Anthropic clients set budget_tokens = max_tokens - 1 to mean
                # "no cap" (thinking shares the max_tokens pool in real Anthropic API).
                # In vLLM, thinking_token_budget is a separate cap, so passing
                # through a near-max budget defeats the processor. Only forward
                # genuinely restrictive budgets; otherwise the
                # apply_reasoning_effort_defaults validator applies the default.
                if (budget is not None and isinstance(budget, int)
                        and budget > 0 and budget < max_tok * 0.8):
                    thinking_token_budget = budget
            elif thinking_type == 'disabled':
                reasoning_effort = 'none'

        req_kwargs: dict = dict(
            model=anthropic_request.model,
            messages=openai_messages,
            max_tokens=anthropic_request.max_tokens,
            max_completion_tokens=anthropic_request.max_tokens,
            stop=anthropic_request.stop_sequences,
            temperature=anthropic_request.temperature,
            top_p=anthropic_request.top_p,
            top_k=anthropic_request.top_k,
            kv_transfer_params=anthropic_request.kv_transfer_params,
        )
        if thinking_token_budget is not None:
            req_kwargs['thinking_token_budget'] = thinking_token_budget
        if reasoning_effort is not None:
            req_kwargs['reasoning_effort'] = reasoning_effort

        return ChatCompletionRequest(**req_kwargs)"""

if old_return not in serving:
    print("ERROR: _build_base_request return not found - vLLM version may differ")
    exit(1)

serving = serving.replace(old_return, new_return, 1)

with open(serving_path, "w") as f:
    f.write(serving)

print("PATCHED (2/2): Added thinking config passthrough in _build_base_request")
