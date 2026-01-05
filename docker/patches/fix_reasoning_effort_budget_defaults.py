#!/usr/bin/env python3
"""Patch vLLM v0.19 to map reasoning_effort to thinking_token_budget defaults.

Problem:
    vLLM v0.19 has native reasoning_effort and thinking_token_budget fields,
    but they are independent: reasoning_effort only passes through to the chat
    template as a kwarg, and thinking_token_budget defaults to None (unlimited).

    For Qwen3.5 models, the chat template uses enable_thinking (not
    reasoning_effort) to control whether <think> is emitted. Without mapping,
    reasoning_effort has no effect on Qwen3.5.

    Without a default budget, thinking consumes the entire max_tokens budget
    when used with structured output/tool calling.

Fix:
    Add a model_validator to ChatCompletionRequest that:
    1. Maps reasoning_effort → chat_template_kwargs["enable_thinking"] (for Qwen3.5)
    2. Maps reasoning_effort → thinking_token_budget (convenience defaults)
    3. Defaults to "medium" (8192 budget) when no thinking control is specified

    Mapping (reasoning_effort → thinking_token_budget):
    - "none"   → enable_thinking=false, no budget needed
    - "low"    → enable_thinking=true, thinking_token_budget=2048
    - "medium" → enable_thinking=true, thinking_token_budget=8192
    - "high"   → enable_thinking=true, no budget cap

    Precedence: explicit thinking_token_budget > reasoning_effort > defaults
"""

file_path = "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/protocol.py"

with open(file_path, "r") as f:
    content = f.read()

if "[PATCH] reasoning_effort budget defaults" in content:
    print("Already patched")
    exit(0)

# Add a model_validator BEFORE the existing set_include_reasoning_for_none_effort validator
old_validator = """    @model_validator(mode="before")
    @classmethod
    def set_include_reasoning_for_none_effort(cls, data: Any) -> Any:
        if data.get("reasoning_effort") == "none":
            data["include_reasoning"] = False
        return data"""

new_validator = """    # [PATCH] reasoning_effort budget defaults: map effort to thinking_token_budget
    # and chat_template_kwargs.enable_thinking for Qwen3.5/Gemma4 template compat.
    # Default budget (8192) prevents runaway thinking loops on all requests.
    @model_validator(mode="before")
    @classmethod
    def apply_reasoning_effort_defaults(cls, data: Any) -> Any:
        _DEFAULT_BUDGET = 8192
        effort = data.get("reasoning_effort")
        budget = data.get("thinking_token_budget")
        ctk = data.get("chat_template_kwargs") or {}

        effort_explicitly_set = effort is not None
        if effort_explicitly_set:
            effort_lower = effort.strip().lower() if isinstance(effort, str) else effort
            effort_map = {
                "none": (False, None),
                "low": (True, 2048),
                "medium": (True, _DEFAULT_BUDGET),
                "high": (True, None),
            }
            if effort_lower in effort_map:
                ef_enable, ef_budget = effort_map[effort_lower]
                # Map to enable_thinking for Qwen3.5 template compat
                if "enable_thinking" not in ctk:
                    ctk["enable_thinking"] = ef_enable
                # Set budget from effort level (None = unlimited for "high")
                if budget is None:
                    data["thinking_token_budget"] = ef_budget

        # Default: cap thinking at 8192 when no thinking control was specified.
        # Prevents runaway thinking loops. Only applies when client sent
        # neither reasoning_effort nor thinking_token_budget.
        if not effort_explicitly_set and budget is None:
            data["thinking_token_budget"] = _DEFAULT_BUDGET

        if ctk:
            data["chat_template_kwargs"] = ctk
        return data

    @model_validator(mode="before")
    @classmethod
    def set_include_reasoning_for_none_effort(cls, data: Any) -> Any:
        if data.get("reasoning_effort") == "none":
            data["include_reasoning"] = False
        return data"""

if old_validator not in content:
    print("ERROR: set_include_reasoning_for_none_effort validator not found - vLLM version may differ")
    exit(1)

content = content.replace(old_validator, new_validator, 1)

with open(file_path, "w") as f:
    f.write(content)

print("PATCHED: Added reasoning_effort → thinking_token_budget defaults")
