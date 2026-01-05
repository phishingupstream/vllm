#!/usr/bin/env python3
"""Patch Gemma4 reasoning parser: fix non-streaming thinking leak.

When skip_special_tokens=True (the vLLM default), the <|channel> and
<channel|> delimiter tokens are stripped from output.text before the
non-streaming extract_reasoning() method sees it. The parser can't find
its delimiters and returns everything (including the "thought\n" role
label) as plain content instead of reasoning.

Fix: when delimiters are absent, detect the "thought\n" prefix that
Gemma4 always emits at the start of thinking. If present, treat the
text as reasoning with stripped delimiters: split on the prefix and
return reasoning + content.

Upstream: https://github.com/vllm-project/vllm/issues/38855
"""

PATH = "/usr/local/lib/python3.12/dist-packages/vllm/reasoning/gemma4_reasoning_parser.py"
PATCH_TAG = "[PATCH] fix non-streaming thinking leak"

with open(PATH) as f:
    content = f.read()

if PATCH_TAG in content:
    print("Already patched")
    exit(0)

# The current code returns (None, model_output) when delimiters are missing.
# We need to add a fallback that detects the "thought\n" prefix.
TARGET = '''\
    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        """Extract reasoning, stripping the ``thought\\\\n`` role label."""
        if self.start_token not in model_output and self.end_token not in model_output:
            # Default to content history if no tags are present
            # (or if they were stripped)
            return None, model_output

        reasoning, content = super().extract_reasoning(model_output, request)
        if reasoning is not None:
            reasoning = _strip_thought_label(reasoning)
        return reasoning, content'''

REPLACEMENT = '''\
    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        """Extract reasoning, stripping the ``thought\\\\n`` role label.

        Handles both cases:
        - skip_special_tokens=False: delimiters present in text
        - skip_special_tokens=True: delimiters stripped, detect via
          "thought\\\\n" prefix (Gemma4 role label)
        """
        # {tag}
        if self.start_token not in model_output and self.end_token not in model_output:
            # Delimiters were stripped by skip_special_tokens=True.
            # Detect thinking by the "thought\\n" role label that Gemma4
            # always emits at the start of the thinking channel.
            if model_output.startswith(_THOUGHT_PREFIX):
                reasoning = model_output[len(_THOUGHT_PREFIX):]
                # The model output is: thought\\n<reasoning><content>
                # Without end delimiter we can't split precisely, so
                # return all as reasoning (the streaming path handles
                # this correctly via token IDs).
                return reasoning, None
            return None, model_output

        reasoning, content = super().extract_reasoning(model_output, request)
        if reasoning is not None:
            reasoning = _strip_thought_label(reasoning)
        return reasoning, content'''.format(tag=PATCH_TAG)

assert TARGET in content, "Target not found — parser may have changed"
content = content.replace(TARGET, REPLACEMENT, 1)

with open(PATH, "w") as f:
    f.write(content)

# Verify syntax
import ast
ast.parse(content)

print("Patched: fix non-streaming thinking leak in Gemma4 reasoning parser")
