#!/usr/bin/env python3
"""Patch: strip start_token from delta_text in BaseThinkingReasoningParser.

Bug:
    When the reasoning start token (e.g. <|channel>) arrives in the SAME
    streaming delta as reasoning content (e.g. "<|channel>thought\\n"),
    the base parser returns DeltaMessage(reasoning=delta_text) — WITH
    the literal start token text still included. The client receives
    thinking blocks containing "<|channel>thought\\n..." rather than the
    clean reasoning text.

    Also handles the rare case where the model emits the start token
    twice (e.g. "<|channel><|channel>0\\n") — we strip the first
    occurrence; any remaining leaks would indicate genuinely odd model
    output.

    Observed symptoms (Gemma 4 session b87c2a54-...):
      - L91 thinking: "<|channel><|channel>0\\n"
      - L100 thinking: "<|channel>thought\\n"
      - L191 thinking: "<|channel>thought\\nThe user says..."

    Downstream effect: Gemma4ReasoningParser's "thought\\n" prefix
    stripper couldn't match because the buffer started with "<|channel>"
    instead of "thought\\n", so Case 3 (diverged) kicked in and emitted
    the entire leaked string.

Fix:
    In both branches where start_token lands in delta_text without (yet)
    the end_token, find the start_token position and strip everything
    up-to-and-including it. Return None if the resulting reasoning is
    empty (avoids emitting zero-length deltas).

Affects:
    BaseThinkingReasoningParser — used by Gemma 4, Qwen3, DeepSeek R1,
    and any other parser that subclasses it.
"""

file_path = "/usr/local/lib/python3.12/dist-packages/vllm/reasoning/basic_parsers.py"

with open(file_path, "r") as f:
    content = f.read()

if "[PATCH] strip start_token from delta_text" in content:
    print("Already patched")
    exit(0)

old_block = """        elif self.start_token_id in delta_token_ids:
            if self.end_token_id in delta_token_ids:
                # start token in delta, end token in delta,
                # extract reasoning content
                start_index = delta_text.find(self.start_token)
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[start_index + len(self.start_token) : end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return DeltaMessage(
                    reasoning=reasoning, content=content if content else None
                )
            else:
                # start token in delta, no end token in delta,
                # reasoning content continues
                return DeltaMessage(reasoning=delta_text)"""

new_block = """        elif self.start_token_id in delta_token_ids:
            if self.end_token_id in delta_token_ids:
                # start token in delta, end token in delta,
                # extract reasoning content
                start_index = delta_text.find(self.start_token)
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[start_index + len(self.start_token) : end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return DeltaMessage(
                    reasoning=reasoning, content=content if content else None
                )
            else:
                # [PATCH] strip start_token from delta_text
                # start token in delta, no end token in delta:
                # peel any leading start_token occurrences from the
                # returned reasoning. Without this, models that emit
                # <|channel> + reasoning text in a single delta step
                # leak the literal "<|channel>" into the thinking block
                # (and a doubled <|channel><|channel> leaks both).
                reasoning = delta_text
                while reasoning.startswith(self.start_token):
                    reasoning = reasoning[len(self.start_token):]
                return (
                    DeltaMessage(reasoning=reasoning) if reasoning else None
                )"""

if old_block not in content:
    print("ERROR: target block not found — vLLM version may differ")
    exit(1)

content = content.replace(old_block, new_block, 1)

with open(file_path, "w") as f:
    f.write(content)

print("PATCHED: strip start_token from delta_text in BaseThinkingReasoningParser")
