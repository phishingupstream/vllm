#!/usr/bin/env python3
"""Patch Gemma4 tool parser: fix '<' doubling in streaming tool call args.

The streaming tool parser buffers delta_text to handle multi-token special
sequences, then reconstructs current_text as `previous_text + delta_text`.
This mixes buffered output state with the upstream accumulated text, causing
'<' characters to be doubled: Vec<u32> → Vec<<uu32>, <div> → <<divdiv>.

Fix: keep current_text from the upstream stream state instead of
reconstructing it from the buffered delta.

Upstream: https://github.com/vllm-project/vllm/issues/38910
Based on: https://github.com/vllm-project/vllm/pull/38909
"""

PATH = "/usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/gemma4_tool_parser.py"
PATCH_TAG = "[PATCH] fix angle bracket doubling"

with open(PATH) as f:
    content = f.read()

if PATCH_TAG in content:
    print("Already patched")
    exit(0)

TARGET = """\
        # Buffer delta text to handle multi-token special sequences
        delta_text = self._buffer_delta_text(delta_text)
        # Reconstruct current_text after buffering to stay in sync
        current_text = previous_text + delta_text"""

REPLACEMENT = """\
        # Buffer delta text to handle multi-token special sequences
        delta_text = self._buffer_delta_text(delta_text)
        # {tag}
        # Keep current_text from the upstream stream state. Do NOT
        # reconstruct from buffered delta — that causes '<' chars to be
        # doubled (e.g. Vec<u32> → Vec<<uu32>) because the buffered '<'
        # gets replayed into current_text and the diff logic emits it twice.""".format(
    tag=PATCH_TAG
)

assert TARGET in content, f"Target not found in {PATH}"
content = content.replace(TARGET, REPLACEMENT, 1)

with open(PATH, "w") as f:
    f.write(content)

print("Patched: fix angle bracket doubling in Gemma4 streaming tool parser")
