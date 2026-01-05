#!/usr/bin/env python3
"""Patch chat_completion serving: initialise delta_message before reasoning_end_arr check.

When reasoning_end_arr[i] is already True from a previous iteration, the
`if not reasoning_end_arr[i]:` block is skipped so delta_message is never
assigned. The following `if reasoning_end_arr[i]:` block (our PATCH for
reasoning-on-transition) then references delta_message, causing:
  UnboundLocalError: cannot access local variable 'delta_message'

Fix: initialise delta_message = None before the conditional so it is always
defined when the tool-call block runs.
"""

PATH = "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py"
PATCH_TAG = "[PATCH] initialise delta_message before reasoning_end check"

with open(PATH) as f:
    content = f.read()

if PATCH_TAG in content:
    print("Already patched")
    exit(0)

TARGET = """\
                        output_token_ids = as_list(output.token_ids)
                        if not reasoning_end_arr[i]:"""

REPLACEMENT = """\
                        output_token_ids = as_list(output.token_ids)
                        delta_message = None  # {tag}
                        if not reasoning_end_arr[i]:""".format(tag=PATCH_TAG)

assert TARGET in content, f"Target not found in {PATH}"
content = content.replace(TARGET, REPLACEMENT, 1)

with open(PATH, "w") as f:
    f.write(content)

print("Patched: initialise delta_message before reasoning_end_arr check in streaming generator")
