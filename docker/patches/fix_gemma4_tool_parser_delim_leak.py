#!/usr/bin/env python3
r"""Patch Gemma4 tool parser: fix string delimiter fragment leak in streamed JSON.

The _emit_argument_diff() method withholds trailing closing chars (}, ", ])
from intermediate JSON to avoid emitting unstable suffixes. But it doesn't
strip partial <|"|> string delimiter chars (<, |, \, >), so when a token
boundary splits the delimiter, fragments leak into the streamed arguments.

Fix: add delimiter chars to the safe_json trailing strip set.

Upstream: https://github.com/vllm-project/vllm/issues/38946
Based on: https://github.com/vllm-project/vllm/pull/38945
"""

PATH = "/usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/gemma4_tool_parser.py"
PATCH_TAG = "[PATCH] fix delimiter fragment leak"

with open(PATH) as f:
    content = f.read()

if PATCH_TAG in content:
    print("Already patched")
    exit(0)

TARGET = '''        # Withhold trailing closing characters that may shift as more
        # tokens arrive. Strip trailing '}', '"', and ']' sequences
        # to get the "safe prefix".
        safe_json = current_args_json
        while safe_json and safe_json[-1] in ("}", '"', "]"):'''

REPLACEMENT = '''        # {tag}
        # Withhold trailing closing characters that may shift as more
        # tokens arrive. Strip trailing '}}', '"', ']' and partial
        # STRING_DELIM fragments ('<', '|', '\\', '>') to get the
        # "safe prefix".
        safe_json = current_args_json
        while safe_json and safe_json[-1] in ("}}", '"', "]", "<", "|", "\\\\", ">"):'''.format(
    tag=PATCH_TAG
)

assert TARGET in content, f"Target not found in {PATH}"
content = content.replace(TARGET, REPLACEMENT, 1)

with open(PATH, "w") as f:
    f.write(content)

print("Patched: fix delimiter fragment leak in Gemma4 streaming tool parser")
