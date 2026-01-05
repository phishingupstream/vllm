#!/usr/bin/env python3
"""Patch vLLM Anthropic serving: skip empty user messages after tool_result extraction.

When a user message contains only tool_result blocks, _convert_message_content
extracts them into separate "tool" role messages but leaves the original user
message as an empty {"role": "user"} shell. This empty message gets appended to
the OpenAI message list, producing two consecutive user messages (one empty).
Chat templates choke on this — the server goes silent or returns an error.

Fix: only append a message if it has content beyond just the 'role' key.
"""
import sys

PATH = "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/anthropic/serving.py"
PATCH_TAG = "[PATCH] skip empty messages after tool_result extraction"

with open(PATH) as f:
    content = f.read()

if PATCH_TAG in content:
    print("Already patched")
    sys.exit(0)

# ── Target: the unconditional append in _convert_messages ────────────────────
TARGET = """\
            if isinstance(msg.content, str):
                openai_msg["content"] = msg.content
            else:
                cls._convert_message_content(msg, openai_msg, openai_messages)

            openai_messages.append(openai_msg)"""

REPLACEMENT = """\
            if isinstance(msg.content, str):
                openai_msg["content"] = msg.content
            else:
                cls._convert_message_content(msg, openai_msg, openai_messages)

            # [PATCH] skip empty messages after tool_result extraction
            # When all content blocks were tool_results (extracted to separate
            # "tool" messages), the user message shell has only {"role": "user"}
            # with no content. Appending it breaks chat templates.
            if len(openai_msg) > 1:
                openai_messages.append(openai_msg)"""

assert TARGET in content, "Target code block not found in serving.py"
content = content.replace(TARGET, REPLACEMENT, 1)

with open(PATH, "w") as f:
    f.write(content)

print("Patched: skip empty messages after tool_result extraction")
