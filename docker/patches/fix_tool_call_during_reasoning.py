#!/usr/bin/env python3
"""Patch vLLM streaming: detect tool call tokens during reasoning and hand off.

When both reasoning_parser and tool_parser are active (e.g. Gemma 4 with
enable_thinking + tool calling), the streaming pipeline only sends tokens to
the tool parser AFTER reasoning ends. If the model emits <|tool_call> before
the reasoning end token (<channel|>), the tool call tokens get swallowed by
the reasoning parser as reasoning text and the tool parser never sees them.

Fix: before calling extract_reasoning_streaming, check if the delta contains
the tool call start token. If so, force reasoning to end and route the tool
call text to the tool parser.
"""
import sys

PATH = "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py"
PATCH_TAG = "[PATCH] detect tool call during reasoning"

with open(PATH) as f:
    content = f.read()

if PATCH_TAG in content:
    print("Already patched")
    sys.exit(0)

# ── Target: the else branch that calls reasoning_parser.extract_reasoning_streaming ──
# This is inside the `elif tool_choice_auto and reasoning_parser:` block
TARGET = """\
                            else:
                                delta_message = (
                                    reasoning_parser.extract_reasoning_streaming(
                                        previous_text,
                                        current_text,
                                        delta_text,
                                        previous_token_ids,
                                        current_token_ids,
                                        output_token_ids,
                                    )
                                )

                                # When encountering think end id in delta_token_ids,
                                # set reasoning status to end.
                                # Remove the text and token ids related
                                # to 'reasoning'.
                                if reasoning_parser.is_reasoning_end(output_token_ids):"""

REPLACEMENT = """\
                            else:
                                # [PATCH] detect tool call during reasoning
                                # If tool call start token appears in delta while
                                # still in reasoning, force reasoning end and route
                                # to the tool parser. Without this, models that emit
                                # <|tool_call> before <channel|> (e.g. Gemma 4) have
                                # their tool calls swallowed as reasoning text.
                                _tc_start_id = getattr(tool_parser, 'tool_call_start_token_id', None)
                                if _tc_start_id is not None and _tc_start_id in output_token_ids:
                                    reasoning_end_arr[i] = True
                                    _tc_start_tok = getattr(tool_parser, 'tool_call_start_token', '')
                                    _tc_pos = delta_text.find(_tc_start_tok)
                                    if _tc_pos >= 0:
                                        current_text = delta_text[_tc_pos:]
                                    else:
                                        current_text = delta_text
                                    _tc_idx = output_token_ids.index(_tc_start_id)
                                    current_token_ids = output_token_ids[_tc_idx:]
                                    delta_message = None
                                else:
                                    delta_message = (
                                        reasoning_parser.extract_reasoning_streaming(
                                            previous_text,
                                            current_text,
                                            delta_text,
                                            previous_token_ids,
                                            current_token_ids,
                                            output_token_ids,
                                        )
                                    )

                                # When encountering think end id in delta_token_ids,
                                # set reasoning status to end.
                                # Remove the text and token ids related
                                # to 'reasoning'.
                                if not reasoning_end_arr[i] and reasoning_parser.is_reasoning_end(output_token_ids):"""

assert TARGET in content, "Target code block not found in serving.py"
content = content.replace(TARGET, REPLACEMENT, 1)

with open(PATH, "w") as f:
    f.write(content)

# Verify syntax
import ast
ast.parse(content)

print("Patched: detect tool call tokens during reasoning streaming")
