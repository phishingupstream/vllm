#!/usr/bin/env python3
"""Patch: preserve reasoning content on the reasoning→content transition delta.

Bug:
    When the model emits the last reasoning character AND the reasoning-end
    marker (e.g. <channel|>) in the SAME streaming delta, the reasoning
    content is discarded. The reasoning parser correctly returns
    DeltaMessage(reasoning="X"), but then serving.py calls
    tool_parser.extract_tool_calls_streaming() with an empty delta_text
    (because current_text was reset to ""), which returns None, which
    overwrites delta_message — losing the reasoning payload.

    Observed symptom: Claude Code sessions show text blocks where the
    first word is missing, because the last reasoning token ("End") was
    generated together with <channel|> and silently dropped.

Fix:
    Preserve the reasoning field from delta_message BEFORE the tool_parser
    call, and restore it on the result.
"""

file_path = "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py"

with open(file_path, "r") as f:
    content = f.read()

if "[PATCH] preserve reasoning on transition" in content:
    print("Already patched")
    exit(0)

old_block = """                        # handle tool calls only after reasoning is done,
                        if reasoning_end_arr[i]:
                            delta_token_ids = output_token_ids
                            # First time to tool call,
                            # add the remaining text and token ids
                            # to delta from previous
                            if not added_content_delta_arr[i]:
                                added_content_delta_arr[i] = True
                                previous_text = ""
                                previous_token_ids = []
                                delta_text = current_text
                                delta_token_ids = current_token_ids

                            delta_message = tool_parser.extract_tool_calls_streaming(
                                previous_text=previous_text,
                                current_text=current_text,
                                delta_text=delta_text,
                                previous_token_ids=previous_token_ids,
                                current_token_ids=current_token_ids,
                                delta_token_ids=delta_token_ids,
                                request=request,
                            )
                            if delta_message and delta_message.tool_calls:
                                tools_streamed[i] = True"""

new_block = """                        # handle tool calls only after reasoning is done,
                        if reasoning_end_arr[i]:
                            delta_token_ids = output_token_ids
                            # First time to tool call,
                            # add the remaining text and token ids
                            # to delta from previous
                            if not added_content_delta_arr[i]:
                                added_content_delta_arr[i] = True
                                previous_text = ""
                                previous_token_ids = []
                                delta_text = current_text
                                delta_token_ids = current_token_ids

                            # [PATCH] preserve reasoning on transition:
                            # when the last reasoning char + <channel|> arrive
                            # in the same delta, the reasoning parser returns
                            # DeltaMessage(reasoning="X"). Without this, the
                            # tool_parser call below overwrites delta_message
                            # and the reasoning is silently dropped.
                            _pending_reasoning = (
                                delta_message.reasoning
                                if delta_message is not None and delta_message.reasoning
                                else None
                            )

                            delta_message = tool_parser.extract_tool_calls_streaming(
                                previous_text=previous_text,
                                current_text=current_text,
                                delta_text=delta_text,
                                previous_token_ids=previous_token_ids,
                                current_token_ids=current_token_ids,
                                delta_token_ids=delta_token_ids,
                                request=request,
                            )
                            if delta_message and delta_message.tool_calls:
                                tools_streamed[i] = True

                            # [PATCH] restore the pending reasoning so it
                            # reaches the client on the transition delta.
                            if _pending_reasoning is not None:
                                if delta_message is None:
                                    delta_message = DeltaMessage(
                                        reasoning=_pending_reasoning
                                    )
                                else:
                                    delta_message.reasoning = _pending_reasoning"""

if old_block not in content:
    print("ERROR: target block not found — vLLM version may differ")
    exit(1)

content = content.replace(old_block, new_block, 1)

with open(file_path, "w") as f:
    f.write(content)

print("PATCHED: preserved reasoning on reasoning→content transition delta")
