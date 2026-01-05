#!/usr/bin/env python3
"""Patch vLLM chat completion serving to fix Qwen3 reasoning streaming.

When enable_thinking=true, the Qwen3 chat template adds <think> to the
assistant prefix (prompt), so the model generates thinking content without
emitting <think>. The reasoning parser's extract_reasoning_streaming()
never finds <think> in the generated tokens and puts everything in 'content'
instead of 'reasoning_content'.

Fix: After detecting that the prompt has active thinking (prompt_is_reasoning_end
is False), seed all_previous_token_ids with the start_token_id so the parser
knows we're inside a thinking block.
"""
import re

file_path = "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py"

with open(file_path, 'r') as f:
    content = f.read()

if "[PATCH] Qwen3 thinking fix" in content:
    print("Already patched")
    exit(0)

# Match the prompt_is_reasoning_end_arr assignment block with flexible whitespace
pattern = re.compile(
    r'([ ]+)(prompt_is_reasoning_end_arr\[i\] = \(\s+'
    r'reasoning_parser\.is_reasoning_end\(res\.prompt_token_ids\)\s+\))'
)

match = pattern.search(content)
if not match:
    print("WARNING: Pattern not found - vLLM version may be different")
    exit(1)

indent = match.group(1)
original = match.group(0)

patch = f"""{original}
{indent}# [PATCH] Qwen3 thinking fix: when the prompt ends in an
{indent}# active thinking block (<think> without </think>), seed
{indent}# the generated token accumulator with the start_token_id
{indent}# so extract_reasoning_streaming() knows we're in thinking.
{indent}if (
{indent}    not prompt_is_reasoning_end_arr[i]
{indent}    and all_previous_token_ids is not None
{indent}    and hasattr(reasoning_parser, 'start_token_id')
{indent}    and reasoning_parser.start_token_id is not None
{indent}    and not all_previous_token_ids[i]
{indent}):
{indent}    all_previous_token_ids[i] = [reasoning_parser.start_token_id]"""

content = content.replace(original, patch, 1)

with open(file_path, 'w') as f:
    f.write(content)
print("PATCHED: Fixed Qwen3 reasoning streaming in chat_completion/serving.py")
