#!/usr/bin/env python3
"""Patch vLLM Anthropic serving: escape < > { } in tool call args for Gemma 4.

Gemma 4 uses <|...|> as special tokens in its chat template. Tool call arguments
containing literal < > collide with these tokens. Similarly, { } inside JSON string
values confuse the PEG grammar parser. This patch escapes these characters in string
values only, using JSON unicode escapes (\\u003c etc.) which are lossless on roundtrip.
"""
import ast
import json
import re
import sys

PATH = "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/anthropic/serving.py"
PATCH_TAG = "[PATCH] gemma4 tool args safe-encode"

with open(PATH) as f:
    content = f.read()

if PATCH_TAG in content:
    print("Already patched")
    sys.exit(0)

# ── Helper code to inject ────────────────────────────────────────────────────
# Raw string so backslashes are literal. In serving.py:
#   '\\u003c' is the 6-char string \u003c (a valid JSON unicode escape for <)
#   r'"(?:[^"\\]|\\.)*"' is a raw regex matching JSON string values
HELPER = r"""
# [PATCH] gemma4 tool args safe-encode
# Escapes < > { } inside JSON string values to prevent Gemma 4 special-token
# collisions and PEG grammar confusion. json.loads roundtrip is lossless.
_GEMMA4_STR_RE = re.compile(r'"(?:[^"\\]|\\.)*"')


def _safe_str_escape(m):
    s = m.group(0)
    inner = s[1:-1]
    inner = inner.replace('<', '\\u003c').replace('>', '\\u003e')
    inner = inner.replace('{', '\\u007b').replace('}', '\\u007d')
    return '"' + inner + '"'


def _safe_tool_args_json(obj):
    return _GEMMA4_STR_RE.sub(_safe_str_escape, json.dumps(obj))


"""

# ── Validate helper is correct Python and works ──────────────────────────────
test_code = "import re, json\n" + HELPER
ast.parse(test_code)

ns = {}
exec(compile(test_code, "<helper>", "exec"), ns)
fn = ns["_safe_tool_args_json"]
for test_input in [{"k": "<a>"}, {"k": "{b}"}, {"m": "HashMap<String, {v}>"}]:
    result = fn(test_input)
    assert json.loads(result) == test_input, f"Roundtrip failed: {test_input}"
    assert "<" not in result and ">" not in result, f"Angle brackets in: {result}"
print("Helper self-test: OK")

# ── Add import re if missing ─────────────────────────────────────────────────
if "\nimport re\n" not in content:
    content = content.replace("import json\n", "import json\nimport re\n", 1)

# ── Insert helper before the class ───────────────────────────────────────────
CLASS_MARKER = "class AnthropicServingMessages"
assert CLASS_MARKER in content, "Target class not found"
content = content.replace(CLASS_MARKER, HELPER + CLASS_MARKER, 1)

# ── Replace call site ────────────────────────────────────────────────────────
OLD = '                "arguments": json.dumps(block.input or {}),'
NEW = '                "arguments": _safe_tool_args_json(block.input or {}),'
assert OLD in content, "Target arguments line not found"
content = content.replace(OLD, NEW, 1)

with open(PATH, "w") as f:
    f.write(content)

print("Patched: safe tool arg encoding for Gemma 4")
