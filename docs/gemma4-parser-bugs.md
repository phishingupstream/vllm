# Gemma 4 parser bugs (reference)

Catalogue of parser bugs discovered while running Gemma 4 26B-A4B-it against
Claude Code sessions, with symptoms, root causes, and the patches that fix
them. Intended as a reference when debugging future Gemma sessions or
upgrading vLLM.

All patches live in `docker/patches/` and are applied at image build time
(see `docker/Dockerfile.runtime`).

## Upstream fix status (as of 2026-04-09)

Upgraded to vLLM v0.19.1rc1.dev133 (main HEAD e80e6339). Most bugs now
fixed upstream — local patches dropped. All tests pass (9/9 validate,
33/33 quality, 4/4 reasoning, 12/12 tool calls, 4/4 quote delimiters).

| Bug | Local patch | Upstream fix | Status |
|-----|-------------|-------------|--------|
| Bug 1 (angle bracket) | `fix_gemma4_tool_parser_angle_bracket.py` | #38909 | **Superseded** |
| Bug 2 (delim leak) | `fix_gemma4_tool_parser_delim_leak.py` | #38992 | **Superseded** |
| Bug 3 (non-streaming thinking) | `fix_gemma4_reasoning_nonstreaming.py` | #39027 | **Superseded** |
| Bug 4 (tool calls swallowed) | `fix_tool_call_during_reasoning.py` | #39027 | **Superseded** |
| Bug 5 (reasoning lost) | `fix_reasoning_lost_on_transition.py` | #34754 | **Superseded** |
| Bug 6 (start token leak) | `fix_reasoning_start_token_leak.py` | #39027 | **Superseded** |
| Bug 7 (delta_message unbound) | `fix_delta_message_unbound.py` | none | **Still applied** |

## Background — Gemma 4 output grammar

Gemma 4 emits a custom structured format using dedicated special tokens:

| Token | ID | Role |
|---|---|---|
| `<|channel>` | 100 | Open reasoning/thinking channel |
| `<channel|>` | 101 | Close reasoning/thinking channel |
| `<|tool_call>` | 48 | Open tool call |
| `<tool_call|>` | 49 | Close tool call |
| `<|"|>` | 52 | String delimiter inside tool call args |
| `<|turn>` | 105 | Open turn |
| `<turn|>` | 106 | Close turn (also an EOS) |
| `<|think|>` | 98 | "Thinking enabled" marker in system prompt |

Typical assistant output:

```
<|channel>thought
...reasoning text...<channel|><|tool_call>call:Bash{command:<|"|>ls<|"|>}<tool_call|><turn|>
```

Tool call args use a custom (non-JSON) serialisation: bare keys, `<|"|>` as
string delimiter, nested `{}`/`[]` for objects/arrays.

Everything the parsers do sits on top of this grammar.

## Bug 1 — Angle bracket doubling in tool args

**Patch:** `fix_gemma4_tool_parser_angle_bracket.py`
**Test:** `tests/test_gemma4_angle_bracket.py`
**Commit:** `85b3c9b` (reproducer), `8255042` (fix)

### Symptom

When the model wrote code containing generics or tags, the characters inside
the angle brackets were duplicated in the tool call output:

```
Vec<u32>       →  Vec<<uu32>
Result<Self>   →  Result<<SelfSelf>
<div>          →  <<ddiv>
Option<u32>    →  Option<<uu32>
```

Observable downstream:

- Claude Code `Write` ops created files with broken Rust / HTML.
- `cargo check` / compilers failed parsing the broken generics.
- The model looped trying to fix files that were corrupted by its own
  tool calls.

### Root cause

The streaming tool parser's `_buffer_delta_text()` buffers trailing chars
that could be the start of `<|tool_call>` or `<tool_call|>` (to handle
multi-token delimiters). A stray `<` at a token boundary landed in the
buffer. In an older revision, the parser reconstructed `current_text` from
the buffer on flush, which caused the buffered `<` to be re-emitted
alongside the upstream stream's `<`. Result: `Vec<u32>` became
`Vec<<uu32>` once the flush diff kicked in.

### Fix

Keep `current_text` from the upstream stream state. Never reconstruct it
from buffered delta text. The buffer is used only to delay emission, not
to synthesise new text.

## Bug 2 — String delimiter fragment leak

**Patch:** `fix_gemma4_tool_parser_delim_leak.py`
**Test:** `tests/test_gemma4_quote_delim.py`
**Commit:** `adffe7e` (reproducer), `8255042` (fix)

### Symptom

Tool call arguments occasionally leaked fragments of the `<|"|>` string
delimiter into streamed JSON deltas:

```
expected: {"content": "Buy milk"}
actual:   {"content": "<|>Buy milk<|>"}  (or similar fragments)
```

When a token boundary split the 5-char `<|"|>` sequence (`<`, `|`, `"`,
`|`, `>`), partial delimiter chars survived past the stripping logic and
ended up in the client's accumulated JSON.

### Root cause

`_emit_argument_diff()` stripped trailing `}`, `"`, `]` from `safe_json`
(to avoid sending closing chars that might move as more tokens arrive)
but did not strip the delimiter chars (`<`, `|`, `\`, `>`). If a partial
delimiter landed at the end of a mid-stream diff, it leaked.

### Fix

Extend the trailing-char strip to include delimiter chars: `{ "}", "\"",
"]", "<", "|", "\\", ">" }`. Leaves a "safe prefix" that never contains
partial delimiter fragments. Final flush happens in
`_handle_tool_call_end()` when `<tool_call|>` arrives.

## Bug 3 — Non-streaming thinking leak

**Patch:** `fix_gemma4_reasoning_nonstreaming.py`
**Test:** `tests/test_gemma4_thinking_leak.py`
**Commit:** `e483b7f` (reproducer), `8255042` (fix)

### Symptom

On the non-streaming `/v1/messages` path, the response returned a single
`thinking` content block that contained BOTH the reasoning text AND the
final answer concatenated together — with no separate `text` block at all.

Example (non-streaming response to "hi"):

```json
{"content": [{
  "type": "thinking",
  "thinking": "The user said \"hi\". Acknowledge... Selected response: \"Hello! How can I help you today?\"Hello! How can I help you today?",
  "signature": "..."
}]}
```

Claude Code sees no `text` block → treats the response as empty → hangs
waiting for a response that never arrives.

### Root cause

The non-streaming chat pipeline calls `tokenizer.decode()` with
`skip_special_tokens=True` (the pydantic default on `ChatCompletionRequest`
which cannot be overridden by `override-generation-config`). Special
tokens — including `<|channel>` and `<channel|>` — are stripped from
`output.text` before `extract_reasoning()` runs. The base parser uses
string-partitioning on `<channel|>` to separate reasoning from content,
so with the delimiters removed it returns everything as reasoning and
None as content.

### Fix

When `extract_reasoning()` can't find either delimiter in the text, look
for Gemma's structural `thought\n` role label at the start of the output
and treat it as the thinking boundary. Also update the base parser's
fallback to handle the stripped case.

The streaming path is unaffected — it operates on token IDs via
`is_reasoning_end(token_ids)`.

## Bug 4 — Tool calls swallowed during reasoning

**Patch:** `fix_tool_call_during_reasoning.py`
**Commit:** `a7807e9`

### Symptom

When Gemma 4 emitted `<|tool_call>` BEFORE `<channel|>` (skipping the
reasoning close), the tool call was swallowed as reasoning text. The
client saw a thinking block with a `<|tool_call>` marker embedded inside
and no corresponding `tool_use` block.

### Root cause

Gemma 4 is free to emit `<|tool_call>` directly (without first emitting
`<channel|>`) when it decides to act without thinking. The serving
pipeline kept `reasoning_end_arr[i] = False` until the reasoning parser
saw `<channel|>`, so all subsequent tokens — including the tool call
payload — were fed to the reasoning parser as reasoning text.

### Fix

Before calling the reasoning parser, check if the tool_parser's
`tool_call_start_token_id` appears anywhere in accumulated output tokens.
If yes, force `reasoning_end_arr[i] = True`, reset `current_text` to the
delta text starting at `<|tool_call>`, and route to the tool parser
directly. The orphan channel is recovered via `<channel|>` appearing
later, but tool call detection no longer requires it.

## Bug 5 — Reasoning content lost on reasoning→content transition

**Patch:** `fix_reasoning_lost_on_transition.py`
**Test:** `tests/test_reasoning_transition_loss.py`
**Commit:** `a21744b`

### Symptom

When the model emitted the final reasoning character(s) AND the
`<channel|>` end-marker in the SAME streaming output step, the trailing
reasoning content was silently dropped. The response had NO thinking
block, and the text block began mid-word.

Example observed in a real session (`3850b2fe-41cc-4c6e-...` L58):

```
Expected text: "End-to-end validation failed because the LLM request..."
Actual text:   "-to-end validation failed because the LLM request..."
Saved thinking block: (none)
```

The model generated `...<channel|>End-to-end validation...` where
`End` and `<channel|>` were adjacent tokens in one engine step.

### Root cause

In `vllm/entrypoints/openai/chat_completion/serving.py`, the
`tool_choice_auto and reasoning_parser` branch has this sequence:

```python
# 1. Reasoning parser correctly splits: reasoning="End", content=None
delta_message = reasoning_parser.extract_reasoning_streaming(...)

# 2. is_reasoning_end fires; current_text reset to ""
if is_reasoning_end(output_token_ids):
    reasoning_end_arr[i] = True
    if delta_message.content:
        current_text = delta_message.content
        delta_message.content = None
    else:
        current_text = ""

# 3. Tool parser called with empty delta_text → returns None
if reasoning_end_arr[i]:
    ...
    delta_message = tool_parser.extract_tool_calls_streaming(...)  # ← overwrites
```

Step 3 overwrites `delta_message`, so `delta_message.reasoning = "End"`
(set by the reasoning parser in step 1) is discarded. The client never
sees it. Subsequent deltas arrive as normal content, so the text block
starts with whatever came after `<channel|>` — hence the mid-word start.

### Fix

Preserve `delta_message.reasoning` across the `tool_parser` call:

```python
_pending_reasoning = (
    delta_message.reasoning
    if delta_message is not None and delta_message.reasoning
    else None
)

delta_message = tool_parser.extract_tool_calls_streaming(...)

if _pending_reasoning is not None:
    if delta_message is None:
        delta_message = DeltaMessage(reasoning=_pending_reasoning)
    else:
        delta_message.reasoning = _pending_reasoning
```

### Why it matters

This bug caused two observable symptoms:

1. Text blocks starting mid-sentence (e.g. `-to-end` instead of
   `End-to-end`). Tracked to session `3850b2fe-...` L58.
2. "Invisible" thinking — the model was reasoning but no thinking block
   was saved because the last reasoning delta (which closed the block)
   never reached the client.

## Bug 6 — Reasoning start_token leaks into thinking content

**Patch:** `fix_reasoning_start_token_leak.py`
**Test:** `tests/test_reasoning_start_token_leak.py`
**Commit:** (this commit)

### Symptom

Thinking blocks in Claude Code sessions contained the literal
`<|channel>` delimiter text at their start. When the model emitted the
start token twice, BOTH leaked:

```
L91 thinking:  "<|channel><|channel>0\n"
L100 thinking: "<|channel>thought\n"
L191 thinking: "<|channel>thought\nThe user says Telegram voice notes..."
```

In some cases the entire reasoning wrapper (`<|channel>thought\n<channel|>`)
ended up in a TEXT block instead of a thinking block.

Observed in real session `b87c2a54-0271-41ad-b48b-4e59c7bbf3c4`.

### Root cause

`BaseThinkingReasoningParser.extract_reasoning_streaming()` has two
branches for when `start_token_id` is in the delta:

1. **start AND end in delta** → strips both correctly.
2. **start in delta, no end** → `return DeltaMessage(reasoning=delta_text)`
   **without stripping the start_token**.

Branch 2 fires when a model emits e.g. `<|channel>thought\n` as a single
engine step (tokens arriving together). The whole `<|channel>thought\n`
string lands in the reasoning field.

`Gemma4ReasoningParser`'s `thought\n` prefix stripper can't recover
because it looks for `thought\n` at position 0, but the buffer starts
with `<|channel>thought\n` — so Case 3 (diverged) kicks in and emits
the leaked delimiter as reasoning.

This affects any parser that subclasses `BaseThinkingReasoningParser`
(Gemma 4, Qwen 3, DeepSeek R1, etc.) when the model emits its start
token together with reasoning text in a single engine step.

### Fix

Strip leading start_token occurrences in the branch-2 path:

```python
reasoning = delta_text
while reasoning.startswith(self.start_token):
    reasoning = reasoning[len(self.start_token):]
return DeltaMessage(reasoning=reasoning) if reasoning else None
```

The `while` loop handles the rare case where the model emits the start
token twice.

## Bug 7 — delta_message UnboundLocalError in streaming after reasoning ends

**Patch:** `fix_delta_message_unbound.py`
**Test:** `tests/test_gemma4_tool_calls.py` (multi-turn with special chars)
**Commit:** `fbc5149`

### Symptom

Multi-turn tool calls with special characters (`<`, `>`, `{`, `}`) in
arguments crashed the OpenAI streaming generator with:

```
UnboundLocalError: cannot access local variable 'delta_message'
where it is not associated with a value
```

The Anthropic streaming path surfaced this as a 500 — the error JSON
body failed Pydantic validation as `ChatCompletionStreamResponse`.

Specifically: the first tool call worked, but the second (after a tool
result was returned) crashed, breaking multi-turn tool sessions.

### Root cause

In `vllm/entrypoints/openai/chat_completion/serving.py`, inside the
`tool_choice_auto and reasoning_parser` branch, `delta_message` is only
assigned inside `if not reasoning_end_arr[i]:`. Once reasoning has ended
(`reasoning_end_arr[i] = True` from a prior chunk), that block is skipped
entirely — but the `if reasoning_end_arr[i]:` block added by Bug 5's fix
reads `delta_message`:

```python
# reasoning already done — block skipped on second+ turns
if not reasoning_end_arr[i]:
    ...
    delta_message = reasoning_parser.extract_reasoning_streaming(...)

# still runs; delta_message was never set on second+ turns
if reasoning_end_arr[i]:
    _pending_reasoning = (
        delta_message.reasoning  # ← UnboundLocalError
        if delta_message is not None and delta_message.reasoning
        else None
    )
```

On the first turn `delta_message` is assigned in the same iteration that
sets `reasoning_end_arr[i] = True`. On subsequent turns `reasoning_end_arr[i]`
is already `True` at iteration start, so the assignment is never reached.

### Fix

Initialise `delta_message = None` immediately before the conditional:

```python
output_token_ids = as_list(output.token_ids)
delta_message = None  # ← added
if not reasoning_end_arr[i]:
    ...
```

## Non-bugs (explained, not fixed)

### Long-context scheduler stalls

Observed symptom: Claude Code request sits in vLLM queue for 3+ minutes
with no engine throughput, then is aborted.

Not a parser bug. When `max_tokens=16384` is sent with a 70K+ prompt and
KV cache budget is 33K tokens (FP8, sliding window config), the scheduler
cannot fit the worst-case `prompt + max_new_tokens` allocation for multi
concurrent requests. Requests sit indefinitely in `Waiting`.

Mitigation: `override-generation-config: {max_new_tokens: 16384}` caps
Claude Code's `max_tokens=32000` server-side. Reducing `max-num-seqs`
(from 4 to 2 or 1) also helps at high context.

### "Corrupted" file content from prior sessions

Observed symptom: model reads a file containing `Result<<SelfSelf>` and
loops trying to fix it.

This was written by a PRE-patch session (before Bug 1 was fixed). The
file on disk remained corrupt. Fix the file manually; the current parser
will not re-introduce the corruption.

### Thinking-block budget cutting sentences mid-word

Observed symptom: thinking ends mid-sentence, content resumes with
`-to-end`.

Can be a legitimate consequence of `thinking_token_budget` hitting mid-
token. But — before the Bug 5 fix — the thinking-end token was ALSO
landing with the cut-off reasoning in the same delta, causing the
reasoning to be dropped entirely rather than just truncated. With Bug 5
fixed, the reasoning is preserved through the cut.

## How to hunt new parser bugs

When you suspect a new parser bug:

1. **Find the session.** `~/.claude/projects/<slug>/<uuid>.jsonl` on the
   host running Claude Code.
2. **Look at the symptom.** Is it in a `thinking` block, `text` block, or
   `tool_use.input`? Does the text start mid-word? Are delimiters
   leaking?
3. **Get the model's raw output.** Send the same prompt to vLLM with
   `/v1/chat/completions`, `stream=true`, and watch the SSE chunks.
   Check each chunk's `delta.reasoning` and `delta.content`.
4. **Isolate the parser.** Run the parser directly against crafted token
   sequences (see `tests/test_gemma4_*.py` for examples using
   `tok.encode(raw_text)` + `extract_*_streaming` in a loop).
5. **Write a test first.** Every bug above has a reproducing test. Add
   yours to `tests/` before fixing.
6. **Patch & deploy.** Add a new file to `docker/patches/`, reference it
   in `Dockerfile.runtime`, apply in-place to the running container
   (`docker exec -u root vllm python3 /tmp/patch.py`), then restart.
7. **Document here.**

## Tests

Run the full Gemma 4 parser test suite:

```bash
python3 tests/test_gemma4_tool_calls.py   # angle brackets, multi-turn, thinking+tools
python3 tests/test_gemma4_quote_delim.py  # string delimiter leak
python3 tests/test_gemma4_reasoning.py    # thinking leak, start-token leak, transition loss
```

Or via the combined runner:

```bash
python3 tests/validate.py --model gemma-4-26b-a4b
```

All tests should pass against a running vLLM container. The
`tool call with thinking enabled` test has ~5% flakiness at high load
(1/20 calls occasionally swallowed at temperature > 0 under concurrency)
— this is a model-sampling artefact, not a code bug.
