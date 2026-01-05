# tests/

## Writing tests

Use `vllm_client.py` for all new tests. It provides `VllmClient` (wraps both OpenAI and Anthropic SDKs) and `run_matrix()` to run a test across all 4 modes automatically.

```python
from vllm_client import VllmClient, argparser, main_exit, run_matrix

def my_test(client, endpoint, stream):
    """endpoint is "openai" or "anthropic". stream is bool."""
    # use client.openai for OpenAI SDK, client.anthropic for Anthropic SDK
    return (passed: bool, detail: str)

args = argparser("description").parse_args()
main_exit(run_matrix(VllmClient(args.url), my_test))
```

Every test must cover all 4 combos: `openai × streaming/non-streaming` and `anthropic × streaming/non-streaming`. This catches bugs that only manifest in one mode (e.g. streaming tool parser corruption doesn't affect non-streaming).

## SDK base URLs

OpenAI client needs `{url}/v1`. Anthropic client needs just `{url}` (it prepends `/v1` itself). `VllmClient` handles this — don't construct clients manually.

## Rules

- Default URL is `http://vllm.localhost` (Caddy proxy, resolves from host). Override with `--url`.
- Use `model="default"` — it aliases whatever model is loaded.
- Tool definitions differ between endpoints. Use `openai_tool_format()` and `anthropic_tool_format()` helpers from `vllm_client.py`.
- For Anthropic tool calls, pass `thinking={"type": "disabled"}` to avoid thinking consuming the token budget.
- Non-streaming is the baseline. If non-streaming passes but streaming fails, the bug is in the streaming parser.
- Keep tests short and focused. One prompt, one assertion. See `test_gemma4_angle_bracket.py` as the reference pattern.
- Tests run from the host, not inside containers.

## Running

```bash
# Single test
python3 tests/test_gemma4_angle_bracket.py

# All validation (9 capability tests against loaded model)
python3 tests/validate.py --model gemma-4-26b-a4b

# Full suite with benchmarks
python3 tests/run.py --suite quick
```
