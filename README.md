# vLLM

vLLM inference server optimized for RTX 5090 (SM120/Blackwell).

## Quick Start

```bash
cd /data/docker

# Build and start (uses cached base image)
docker compose build vllm
docker compose up -d vllm

# Switch models
VLLM_MODEL=nemotron-nano-9b-nvfp4 docker compose up -d vllm
```

## Structure

```
├── docker/           # Dockerfiles and build patches
├── configs/          # Model YAML configs
├── parsers/          # Custom tool call parsers
└── logs/             # Runtime logs
```

## Using the API

Both OpenAI-compatible and Anthropic-compatible endpoints are available via the `vllm` subdomain (HTTPS, routed through Caddy):

| Setting | OpenAI | Anthropic |
|---------|--------|-----------|
| Endpoint | `https://vllm.localhost/v1/chat/completions` | `https://vllm.localhost/v1/messages` |
| Model | `default` | `default` |
| API Key | any non-empty string | any non-empty string |

Direct access on port 8000 also works from the host: `http://localhost:8000/v1/...`

```python
from openai import OpenAI

client = OpenAI(base_url="https://vllm.localhost/v1", api_key="x")
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello"}]
)
```

The Anthropic endpoint returns thinking as `thinking` content blocks (with signatures), matching the real Anthropic API. This enables Claude Code's thinking display when using `claudeqwen`.

Use the model's specific name (e.g. `qwen3.5-9b`) to target a particular model regardless of what's set as default.

### Key parameters

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=1024,        # max tokens to generate
    stream=True,            # stream tokens as they're generated
    # Recommended for thinking mode (Qwen3.5 official params):
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    presence_penalty=0.4,   # keep ≤0.4 with tool calling (higher corrupts JSON)
)
```

### Thinking / reasoning (models with thinking enabled)

Use `reasoning_effort` to control thinking level (OpenAI-compatible, `/v1/chat/completions` only):

```python
# No thinking — fastest, direct answer
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the capital of France?"}],
    extra_body={"reasoning_effort": "none"},
)

# Low thinking — 2048 token budget
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Solve: 12 * 37"}],
    extra_body={"reasoning_effort": "low"},
)

# Medium thinking — 8192 token budget
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Explain quantum entanglement"}],
    extra_body={"reasoning_effort": "medium"},
)

# High thinking — unlimited (model decides when to stop)
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Prove the Pythagorean theorem"}],
    extra_body={"reasoning_effort": "high"},
)
```

| `reasoning_effort` | Thinking | Token Budget |
|---------------------|----------|--------------|
| `none` | off | — |
| `low` | on | 2,048 |
| `medium` | on | 8,192 |
| `high` | on | unlimited |

You can also use `thinking_token_budget` (v0.19 native) for precise control:

```python
# Precise token budget (v0.19 native)
extra_body={"thinking_token_budget": 512}

# Boolean toggle via chat_template_kwargs
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

#### Anthropic endpoint (`/v1/messages`)

The Anthropic endpoint accepts the standard `thinking` configuration:

```python
from anthropic import Anthropic

client = Anthropic(base_url="https://vllm.localhost/v1", api_key="x")

# Enable thinking with token budget
response = client.messages.create(
    model="default",
    max_tokens=4096,
    thinking={"type": "enabled", "budget_tokens": 2048},
    messages=[{"role": "user", "content": "Solve: 12 * 37"}],
)

# Disable thinking
response = client.messages.create(
    model="default",
    max_tokens=4096,
    thinking={"type": "disabled"},
    messages=[{"role": "user", "content": "What's the capital of France?"}],
)
```

#### Reading thinking output

```python
# OpenAI endpoint — reasoning field
msg = response.choices[0].message
print(msg.reasoning)   # thinking content
print(msg.content)     # final answer

# Anthropic endpoint — thinking content blocks
# [{"type": "thinking", "thinking": "...", "signature": "..."}, {"type": "text", "text": "..."}]
```

For multi-turn conversations, pass prior thinking back in `reasoning` (not as plain text) to avoid context bloat. Note: v0.19 renamed the field from `reasoning_content` to `reasoning`.

### Vision / VLM (image understanding)

The default model (`qwen3.5-9b`) has a built-in vision encoder. Pass images via the standard OpenAI multimodal format:

```python
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "file:///data/path/to/image.jpg"}},
            {"type": "text", "text": "Describe this image."},
        ],
    }],
    max_tokens=512,
)
```

Supported image sources:
- **Local files**: `file:///data/...` (must be under `allowed-local-media-path` in model config)
- **URLs**: `https://example.com/image.jpg`
- **Base64**: `data:image/jpeg;base64,...`

## Observability (Langfuse + OpenLIT)

All LLM requests are traced via an ASGI middleware plugin (`plugins/langfuse_tracer.py`) using OTEL GenAI semantic conventions. A single `TracerProvider` exports spans to multiple backends:

- **Langfuse** — Generation observations with chat-style rendering at `https://langfuse.localhost`
- **OpenLIT** — LLM trace analytics and GPU metrics at `https://openlit.localhost`

### Configuration

Set in `docker-compose.yml` environment:
```yaml
# Langfuse
LANGFUSE_OTEL_ENDPOINT=http://langfuse:3000/api/public/otel/v1/traces
LANGFUSE_PUBLIC_KEY=pk-lf-xxx...
LANGFUSE_SECRET_KEY=sk-lf-xxx...
# OpenLIT
OPENLIT_OTLP_ENDPOINT=http://openlit:4318/v1/traces
```

The middleware is loaded via model config YAML:
```yaml
middleware:
  - "plugins.langfuse_tracer.LangfuseTracerMiddleware"
```

Supports all three endpoints: `/v1/chat/completions`, `/v1/completions` (OpenAI), and `/v1/messages` (Anthropic).

### What's captured

| Field | Non-streaming | Streaming |
|-------|--------------|-----------|
| Prompt (messages) | Yes (chat-style rendering) | Yes |
| Completion text | Yes | Yes (accumulated from SSE deltas) |
| Thinking/reasoning | Yes (separate from content) | Yes |
| Model name | Yes (resolved from alias) | Yes |
| Token usage | Yes | Yes |
| TTFT | — | Yes (`completionStartTime`) |
| Model parameters | Yes (temperature, top_p, etc.) | Yes |
| Caller identification | Yes (from headers/User-Agent) | Yes |
| Request ID | Yes (`chatcmpl-*`) | Yes |

Callers are identified via `X-Caller` header (preferred) or User-Agent pattern matching, and appear as both `userId` and a trace tag.

### Disabling

Remove the `middleware` entry from the model config YAML, or unset both `LANGFUSE_OTEL_ENDPOINT` and `OPENLIT_OTLP_ENDPOINT`. The middleware is a no-op when no endpoints are configured.

## Adding Models

1. Create `configs/model-name.yaml` (see existing configs for examples)
2. Start with: `VLLM_MODEL=model-name docker compose up -d vllm`

See `CLAUDE.md` for build details, known issues, and workarounds.

---

## License

This project is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Free for non-commercial use with attribution.
