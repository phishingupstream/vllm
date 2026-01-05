# CLAUDE.md

@README.md

Guidance for Claude Code when working with this vLLM setup.

> **Operational commands** (status, switch model, restart, bench) are available via `/vllm` skill from any project.
> **Unified test runner**: `python3 tests/run.py --suite quick|standard|full [--skip-quality] [--skip-validate]`
> **Validate only**: `python3 tests/validate.py --model <config-name>` — see `tests/models.yaml` for per-model capability definitions.
> **Quality tests only**: `python3 tests/test_model_quality.py --url http://localhost:8000`
> **Langfuse tracer tests**: `python3 tests/test_langfuse_tracer.py` — tests all endpoints (OpenAI + Anthropic), streaming/non-streaming, thinking, then verifies Langfuse traces.

## Architecture

**Current version:** vLLM v0.19.1rc1.dev133 (main HEAD e80e6339, upgraded from v0.19.0 for upstream Gemma 4 parser fixes). Requires Transformers >= 5.5.0. Built 2026-04-09. **Note:** Must build from source (`VLLM_USE_PRECOMPILED=0`) — NGC PyTorch 2.11.0a0 ABI incompatible with stock precompiled wheels.
**Base image:** `base:cuda131` (from `nvcr.io/nvidia/pytorch:26.02-py3` — CUDA 13.1, PyTorch 2.11)

**Two-stage build:**
- `vllm:sm120-26.02` — CUDA extensions only: causal-conv1d, mamba-ssm compiled for SM120. ~5 min rebuild. FlashInfer no longer built from source (prebuilt wheels since v0.6.7).
- `vllm:runtime` — Standard base:cuda131 image + SM120 extensions copied from sm120 build + FlashInfer 0.6.7.post2 prebuilt wheels + vLLM v0.19.0 (Python-only, VLLM_USE_PRECOMPILED=1) + patches. ~3 min rebuild.

**FlashInfer 0.6.7.post2** — installed via prebuilt wheels (cu130 jit-cache, ~2 GiB, 822 prebuilt .so files including 5 SM120 modules). The cu130 wheel works on CUDA 13.1. Replaces the ~90 min source build from v0.6.4. New features: NVFP4 KV decode on SM120, NVFP4 MoE kernels (`fused_moe_120.so`), SM121 detection fix, CUTLASS 4.4, autotuner caching. Prebuilt cubin package still lacks SM120 FMHA kernels (PR #2885 pending) — attention falls back to TRITON_ATTN or FA2 for SM120.

**Why source builds still needed:** causal-conv1d and mamba-ssm don't ship SM120 wheels — must compile from source. vLLM itself ships precompiled wheels with SM120 cubins.

**Why v0.19.0 (upgraded from main/v0.18):** v0.19.0 adds native `thinking_token_budget` and `reasoning_effort` on the OpenAI-compatible API, replacing the custom `ThinkingBudgetLogitsProcessor`. Also adds `reasoning-config` for server-side reasoning parser configuration. **Breaking change:** `reasoning_content` field renamed to `reasoning` in API responses. Three active patches remain for SM120-specific and custom features. The custom logits processor (`plugins/thinking_budget_processor.py`) is deprecated — kept for reference but no longer loaded via model configs.

**v0.19 memory profiling regression — DeltaNet models:** v0.19's `determine_available_memory()` records the Triton autotuner PEAK allocation (~6 GiB transient workspace for DeltaNet `gdn_attention_core`) as permanent overhead, leaving negative KV budget. v0.18 measured available memory AFTER the autotuner workspace was freed. **Fix:** use `kv-cache-memory-bytes` to bypass profiling and set KV cache size explicitly. The 3.0 GiB value matches v0.18 steady-state. This workaround is needed for all DeltaNet hybrid models (Qwen3.5-35B, potentially Qwen3.5-9B).

## Model Cards (`model-cards/`)

HuggingFace model cards saved locally with architecture details, config.json, layer types, attention patterns, special tokens, sampling params, and benchmarks. **Check these before writing or modifying model configs** — they have the authoritative architecture specs (head dims, KV heads, sliding window sizes, MoE expert counts, etc.).

## Model Compatibility Matrix

| Config | Model | Quant | Status | Notes |
|--------|-------|-------|--------|-------|
| `nemotron-30b-a3b-nvfp4` | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 | NVFP4 | ✅ Default | Mamba hybrid, prefix-caching off. nano_v3 parser. qwen3_coder tool parser |
| `nemotron-nano-9b-nvfp4` | nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4 | NVFP4 | ✅ Working | Same Mamba constraints. nemotron_json tool parser (streaming variant) |
| `qwen3-14b-fp8` | Qwen/Qwen3-14B-FP8 | FP8 | ✅ Fallback | Thinking disabled. prefix-caching works |
| `qwen3-30b-a3b-nvfp4` | nvidia/Qwen3-30B-A3B-FP4 | NVFP4 MoE | ✅ Working | enforce-eager. Benchmarked 1,271 tok/s @ concurrency 32 |
| `qwen3.5-35b-a3b-nvfp4-nvidia` | txn545/Qwen3.5-35B-A3B-NVFP4 | NVFP4 MoE | ✅ **Active** | Nvidia official quant. **v0.19 workaround:** `kv-cache-memory-bytes: 3221225472` bypasses broken memory profiling (Triton autotuner peak recorded as permanent). 262K context, CUDA graphs [1,2]. Native thinking_token_budget. pp=0.4 recommended |
| `qwen3.5-35b-a3b-nvfp4` | Kbenkhaled/Qwen3.5-35B-A3B-NVFP4 | NVFP4 MoE | ✅ Fallback | 35B/3B-active MoE (256 experts). DeltaNet hybrid (30 linear + 10 full attn). **262K context** (native max, single-seq, 0.86 util). Tested up to 261K input tokens. prefix-caching off. qwen3_coder tool parser. Multimodal. Benchmarked: 22K tok/s prefill @10K, 13K @86K, 10K @129K, 7.6K @200K, 6K @261K. Decode: 160 tok/s. CUDA graphs [1,2] |
| `qwen3.5-9b` | Qwen/Qwen3.5-9B | BF16 | ✅ Working | Multimodal VLM (image+video). Linear attention hybrid (24 linear + 8 full). prefix-caching off. gpu-memory-utilization 0.88 (gdn_attention_core Triton workspace outside PyTorch pool). mm-encoder-attn-backend TORCH_SDPA (flash_attn PTX incompatible on SM120). allowed-local-media-path /data. nano_v3 parser. Benchmarked: 12.5K tok/s prefill, 95/174/333 tok/s decode @ conc 1/2/4. Effective context ~28K tokens |
| `gemma-4-26b-a4b` | google/gemma-4-26B-A4B-it | FP8 | ✅ **Active** | MoE (128 experts, top-8, 3.8B active). Hybrid attn: 25 sliding (1024) + 5 full. Server-side FP8 (25.65 GiB). TRITON_ATTN (FLASHINFER rejects head_dim 512). gemma4 tool+reasoning parser. Multimodal. **262K context** (0.97 util, 3.8 GiB KV, 33K tokens). `max_new_tokens: 16384` caps output (Claude Code sends 32K). torch compile + CUDA graphs [1,2,4]. Sweep: **seqs=4 batch=4096 → 418 tok/s, TPOT 8.9ms**. Quality: 100% (33/33). 2026-04-04 |
| `gemma-4-31b-nvfp4` | nvidia/Gemma-4-31B-IT-NVFP4 | NVFP4 | ❌ OOM | Dense 30.7B. Model loads at 30.50 GiB (mixed BF16/FP8/FP4 tensors — vision encoder, embeddings, layer norms not quantized). Exceeds 31.36 GiB GPU. Same issue as qwen3-32b-fp4 |
| `gemma-3-12b` | google/gemma-3-12b-it | BF16 | ✅ Working | Multimodal. gpu-memory-utilization 0.85 for vision encoder headroom |
| `gemma-3-12b-it-abliterated` | huihui-ai/gemma-3-12b-it-abliterated | BF16 | ✅ Working | Abliterated variant. hermes tool parser |
| `huihui-qwen3-vl-8b-instruct` | huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated | BF16 | ✅ Working | VLM (image+text). enforce-eager (FlashInfer GQA warmup shape mismatch). VIT attn: TRITON_ATTN (FA2 PTX broken). Main attn: FLASHINFER. qwen3 reasoning parser. hermes tool parser. max-model-len 8k. allowed-local-media-path /data |
| `devstral-small-fp8` | mistralai/Devstral-Small-2-24B-Instruct | BF16 | ✅ Working | FP8 has cuBLAS errors on SM120 → run BF16. enforce-eager. mistral tool parser |
| `nemotron-flash-3b-instruct` | nvidia/Nemotron-Flash-3B-Instruct | BF16 | ✅ Working | Small/fast. mamba-ssm-cache-dtype float32 |
| `qwen3-0-6b-embedding` | Qwen/Qwen3-Embedding-0.6B | F16 | ✅ Working | Separate container (official image, NOT sm120). enforce-eager. truncate_prompt_tokens 512 in API |
| `gpt-oss-20b` | openai/gpt-oss-20b | MXFP4 | ❌ Broken | All 4 MXFP4 backends fail on SM120. CUDA 12.x kernels incompatible with CUDA 13 |
| `qwen3.5-27b-nvfp4` | txn545/Qwen3.5-27B-NVFP4 | NVFP4 dense | ❌ Broken | FLA (linear attention) tensor layout mismatch on this vLLM build: seq_len < num_heads warning, garbage output. Re-test after upstream FLA fix |
| `qwen3-32b-fp4` | nvidia/Qwen3-32B-FP4 | NVFP4 dense | ⚠️ Impractical | 19GB model → only 8K context on 32GB VRAM |
| — | huihui-ai/Huihui-Qwen3-30B-A3B (Qwen HF) | NVFP4 MoE | ❌ Broken | experts_cls=None assertion in vLLM 0.16. Use nvidia/Qwen3-30B-A3B-FP4 instead |

**In cache, not yet configured:** Nemotron-Nano-12B-v2, Nemotron-Nano-12B-v2-VL-NVFP4-QAD, Nemotron-Flash-1B, Huihui-Qwen3-VL-30B-A3B, Huihui-Qwen3-VL-32B-Thinking, Huihui-gpt-oss-20b-abliterated

## Patches (`docker/patches/`)

Gemma 4 parser bugs have their own deep-dive doc: [`docs/gemma4-parser-bugs.md`](docs/gemma4-parser-bugs.md)
— symptoms, root causes, fixes, and a guide for hunting new ones.

| Patch | Status | Notes |
|-------|--------|-------|
| `fix_opaque_base_ngc.py` | ✅ Applied | NGC PyTorch 2.11 doesn't ship `torch._opaque_base` private API. Wraps import in try/except |
| `fix_mxfp4_sm120.py` | ⚠️ Pattern not found | v0.19 may have changed MXFP4 backend selection. GPT-OSS-20B still broken on SM120 regardless |
| `fix_qwen3_reasoning_streaming.py` | ⬜ Superseded v0.19.1 | Upstream #34779 fixed `<think>` in prompt prefix detection |
| `fix_reasoning_effort_budget_defaults.py` | ✅ Applied | Maps `reasoning_effort` → `thinking_token_budget` defaults + `enable_thinking` for Qwen3.5/Gemma4 template compat. Only sets budget when effort explicitly provided |
| `fix_anthropic_thinking_compat.py` | ✅ Applied | Accepts `thinking` config on Anthropic `/v1/messages` endpoint. Maps to native `thinking_token_budget`. Filters near-max `budget_tokens` (≥80% of `max_tokens`) |
| `fix_tool_call_during_reasoning.py` | ⬜ Superseded v0.19.1 | Upstream #39027 fixes `is_reasoning_end` for Gemma 4 |
| `fix_gemma4_tool_parser_angle_bracket.py` | ⬜ Superseded v0.19.1 | Upstream #38909 fixes `current_text` reconstruction |
| `fix_gemma4_tool_parser_delim_leak.py` | ⬜ Superseded v0.19.1 | Upstream #38992 strips partial delimiters |
| `fix_gemma4_reasoning_nonstreaming.py` | ⬜ Superseded v0.19.1 | Upstream #39027 uses `adjust_request()` to set `skip_special_tokens=False` |
| `fix_reasoning_lost_on_transition.py` | ⬜ Superseded v0.19.1 | Upstream #34754 fixes streaming event processing |
| `fix_reasoning_start_token_leak.py` | ⬜ Superseded v0.19.1 | Upstream #39027 comprehensive reasoning parser fix |
| `fix_anthropic_empty_tool_message.py` | ⬜ Superseded v0.19.1 | Upstream added `if not (msg.role == "user" and "content" not in openai_msg)` guard |
| `fix_enable_thinking_compat.py` | ⬜ Obsolete v0.19 | Replaced by native `reasoning_effort` + `thinking_token_budget` on ChatCompletionRequest + `fix_reasoning_effort_budget_defaults.py` |
| `fix_nvfp4_experts_cls.py` | ⬜ Obsolete v0.17 | Upstream rewrote MoE oracle, bug eliminated |
| `fix_vit_attn_sm120.py` | ⬜ Obsolete v0.17 | VIT crash handled by `mm-encoder-attn-backend: TORCH_SDPA` in model config |
| `fix_torch_compile.py` | ⬜ Obsolete v0.17 | Upstream fixed version check |

## Environment Variables

| Variable | Value | Status |
|----------|-------|--------|
| `VLLM_USE_V1` | — | ⬜ **Removed** — V1 is the default engine in v0.17, env var no longer exists |
| `VLLM_ATTENTION_BACKEND` | — | ⬜ **Removed** — v0.17 auto-selects FLASHINFER on SM120. Use `attention-config` in model YAML if needed |
| `VLLM_USE_FLASHINFER_MOE_FP4` | `1` | ✅ Required — NVFP4 MoE. Auto-selects `flashinfer_cutlass` MoE backend (best on SM120). `flashinfer_trtllm`/`cutedsl` crash ("unsupported device"), `vllm cutlass` is 8x slower. Tested 2026-03-13 |
| `VLLM_NVFP4_GEMM_BACKEND` | `flashinfer-cutlass` | ✅ Required — NVFP4 linear layers. All flashinfer variants perform identically on MoE models (MoE backend dominates). Tested 2026-03-13 |
| `FLASHINFER_DISABLE_AUTOTUNE` | — | ⬜ **Removed** — not read by FlashInfer or vLLM (dead env var, was set speculatively) |
| `VLLM_FLASH_ATTN_VERSION` | — | ⬜ **Removed** — env var no longer exists in v0.17. FA version controlled by `attention-config` in model YAML |
| `PYTHONUNBUFFERED` | `1` | ✅ Set — ensures log flushing |
| `VLLM_USE_FLASHINFER_SAMPLER` | — | ✅ **Removed** in v0.16.1rc0 upgrade (2026-03-06). FlashInfer sampler is opt-in since v0.16.0 |
| `HF_HUB_OFFLINE` | `1` | ✅ Required — use cached models only |

## Known Issues & Workarounds

### GPT-OSS-20B MXFP4 — All backends fail on SM120
All 4 MXFP4 backends fail: Marlin (PTX mismatch), SM90 CUTLASS (no SM120 tactics), SM100 TRT-LLM (rejects major=12), Triton (SparseMatrix import error). Additionally, model ships with CUDA 12.x custom kernels incompatible with CUDA 13. No workaround; upstream tracking this.

### Mamba Prefix Caching — Potentially fixed in v0.15.0
Old behaviour: prefix caching causes garbage output on Nemotron/Mamba hybrid models. v0.15.0 added block-aligned prefix caching (`_mamba_block_aligned_split`) and `NemotronHForCausalLM` now implements `SupportsMambaPrefixCaching`. **Re-test with `enable-prefix-caching: true` and `mamba-block-size: 16`.**

### VLM Attention on SM120
FA2 `varlen_fwd` PTX is incompatible with SM120 — crashes VIT encoder. v0.17 fixed main model backend ordering (FLASHINFER > TRITON > FA2) but VIT auto-selection still picks FA2 first.

**Workarounds:**
- `mm-encoder-attn-backend: TORCH_SDPA` in model config — bypasses FA2 for VIT encoder
- `attention-config: '{"backend": "FLASHINFER"}'` in model config — explicit (v0.17 would auto-select this anyway)
- `enforce-eager: true` in some VLM configs — skips CUDA graph capture that triggers FlashInfer GQA warmup bug on models with 32 query / 8 KV heads

### FP8 on SM120
FP8 quantization (`quantization: fp8` via bitsandbytes/cuBLAS) fails on SM120 for dense models like Devstral. Workaround: run in native BF16 (`quantization: null`). Server-side FP8 KV cache (`kv-cache-dtype: fp8`) works fine.

### Qwen3 Reasoning Streaming (PATCHED)
`<think>` injected by chat template into assistant prompt prefix is not detected by the reasoning parser — all thinking goes to `content` instead of `reasoning`. Fix: `fix_qwen3_reasoning_streaming.py` seeds `all_previous_token_ids` with `start_token_id`.

### DeltaNet Triton Autotuner OOM at Long Context
Qwen3.5 DeltaNet models (35B, 9B) use Triton `@triton.autotune` kernels for `gdn_attention_core` and `solve_tril`. The autotuner benchmarks multiple kernel configs on first inference, and its workspace scales with `max_model_len`. **262K context stable** for qwen3.5-35b at 0.86 util, max-num-seqs=1 (tested up to 261K input tokens). **gpu-memory-utilization ≥ 0.90 OOMs during autotuning** (`solve_tril` needs ~4GB headroom beyond model weights). qwen3.5-9b stable at **65K** (0.88 util). `TRITON_ALWAYS_COMPILE=1` does not help — autotuner is a Python-level decorator.

### DeltaNet Batching Penalty
Mamba/DeltaNet hybrid models have a TPOT cliff between seqs=2 and seqs=4. Workload sweep 2026-03-13 (1024in/256out, rates 0.5-inf, 30 prompts each):

| Config | TPOT_p50 | Max Throughput | Best TTFT (rate=1) | Best E2E (rate=1) |
|--------|----------|----------------|--------------------|--------------------|
| seqs=1 | 6.1ms | 160 t/s | 9196ms | 10750ms |
| **seqs=2** | **8.3ms** | **232 t/s (+45%)** | **1472ms (6x better)** | **3633ms (3x better)** |
| seqs=4 | 32.1ms (5.3x) | 126 t/s (-21%) | 12726ms | 21085ms |
| seqs=8 | 33.0ms (5.4x) | 181 t/s (+13%) | 4503ms | 12918ms |

**seqs=2 is optimal** — 45% more throughput with only 37% TPOT increase. The cliff at seqs=4 is catastrophic: 5x TPOT penalty exceeds any throughput gain. seqs=8 partially recovers throughput (penalty plateaus) but still worse than seqs=2 on all metrics. Cause: DeltaNet recurrent state maintenance is expensive per-sequence.

### Batched Tokens Sweep
Sweep of `max-num-batched-tokens` (2048/4096/8192/16384/32768) with seqs=2 locked. Two workloads: short (1024in/256out) and long prefill (16384in/64out). Tested 2026-03-13.

**Short workload (decode-heavy):** No meaningful difference across batch sizes. All configs: ~100 t/s at rate=1, ~230 t/s at rate=inf, ~8.3ms TPOT. Initial batch8192 result (854ms TTFT) was an artifact of warm CUDA graph cache — cold restart confirmed identical ~29s TTFT from JIT compilation on first requests. TPOT is identical across all configs.

**Long prefill workload (prefill-heavy, rate=1):**
| Config | Out t/s | TTFT_p50 | TPOT_p50 |
|--------|---------|----------|----------|
| **batch32768** | **56.8** | **653ms** | **16.4ms** |
| batch4096 | 42.1 | 6131ms | 17.7ms |
| batch8192 | 39.7 | 6447ms | 19.3ms |
| batch16384 | CRASHED (OOM at 5/10) | - | - |

**Conclusion: batch size only matters for long prefill.** batch32768 wins long prefill (10x better TTFT, 43% more throughput) but batch16384 is unstable (OOM). Keeping batch8192 as safe default — it handles all workloads without crashes. At high rates all configs converge (~70 t/s long).

### Context Inflation with Thinking in Multi-turn
Clients sending thinking content as plain text (`requiresThinkingAsText: true`) cause 78% context to be thinking by turn 3 → model degradation. Fix: clients must send thinking in `reasoning_content` field so Qwen3's `truncate_history_thinking` can strip it. OpenClaw: set `"requiresThinkingAsText": false`.

### causal-conv1d / mamba-ssm Build
These packages ignore `TORCH_CUDA_ARCH_LIST`, hardcode arch from CUDA version. Build for all archs (~2 min each). Patching is fragile.

## Default Model Alias

Every config includes `default` as a second `served-model-name`. Services can hardcode `model: default` and it will always resolve to whatever model is currently loaded — no need to track the active model name.

```yaml
served-model-name:
  - qwen3-14b   # specific name for targeted use
  - default     # universal alias for "whatever is loaded"
```

Embedding model (`qwen3-0-6b-embedding`) is excluded — it runs in a separate container and is not a chat model.

## Configuration Patterns

```yaml
# SM120 attention — v0.17 auto-selects FLASHINFER for main model
# VIT encoder needs explicit override (auto-selects FA2 which crashes on SM120)
attention-config: '{"backend": "FLASHINFER"}'        # explicit, optional
mm-encoder-attn-backend: TORCH_SDPA                  # required for VLM on SM120

# CUDA graphs — align capture sizes with max-num-seqs
compilation-config: '{"mode": "VLLM_COMPILE", "cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [1,2,4,8,16]}'

# Thinking on (Nemotron/Qwen3)
default-chat-template-kwargs: '{"enable_thinking": true}'
reasoning-parser-plugin: ~/projects/vllm/parsers/nano_v3_reasoning_parser.py
structured-outputs-config: '{"reasoning_parser": "nano_v3"}'

# Tool calling parsers
tool-call-parser: qwen3_coder   # Qwen3-based + Nemotron-30B (generates <tool_call> XML)
tool-call-parser: hermes        # Hermes-format models (Gemma, Qwen3-VL, Qwen3-30B)
tool-call-parser: mistral       # Devstral
tool-call-parser: nemotron_json # Nemotron-9B (custom parser, streaming variant)
```

## Embedding Model (Separate Container)

The embedding model (`qwen3-0-6b-embedding`) runs in a **separate container** using the official `vllm/vllm-openai:latest` image (NOT the custom sm120 build — PTX incompatibility).

```bash
# Start embedding server on port 8002 (host) → 8001 (container)
docker run -d --name vllm-embed \
  --network docker_default \
  --gpus '"device=0"' \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/projects/vllm/configs/qwen3-0-6b-embedding.yaml:/config.yaml:ro \
  -p 8002:8001 \
  --entrypoint "" \
  vllm/vllm-openai:latest \
  vllm serve --config /config.yaml
```

**Config notes:** `enforce-eager: true` required for SM120. `runner: "pooling"` + `convert: "embed"` replaces broken `--task embed` CLI flag. `truncate_prompt_tokens: 512` in API requests avoids 400 errors on long texts. Model is ~1.1 GB VRAM; with 0.85 util reserves ~4 GB total (mostly KV cache), leaving ~28 GB for the main chat model.

**Throughput (RTX 5090):** ~500 emb/sec single, ~720 concurrent (8 workers), ~640 sustained with Qdrant upload overhead.

## KV Cache Math (Gemma 4 26B — sliding window hybrid)
30 layers: 25 sliding window (1024 tokens) + 5 full attention.
- Sliding layers: 8 KV heads × 256 dim × FP8 = 4,096 bytes/token/layer. Fixed cost: 25 × 1024 × 4096 = **100 MiB** (constant regardless of seq_len).
- Full attn layers: 2 KV heads × 512 dim × FP8 = 2,048 bytes/token/layer. Scales with seq_len: 5 × L × 2048.

| Seq length | Sliding (fixed) | Full attn | Total | Concurrency (3.8 GiB pool) |
|---|---|---|---|---|
| 20K (typical Claude Code) | 100 MB | 205 MB | **0.29 GiB** | 13.1x |
| 36K (prompt + 16K output) | 100 MB | 369 MB | **0.44 GiB** | 8.6x |
| 52K (prompt + 32K output) | 100 MB | 532 MB | **0.59 GiB** | 6.4x |
| 160K | 100 MB | 1,678 MB | **1.66 GiB** | 2.3x |
| 262K (native max) | 100 MB | 2,684 MB | **2.60 GiB** | 1.5x |

The sliding window makes KV extremely light — a typical Claude Code request (20K prompt + 16K output) uses only **0.44 GiB** (~8% of the 33K token pool). Even at 262K max_model_len, one full request fits with 1.2 GiB to spare.

**`max_new_tokens: 16384` cap**: Claude Code sends `max_tokens=32000`. Without a cap, the scheduler pre-allocates KV for `prompt + 32000` output tokens. If this exceeds the pool, the request is permanently stuck in `Waiting` (scheduler deadlock — not an OOM, just can't fit the worst-case allocation). The `max_new_tokens` in `override-generation-config` caps output server-side via `get_max_tokens()` → `min(request, override, model_max)`.

## KV Cache Math (Nemotron-30B Mamba hybrid)
6/52 layers need KV cache, ~3 KB/token (FP8). At 93% util: ~588K tokens available.
262K context × 2 seqs = 524K = 89% pool, 11% headroom. Config: `max-model-len: 262144`, `max-num-seqs: 2`.

## KV Cache Math (Qwen3.5-35B DeltaNet hybrid)
10/40 layers use full attention KV cache (2 KV heads × 256 dim × fp8). 30 DeltaNet layers use O(1) fixed recurrent state.
At 0.86 util, max-model-len 262K (native max), max-num-seqs 2: optimal batching config (see DeltaNet Batching Penalty).
Available KV cache: 3.08 GiB, 79,648 token-slots. DeltaNet O(1) state allows serving sequences far beyond KV slot count.
Stability tested: 100→261K input tokens all pass. Prefill degrades gracefully (22K→6K tok/s). Decode steady at 160 tok/s.
0.88 util OOMs at startup (CUDA graph allocation fails with only 235 MiB free).

## File Structure

```
├── docker/
│   ├── Dockerfile.sm120      # CUDA extension builder only (~90min, rarely rebuild)
│   ├── Dockerfile.runtime    # Runtime: base:cuda131 + sm120 extensions + vLLM v0.17 (~3min)
│   └── patches/              # Applied in Dockerfile.runtime (4 active, 3 obsolete)
├── configs/                  # Per-model YAML configs
├── parsers/                  # Custom tool call + reasoning parsers
├── plugins/
│   ├── thinking_budget_processor.py  # V1 logits processor: caps thinking tokens
│   └── langfuse_tracer.py           # ASGI middleware: traces prompts/completions to Langfuse
├── model-cards/              # HuggingFace model cards with architecture details, config.json, benchmarks
├── sweeps/                   # Performance tuning sweep scripts and results
│   ├── gemma4_adaptive_sweep.sh  # Adaptive: exponential probe → binary search → batch sweep
│   ├── workload_sweep.sh         # Brute-force grid: seqs × rates (Qwen3.5 era)
│   ├── batched_tokens_sweep.sh   # Batch token sweep (Qwen3.5 era)
│   ├── configs/                  # Generated per-sweep YAML configs
│   └── results/                  # Raw sweep output (per-config/per-rate JSON)
├── results/                  # Git-tracked benchmark summaries (by model/date)
├── tests/
│   ├── run.py                # Unified test runner (validate + GuideLLM benchmarks)
│   ├── validate.py           # Content/correctness regression tests
│   ├── test_model_quality.py # Off-by-one, fence-post, scheduling quality tests
│   ├── test_langfuse_tracer.py   # Langfuse middleware integration tests
│   ├── test_thinking_budget.py # Thinking budget processor tests
│   ├── models.yaml           # Per-model capability definitions for validate.py
│   └── bench_data/           # Pre-built JSONL benchmark datasets (avoids tokenizer overhead)
│       └── generate.py       # Regenerate datasets: python3 tests/bench_data/generate.py
├── benchmarks/               # Raw test output directories (gitignored)
└── logs/                     # Runtime logs (gitignored)
```

## Test Runner (`tests/run.py`)

Unified entry point for all testing. Auto-detects served model and HF cache processor path.

| Suite | What runs | Time |
|-------|-----------|------|
| `quick` | validate.py + GuideLLM smoke (5 sync requests) | ~30s |
| `standard` | quick + stability (100→64K tokens) + quality + decode speed + throughput sweep | ~5min |
| `full` | standard + long context (96K, 130K tokens) | ~10min |

Quality tests run by default in standard/full. Add `--skip-quality` to exclude. Add `--skip-validate` for benchmarks only.

**GuideLLM**: v0.3.1. Uses `--rate-type` and `--disable-progress`. Pre-built JSONL datasets in `tests/bench_data/` avoid expensive tokenizer overhead (regenerate with `python3 tests/bench_data/generate.py`).

**Results tracking**: Compact summaries saved to `results/<model-slug>/<date>/summary.json` (git-tracked). Raw GuideLLM output in `benchmarks/` (gitignored). Use `git log -p results/` to compare performance over time.

## Performance Sweep (`sweeps/`)

Scripts for finding optimal `max-num-seqs` and `max-num-batched-tokens` per model. Each sweep restarts vLLM with a modified config, warms up torch compile + CUDA graphs, then benchmarks with `vllm bench serve`.

### Adaptive Sweep (preferred)

`bash sweeps/gemma4_adaptive_sweep.sh`

Three-phase approach that minimises restarts (typically ~12 vs 24+ for brute-force):

1. **Exponential probe** — Test seqs=1,2,4,8,16,32,64 at rate=inf. One benchmark per config. Detect OOM or TPOT cliff (>2x degradation). CUDA graph capture sizes auto-scaled to match seqs (powers of 2 up to seqs).
2. **Binary search** — Narrow between best-performing seqs and cliff point. 2-3 iterations.
3. **Batch token sweep** — Test batch=4096,8192,16384,32768,65536 on the winning seqs.

**Torch compile warmup**: After each server restart, the script sends concurrent warmup requests at each CUDA graph capture size [1,2,4,...,seqs] to trigger JIT compilation before benchmarking. Without this, TTFT on first requests includes compile overhead and pollutes results.

**Adapting for a new model**: Copy the script, change `BASE_CONFIG` to the model's YAML, and `RESULT_DIR` to a new path. The `make_config` function patches `max-num-seqs`, `max-num-batched-tokens`, and `cudagraph_capture_sizes` from the base config. If the model doesn't use torch compile, remove the warmup logic.

### Brute-Force Grid (legacy)

`bash sweeps/workload_sweep.sh` — Tests fixed seqs=[1,2,4,8] × rates=[0.5,1,2,4,inf]. More thorough for understanding latency-throughput tradeoffs at different load levels, but 20+ restarts. Used for the Qwen3.5-35B DeltaNet batching penalty analysis.

### Key findings per model

| Model | Optimal seqs | Optimal batch | Throughput | Cliff | Cause |
|-------|-------------|--------------|------------|-------|-------|
| Qwen3.5-35B (DeltaNet) | 2 | 8192 | 232 t/s | seqs=4 (5x TPOT) | DeltaNet recurrent state overhead |
| Gemma 4 26B (MoE) | 4 | 4096 | 418 t/s | seqs=6 (OOM) | CUDA graph capture memory |

## LLM Tracing Middleware (Langfuse + OpenLIT)

ASGI middleware at `plugins/langfuse_tracer.py` that captures full prompt/completion content and exports to Langfuse and OpenLIT via OpenTelemetry. Uses a single `TracerProvider` with multiple `BatchSpanProcessor` exporters — one per destination. Separate from vLLM's native OTEL tracing.

**Loaded via**: `middleware` in model config YAML. Requires `PYTHONPATH=~/projects/vllm` in container env.

**Environment variables** (set in `docker-compose.yml`):
- `LANGFUSE_OTEL_ENDPOINT` — OTLP HTTP endpoint (e.g. `http://langfuse:3000/api/public/otel/v1/traces`)
- `LANGFUSE_PUBLIC_KEY` — Langfuse public API key
- `LANGFUSE_SECRET_KEY` — Langfuse secret API key
- `OPENLIT_OTLP_ENDPOINT` — OTLP HTTP endpoint for OpenLIT (e.g. `http://openlit:4318/v1/traces`)

**Supported endpoints**: `/v1/chat/completions`, `/v1/completions` (OpenAI), `/v1/messages` (Anthropic)

**How it works**:
- Intercepts all three endpoints, auto-detects format from path
- Reads request body to extract `messages`/`prompt` and `model`
- Adds input messages as OTEL GenAI events (`gen_ai.user.message`, etc.) for chat-style rendering in Langfuse
- For non-streaming: reads response body, extracts completion text and usage
- For streaming: wraps ASGI `send()` to accumulate content deltas (OpenAI `choices[].delta` or Anthropic `content_block_delta`)
- Separates thinking/reasoning from content (`message.reasoning_content` vs `message.content`)
- Tracks TTFT (`langfuse.observation.completion_start_time`) for streaming
- Extracts model parameters (temperature, top_p, etc.) into `langfuse.observation.model.parameters`
- Identifies caller from `X-Caller`/`X-Service-Name` headers or User-Agent patterns (openclaw, claude-code, etc.)
- Tags traces with `vllm`, `streaming`, `anthropic-api`, and caller name

**Adding to new model configs**: Add `middleware: ["plugins.langfuse_tracer.LangfuseTracerMiddleware"]` to the model YAML. The middleware is a no-op when no OTEL endpoints are configured.

**Testing**: `python3 tests/test_langfuse_tracer.py` — sends 6 requests (3 endpoints × 2 modes), verifies all 6 traces in Langfuse have correct tags, usage, model resolution, TTFT, and reasoning separation.

**Note**: The `langfuse` Python package is installed in the Dockerfile (`pip install langfuse`). The middleware itself only uses `opentelemetry-*` packages directly but `langfuse` is available for future SDK-based enhancements.

## Thinking Budget Control (v0.19 Native)

v0.19 introduces native `thinking_token_budget` on SamplingParams and `reasoning_effort` on the OpenAI-compatible API. This replaces the custom `ThinkingBudgetLogitsProcessor` (deprecated, kept in `plugins/` for reference).

**Server config required**: Models must have `reasoning-config` with `reasoning_start_str` and `reasoning_end_str` to use thinking budgets. Without it, `thinking_token_budget` causes a 400 error.

```yaml
# Example model config for Qwen3.5
reasoning-parser: qwen3
reasoning-config: '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}'

# Example for Gemma 4
reasoning-parser: gemma4
reasoning-config: '{"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}'
```

**Client API** — three options:

1. **`reasoning_effort`** (preferred — native v0.19 + patched defaults):
```python
client.chat.completions.create(
    model="default",
    messages=[...],
    extra_body={"reasoning_effort": "medium"}  # none, low, medium, high
)
```

| `reasoning_effort` | `enable_thinking` | `thinking_token_budget` |
|---|---|---|
| `none` | `false` | — |
| `low` | `true` | 2048 |
| `medium` | `true` | 8192 |
| `high` | `true` | unlimited |

The mapping from `reasoning_effort` → `thinking_token_budget` + `enable_thinking` is via `fix_reasoning_effort_budget_defaults.py` patch (not native v0.19). Without the patch, `reasoning_effort` only passes through to the chat template as a kwarg.

2. **Anthropic `/v1/messages` endpoint** (patched in `fix_anthropic_thinking_compat.py`):
```python
response = client.messages.create(
    model="default",
    max_tokens=16384,
    thinking={"type": "enabled", "budget_tokens": 4096},
    messages=[...]
)
```
Note: Near-max `budget_tokens` (≥80% of `max_tokens`) are filtered out as "no cap" — server defaults apply.

3. **Direct `thinking_token_budget`** (for precise control):
```python
client.chat.completions.create(
    model="default",
    messages=[...],
    extra_body={"thinking_token_budget": 512}
)
```

**v0.19 breaking change**: Response field `reasoning_content` → `reasoning`. The OTEL tracer already handles both field names.

**Penalty interaction**: High `presence_penalty` (≥0.6) still corrupts JSON tool call output. Use pp=0.4 max with tools. The thinking budget handles loop prevention.

**Test**: `python3 tests/test_thinking_budget.py [--step N]` — steps 0-5: OpenAI endpoint tests, step 6: Anthropic endpoint tests.
