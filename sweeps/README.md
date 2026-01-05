# Performance Sweeps

Scripts for finding optimal `max-num-seqs` and `max-num-batched-tokens` per model. Each sweep restarts vLLM with modified configs, benchmarks, and reports.

## Adaptive Sweep (preferred)

Three-phase approach that minimises server restarts (~12 vs 24+ for grid search):

```
Phase 1: Exponential probe    seqs=1,2,4,8,16,32,64 at rate=inf
          ↓ find OOM or TPOT cliff
Phase 2: Binary search         narrow between best and cliff
          ↓ find optimal seqs
Phase 3: Batch token sweep     batch=4096..65536 on winning seqs
```

### Usage

```bash
# Run for Gemma 4
bash sweeps/gemma4_adaptive_sweep.sh

# Adapt for a new model:
# 1. Copy the script
# 2. Change BASE_CONFIG to the model's YAML
# 3. Change RESULT_DIR to a new path
# 4. Adjust INPUT_LEN/OUTPUT_LEN for your workload
```

### How it works

1. `make_config(seqs, batch)` — copies base YAML, patches `max-num-seqs`, `max-num-batched-tokens`, and auto-scales `cudagraph_capture_sizes` to powers of 2 up to seqs.

2. `start_server(config, name, seqs)` — stops existing vLLM, starts fresh container with the config. After health check, sends warmup requests at each CUDA graph capture size to trigger torch compile JIT before benchmarking.

3. `bench_one(seqs, batch)` — starts server, runs `vllm bench serve` at rate=inf with random dataset, returns throughput/TPOT/TTFT. Caches results (skips if `run=0.json` exists).

4. Phase 1 detects cliff by either: OOM/crash on startup, or TPOT degradation >2x vs previous config.

5. Phase 2 binary searches between best throughput and cliff point (2-3 iterations).

6. Phase 3 sweeps batch sizes on the winning seqs count.

### Torch compile warmup

Critical for accurate TTFT measurements. After each server restart:

- Each new CUDA graph capture size triggers JIT compilation on first use
- The script sends concurrent requests at each power-of-2 up to max-num-seqs
- Extra stabilisation passes follow
- Without this, first benchmark request's TTFT includes compile overhead

## Legacy Scripts

| Script | Purpose |
|--------|---------|
| `workload_sweep.sh` | Grid: seqs=[1,2,4,8] × rates=[0.5,1,2,4,inf]. Thorough but 20+ restarts |
| `batched_tokens_sweep.sh` | Batch tokens only: [2048,4096,8192,16384,32768] at fixed seqs |
| `run_sweep.sh` / `run_sweep2.sh` | Early prototypes of config sweep |

## Results

Results stored in two places:
- `sweeps/results/` — raw per-config, per-rate JSON from `vllm bench serve`
- `results/<model>/<date>/sweep-summary.json` — git-tracked summary with findings

## Key Findings

| Model | Optimal seqs | Optimal batch | Throughput | Cliff | Cause |
|-------|-------------|--------------|------------|-------|-------|
| Qwen3.5-35B (DeltaNet) | 2 | 8192 | 232 t/s | seqs=4 (5x TPOT) | DeltaNet recurrent state |
| Gemma 4 26B (MoE FP8) | 4 | 4096 | 418 t/s | seqs=6 (OOM) | CUDA graph capture memory |

MoE models scale better with batching than DeltaNet/recurrent models — the cliff is memory (OOM) not compute (TPOT degradation).
