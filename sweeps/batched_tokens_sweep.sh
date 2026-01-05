#!/bin/bash
# Batched tokens sweep: test max-num-batched-tokens with seqs=2 (locked)
# Measures prefill chunking impact on throughput and latency
set -o pipefail

RESULT_DIR="~/projects/vllm/sweeps/results/batched-tokens-sweep"
BASE_CONFIG="~/projects/vllm/configs/qwen3.5-35b-a3b-nvfp4-nvidia.yaml"
mkdir -p "$RESULT_DIR"

# Benchmark parameters — two workloads:
# short: tests decode-heavy (small prefill, many prompts)
# long_prefill: tests prefill-heavy (large input, fewer prompts)
NUM_PROMPTS=20
RATES="1 4 inf"

run_benchmarks() {
    local CONFIG_NAME=$1
    echo "  Benchmarks for $CONFIG_NAME"

    # Short workload (1024in/256out) — decode-heavy
    for RATE in $RATES; do
        local OUTDIR="$RESULT_DIR/${CONFIG_NAME}/short-rate-${RATE}"
        if [ -f "$OUTDIR/run=0.json" ]; then
            echo "    Skip short rate=$RATE (exists)"
            continue
        fi
        if ! docker exec vllm curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "    ERROR: Server not healthy"; continue
        fi
        echo -n "    short rate=$RATE ... "
        mkdir -p "$OUTDIR"
        docker exec vllm vllm bench serve \
            --dataset-name random \
            --input-len 1024 --output-len 256 \
            --num-prompts $NUM_PROMPTS \
            --request-rate "$RATE" --ignore-eos \
            --percentile-metrics ttft,tpot,itl,e2el \
            --save-result --result-dir "$OUTDIR" --result-filename "run=0.json" 2>&1 > /dev/null
        if [ -f "$OUTDIR/run=0.json" ]; then
            python3 -c "
import json; d = json.load(open('$OUTDIR/run=0.json'))
print(f'done ({d.get(\"completed\",0)}/{d.get(\"num_prompts\",0)}) out={d.get(\"output_throughput\",0):.1f}t/s TTFT={d.get(\"median_ttft_ms\",0):.0f}ms TPOT={d.get(\"median_tpot_ms\",0):.1f}ms')
" 2>/dev/null || echo "done"
        else echo "FAILED"; fi
    done

    # Long prefill workload (16384in/64out) — prefill-heavy
    for RATE in $RATES; do
        local OUTDIR="$RESULT_DIR/${CONFIG_NAME}/long-rate-${RATE}"
        if [ -f "$OUTDIR/run=0.json" ]; then
            echo "    Skip long rate=$RATE (exists)"
            continue
        fi
        if ! docker exec vllm curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "    ERROR: Server not healthy"; continue
        fi
        echo -n "    long rate=$RATE ... "
        mkdir -p "$OUTDIR"
        docker exec vllm vllm bench serve \
            --dataset-name random \
            --input-len 16384 --output-len 64 \
            --num-prompts 10 \
            --request-rate "$RATE" --ignore-eos \
            --percentile-metrics ttft,tpot,itl,e2el \
            --save-result --result-dir "$OUTDIR" --result-filename "run=0.json" 2>&1 > /dev/null
        if [ -f "$OUTDIR/run=0.json" ]; then
            python3 -c "
import json; d = json.load(open('$OUTDIR/run=0.json'))
print(f'done ({d.get(\"completed\",0)}/{d.get(\"num_prompts\",0)}) out={d.get(\"output_throughput\",0):.1f}t/s TTFT={d.get(\"median_ttft_ms\",0):.0f}ms TPOT={d.get(\"median_tpot_ms\",0):.1f}ms')
" 2>/dev/null || echo "done"
        else echo "FAILED"; fi
    done
}

make_config() {
    local BATCH=$1
    local OUT=$2
    cp "$BASE_CONFIG" "$OUT"
    sed -i "s/^max-num-seqs:.*/max-num-seqs: 2/" "$OUT"
    sed -i "s/^max-num-batched-tokens:.*/max-num-batched-tokens: $BATCH/" "$OUT"
}

swap_server() {
    local CONFIG=$1
    local NAME=$2
    echo ""
    echo "=== Server: $NAME ==="

    docker stop vllm 2>/dev/null
    docker rm vllm 2>/dev/null
    sleep 5

    docker run --rm -d \
      --name vllm \
      --gpus all \
      --ipc host \
      --shm-size=16g \
      --ulimit memlock=-1 \
      -v /data:/data \
      -v /faststore:/faststore \
      -v /data/cache:~/.cache \
      -v ~/.gitconfig:~/.gitconfig:ro \
      -e HF_HOME=/faststore/huggingface \
      -e HF_HUB_OFFLINE=1 \
      -e VLLM_USE_FLASHINFER_MOE_FP4=1 \
      -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass \
      -e PYTORCH_ALLOC_CONF=expandable_segments:True \
      -e PYTHONUNBUFFERED=1 \
      -e PYTHONHASHSEED=0 \
      vllm:runtime \
      bash -c "vllm serve --config $CONFIG 2>&1 | tee -a ~/projects/vllm/logs/vllm-sweep.log"

    echo "  Waiting for server..."
    for i in $(seq 1 120); do
        if docker exec vllm curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "  Ready (${i}x5s)"
            RESP=$(docker exec vllm curl -s http://localhost:8000/v1/completions \
                -H "Content-Type: application/json" \
                -d '{"model": "qwen3.5-35b", "prompt": "Hi", "max_tokens": 3}' 2>&1)
            if echo "$RESP" | grep -q "choices"; then
                echo "  Verified: serving OK"
                return 0
            else
                echo "  WARNING: health OK but completions failed: $RESP"
                sleep 10
                return 0
            fi
        fi
        if ! docker ps -q --filter name=vllm | grep -q .; then
            echo "  ERROR: Container died"
            return 1
        fi
        sleep 5
    done
    echo "  ERROR: Timeout"
    return 1
}

# ============================================================
# Configs: seqs=2, varying max-num-batched-tokens
# 2048 = small chunks (more scheduling overhead, less memory per step)
# 4096 = moderate
# 8192 = current default
# 16384 = large chunks (fewer scheduling rounds, more memory per step)
# 32768 = very large (may stress memory)
# ============================================================

CONFIGS="2048 4096 8192 16384 32768"
mkdir -p ~/projects/vllm/sweeps/configs

# Phase 1: use current running server if it matches 8192
echo "=== Phase 1: batch=8192 (current server) ==="
if docker exec vllm curl -s http://localhost:8000/health > /dev/null 2>&1; then
    run_benchmarks "batch8192"
else
    make_config 8192 ~/projects/vllm/sweeps/configs/sweep_batch8192.yaml
    if swap_server ~/projects/vllm/sweeps/configs/sweep_batch8192.yaml "batch8192"; then
        run_benchmarks "batch8192"
    fi
fi

# Remaining configs
for BATCH in $CONFIGS; do
    [ "$BATCH" = "8192" ] && continue
    make_config "$BATCH" "~/projects/vllm/sweeps/configs/sweep_batch${BATCH}.yaml"
    if swap_server "~/projects/vllm/sweeps/configs/sweep_batch${BATCH}.yaml" "batch${BATCH}"; then
        run_benchmarks "batch${BATCH}"
    fi
done

# ============================================================
# Restore original
# ============================================================
echo ""
echo "=== Restoring original vLLM ==="
docker stop vllm 2>/dev/null
docker rm vllm 2>/dev/null
sleep 3
cd /data/docker && docker compose up -d vllm 2>&1

# ============================================================
# Summary
# ============================================================
echo ""
echo "=== RESULTS TABLE ==="
echo ""
printf "%-12s %-6s %6s  %8s %8s %10s %8s %10s %6s\n" \
    "Config" "Work" "Rate" "Out_t/s" "Req/s" "TTFT_p50" "TPOT_50" "E2E_p50" "Done"
echo "--------------------------------------------------------------------------------------"
for f in $(find "$RESULT_DIR" -name "run=0.json" | sort); do
    python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
path = sys.argv[1]
parts = path.split('/')
config = parts[-3]
bench = parts[-2]
done = d.get('completed', 0)
total = d.get('num_prompts', 0)
if done == 0: sys.exit()
print(f'{config:<12} {bench:<6} {d.get(\"output_throughput\",0):>8.1f} {d.get(\"request_throughput\",0):>8.2f} {d.get(\"median_ttft_ms\",0):>10.0f} {d.get(\"median_tpot_ms\",0):>8.2f} {d.get(\"median_e2el_ms\",0):>10.0f} {done:>3}/{total}')
" "$f" 2>/dev/null
done
