#!/bin/bash
# Workload sweep: test latency-throughput tradeoff across max-num-seqs configs
# Uses the FULL production config, only modifying max-num-seqs and max-num-batched-tokens
set -o pipefail

RESULT_DIR="~/projects/vllm/sweeps/results/workload-sweep"
BASE_CONFIG="~/projects/vllm/configs/qwen3.5-35b-a3b-nvfp4-nvidia.yaml"
mkdir -p "$RESULT_DIR"

# Benchmark parameters
INPUT_LEN=1024
OUTPUT_LEN=256
NUM_PROMPTS=30
RATES="0.5 1 2 4 inf"

run_benchmarks() {
    local CONFIG_NAME=$1
    echo "  Benchmarks for $CONFIG_NAME"

    for RATE in $RATES; do
        local OUTDIR="$RESULT_DIR/${CONFIG_NAME}/rate-${RATE}"
        if [ -f "$OUTDIR/run=0.json" ]; then
            echo "    Skip rate=$RATE (exists)"
            continue
        fi

        # Check server alive
        if ! docker exec vllm curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "    ERROR: Server not healthy, skipping rate=$RATE"
            continue
        fi

        echo -n "    rate=$RATE ... "
        mkdir -p "$OUTDIR"
        docker exec vllm vllm bench serve \
            --dataset-name random \
            --input-len $INPUT_LEN \
            --output-len $OUTPUT_LEN \
            --num-prompts $NUM_PROMPTS \
            --request-rate "$RATE" \
            --ignore-eos \
            --percentile-metrics ttft,tpot,itl,e2el \
            --save-result \
            --result-dir "$OUTDIR" \
            --result-filename "run=0.json" 2>&1 > /dev/null

        # Quick result
        if [ -f "$OUTDIR/run=0.json" ]; then
            python3 -c "
import json
d = json.load(open('$OUTDIR/run=0.json'))
done = d.get('completed',0)
total = d.get('num_prompts',0)
print(f'done ({done}/{total}) out={d.get(\"output_throughput\",0):.1f}t/s TPOT={d.get(\"median_tpot_ms\",0):.1f}ms')
" 2>/dev/null || echo "done"
        else
            echo "FAILED"
        fi
    done
}

make_config() {
    local SEQS=$1
    local BATCH=$2
    local OUT=$3
    cp "$BASE_CONFIG" "$OUT"
    sed -i "s/^max-num-seqs:.*/max-num-seqs: $SEQS/" "$OUT"
    sed -i "s/^max-num-batched-tokens:.*/max-num-batched-tokens: $BATCH/" "$OUT"
}

swap_server() {
    local CONFIG=$1
    local NAME=$2
    echo ""
    echo "=== Server: $NAME ==="

    # Stop existing
    docker stop vllm 2>/dev/null
    docker rm vllm 2>/dev/null
    sleep 5

    # Start fresh
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
            # Verify it can actually serve
            RESP=$(docker exec vllm curl -s http://localhost:8000/v1/completions \
                -H "Content-Type: application/json" \
                -d '{"model": "qwen3.5-35b", "prompt": "Hi", "max_tokens": 3}' 2>&1)
            if echo "$RESP" | grep -q "choices"; then
                echo "  Verified: serving OK"
                return 0
            else
                echo "  WARNING: health OK but completions failed: $RESP"
                echo "  Waiting for warmup..."
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
# Phase 1: seqs=1 (current running server)
# ============================================================
echo "=== Phase 1: seqs=1 (current server) ==="
if docker exec vllm curl -s http://localhost:8000/health > /dev/null 2>&1; then
    run_benchmarks "seqs1"
else
    echo "Server not running, creating config and starting..."
    make_config 1 8192 ~/projects/vllm/sweeps/configs/sweep_seqs1.yaml
    if swap_server ~/projects/vllm/sweeps/configs/sweep_seqs1.yaml "seqs1"; then
        run_benchmarks "seqs1"
    fi
fi

# ============================================================
# Phase 2: seqs=2
# ============================================================
make_config 2 8192 ~/projects/vllm/sweeps/configs/sweep_seqs2.yaml
if swap_server ~/projects/vllm/sweeps/configs/sweep_seqs2.yaml "seqs2"; then
    run_benchmarks "seqs2"
fi

# ============================================================
# Phase 3: seqs=4
# ============================================================
make_config 4 16384 ~/projects/vllm/sweeps/configs/sweep_seqs4.yaml
if swap_server ~/projects/vllm/sweeps/configs/sweep_seqs4.yaml "seqs4"; then
    run_benchmarks "seqs4"
fi

# ============================================================
# Phase 4: seqs=8
# ============================================================
make_config 8 32768 ~/projects/vllm/sweeps/configs/sweep_seqs8.yaml
if swap_server ~/projects/vllm/sweeps/configs/sweep_seqs8.yaml "seqs8"; then
    run_benchmarks "seqs8"
fi

# ============================================================
# Restore original
# ============================================================
echo ""
echo "=== Restoring original vLLM ==="
docker stop vllm 2>/dev/null
docker rm vllm 2>/dev/null
sleep 3
# Restart the original container (docker start works if it still exists)
docker start vllm 2>/dev/null || echo "NOTE: Restart original vllm container manually"

# ============================================================
# Summary
# ============================================================
echo ""
echo "=== RESULTS TABLE ==="
echo ""
printf "%-8s %6s  %8s %8s %10s %8s %8s %10s %6s\n" \
    "Config" "Rate" "Out_t/s" "Req/s" "TTFT_p50" "TPOT_50" "ITL_p50" "E2E_p50" "Done"
echo "-------------------------------------------------------------------------------------"
for f in $(find "$RESULT_DIR" -name "run=0.json" | sort); do
    python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
path = sys.argv[1]
parts = path.split('/')
config = parts[-3]
rate = parts[-2].replace('rate-','')
done = d.get('completed', 0)
total = d.get('num_prompts', 0)
if done == 0: sys.exit()
print(f'{config:<8} {rate:>6}  {d.get(\"output_throughput\",0):>8.1f} {d.get(\"request_throughput\",0):>8.2f} {d.get(\"median_ttft_ms\",0):>10.0f} {d.get(\"median_tpot_ms\",0):>8.2f} {d.get(\"median_itl_ms\",0):>8.2f} {d.get(\"median_e2el_ms\",0):>10.0f} {done:>3}/{total}')
" "$f" 2>/dev/null
done
