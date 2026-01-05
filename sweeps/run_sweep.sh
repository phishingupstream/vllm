#!/bin/bash
# Manual sweep runner - one container per config for clean GPU lifecycle
set -e

RESULT_DIR="~/projects/vllm/sweeps/results/qwen35-35b-config-sweep"
COMMON_SERVE_ARGS="--model txn545/Qwen3.5-35B-A3B-NVFP4 --served-model-name qwen3.5-35b --host 0.0.0.0 --port 8000 --trust-remote-code --attention-config '{\"backend\": \"FLASHINFER\"}' --mm-encoder-attn-backend TORCH_SDPA --compilation-config '{\"mode\": \"VLLM_COMPILE\", \"cudagraph_mode\": \"FULL_AND_PIECEWISE\", \"cudagraph_capture_sizes\": [1,2]}' --kv-cache-dtype fp8 --max-model-len 131072 --enable-chunked-prefill"
COMMON_DOCKER="docker run --rm --gpus all --ipc host --shm-size=16g --ulimit memlock=-1 -v /data:/data -v /faststore:/faststore -v /data/cache:~/.cache -e HF_HOME=/faststore/huggingface -e HF_HUB_OFFLINE=1 -e VLLM_USE_FLASHINFER_MOE_FP4=1 -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass -e PYTORCH_ALLOC_CONF=expandable_segments:True -e PYTHONUNBUFFERED=1 -e PYTHONHASHSEED=0"

run_config() {
    local NAME=$1
    local SERVE_EXTRA=$2

    # Skip if all 3 benchmarks already exist
    if [ -f "$RESULT_DIR/SERVE--${NAME}-BENCH--short/run=0.json" ] && \
       [ -f "$RESULT_DIR/SERVE--${NAME}-BENCH--medium/run=0.json" ] && \
       [ -f "$RESULT_DIR/SERVE--${NAME}-BENCH--long_input/run=0.json" ]; then
        echo "=== SKIP $NAME (already done) ==="
        return
    fi

    echo "=== START $NAME ==="

    # Start server in background container
    $COMMON_DOCKER --name vllm-sweep -d vllm:runtime \
        bash -c "vllm serve $COMMON_SERVE_ARGS $SERVE_EXTRA 2>&1 &
        # Wait for server ready
        for i in \$(seq 1 120); do
            if curl -s http://localhost:8000/health > /dev/null 2>&1; then
                echo 'SERVER READY'
                break
            fi
            sleep 5
        done
        sleep infinity"

    # Wait for server ready
    echo "Waiting for server..."
    for i in $(seq 1 120); do
        if docker exec vllm-sweep curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "Server ready after ${i}*5 seconds"
            break
        fi
        sleep 5
    done

    # Run benchmarks
    for BENCH in "short 512 256 20" "medium 2048 512 10" "long_input 8192 256 5"; do
        read -r BNAME INPUT OUTPUT NUMPROMPTS <<< "$BENCH"
        local OUTDIR="$RESULT_DIR/SERVE--${NAME}-BENCH--${BNAME}"

        if [ -f "$OUTDIR/run=0.json" ]; then
            echo "  Skip $BNAME (exists)"
            continue
        fi

        echo "  Running $BNAME..."
        docker exec vllm-sweep vllm bench serve \
            --dataset-name random \
            --input-len $INPUT \
            --output-len $OUTPUT \
            --num-prompts $NUMPROMPTS \
            --request-rate inf \
            --ignore-eos \
            --percentile-metrics ttft,tpot,itl,e2el \
            --save-result \
            --result-dir "$OUTDIR" \
            --result-filename "run=0.json" 2>&1 | tail -5
        echo "  Done $BNAME"
    done

    echo "=== DONE $NAME ==="
    docker stop vllm-sweep 2>/dev/null
    docker rm vllm-sweep 2>/dev/null
    sleep 5  # Let GPU memory fully release
}

# Configs to test
run_config "baseline" "--gpu-memory-utilization 0.86 --no-enable-prefix-caching --max-num-seqs 1 --max-num-batched-tokens 8192"
run_config "gpu90_seqs1" "--gpu-memory-utilization 0.90 --no-enable-prefix-caching --max-num-seqs 1 --max-num-batched-tokens 8192"
run_config "gpu95_seqs1" "--gpu-memory-utilization 0.95 --no-enable-prefix-caching --max-num-seqs 1 --max-num-batched-tokens 8192"
run_config "prefix_cache_on" "--gpu-memory-utilization 0.90 --enable-prefix-caching --max-num-seqs 1 --max-num-batched-tokens 8192"
run_config "seqs4_batch16k" "--gpu-memory-utilization 0.90 --no-enable-prefix-caching --max-num-seqs 4 --max-num-batched-tokens 16384"

echo "=== ALL CONFIGS COMPLETE ==="

# Summary
echo ""
echo "Results:"
for f in $RESULT_DIR/*/run=0.json; do
    DIR=$(basename $(dirname $f))
    python3 -c "
import json
d = json.load(open('$f'))
print(f'  {\"$DIR\":<55} out={d.get(\"output_throughput\",0):>7.1f} tok/s  TPOT_p50={d.get(\"median_tpot_ms\",0):>6.2f}ms  TTFT_p50={d.get(\"median_ttft_ms\",0):>8.0f}ms')
" 2>/dev/null
done
