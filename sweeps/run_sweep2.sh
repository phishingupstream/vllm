#!/bin/bash
# Sweep runner v2 - proper container lifecycle with YAML configs
RESULT_DIR="~/projects/vllm/sweeps/results/qwen35-35b-config-sweep"

run_bench() {
    local NAME=$1
    local CONFIG=$2

    # Skip if all benchmarks exist
    local ALL_DONE=true
    for B in short medium long_input; do
        [ ! -f "$RESULT_DIR/SERVE--${NAME}-BENCH--${B}/run=0.json" ] && ALL_DONE=false
    done
    if $ALL_DONE; then
        echo "=== SKIP $NAME (done) ==="
        return 0
    fi

    echo "=== START $NAME ==="
    docker stop vllm-sweep 2>/dev/null; docker rm vllm-sweep 2>/dev/null
    sleep 3

    # Start vLLM serve with config YAML as CMD
    docker run --rm -d \
      --name vllm-sweep \
      --gpus all \
      --ipc host \
      --shm-size=16g \
      --ulimit memlock=-1 \
      -v /data:/data \
      -v /faststore:/faststore \
      -v /data/cache:~/.cache \
      -e HF_HOME=/faststore/huggingface \
      -e HF_HUB_OFFLINE=1 \
      -e VLLM_USE_FLASHINFER_MOE_FP4=1 \
      -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass \
      -e PYTORCH_ALLOC_CONF=expandable_segments:True \
      -e PYTHONUNBUFFERED=1 \
      -e PYTHONHASHSEED=0 \
      vllm:runtime \
      vllm serve --config "$CONFIG"

    # Wait for server ready
    echo "  Waiting for server..."
    for i in $(seq 1 120); do
        if docker exec vllm-sweep curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "  Server ready (${i}x5s)"
            break
        fi
        if ! docker ps -q --filter name=vllm-sweep | grep -q .; then
            echo "  ERROR: Container died during startup"
            return 1
        fi
        sleep 5
    done

    # Run benchmarks (short, medium, long_input)
    local BENCHMARKS="short:512:256:20 medium:2048:512:10 long_input:8192:256:5"
    for BENCH in $BENCHMARKS; do
        IFS=: read -r BNAME INPUT OUTPUT NUMPROMPTS <<< "$BENCH"
        local OUTDIR="$RESULT_DIR/SERVE--${NAME}-BENCH--${BNAME}"

        if [ -f "$OUTDIR/run=0.json" ]; then
            echo "  Skip $BNAME (exists)"
            continue
        fi

        # Check server still alive
        if ! docker exec vllm-sweep curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "  ERROR: Server died before $BNAME"
            break
        fi

        echo "  Bench $BNAME (${INPUT}in/${OUTPUT}out x${NUMPROMPTS})..."
        docker exec vllm-sweep vllm bench serve \
            --dataset-name random \
            --input-len "$INPUT" \
            --output-len "$OUTPUT" \
            --num-prompts "$NUMPROMPTS" \
            --request-rate inf \
            --ignore-eos \
            --percentile-metrics ttft,tpot,itl,e2el \
            --save-result \
            --result-dir "$OUTDIR" \
            --result-filename "run=0.json" 2>&1 | grep -E "Throughput|Output|completed|tok/s" || true
        echo "  Done $BNAME"
    done

    echo "=== DONE $NAME ==="
    docker stop vllm-sweep 2>/dev/null
    docker rm vllm-sweep 2>/dev/null
    sleep 5
}

# Run remaining configs
run_bench "gpu90_seqs1" "~/projects/vllm/sweeps/configs/gpu90.yaml"
run_bench "gpu95_seqs1" "~/projects/vllm/sweeps/configs/gpu95.yaml"
run_bench "prefix_cache_on" "~/projects/vllm/sweeps/configs/prefix_cache.yaml"

echo ""
echo "=== RESULTS SUMMARY ==="
for f in $RESULT_DIR/*/run=0.json; do
    DIR=$(basename $(dirname "$f"))
    python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
name = sys.argv[2]
done = d.get('completed', 0)
total = d.get('num_prompts', 0)
print(f'  {name:<55} out={d.get(\"output_throughput\",0):>7.1f} tok/s  TPOT_p50={d.get(\"median_tpot_ms\",0):>6.2f}ms  done={done}/{total}')
" "$f" "$DIR" 2>/dev/null
done
