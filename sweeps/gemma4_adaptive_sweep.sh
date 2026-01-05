#!/bin/bash
# Adaptive sweep: exponential probe → binary search → fine-tune
# Finds optimal max-num-seqs by detecting TPOT degradation cliff or OOM
set -o pipefail

RESULT_DIR="~/projects/vllm/sweeps/results/gemma4-adaptive"
BASE_CONFIG="~/projects/vllm/configs/gemma-4-26b-a4b.yaml"
SWEEP_CONFIGS="~/projects/vllm/sweeps/configs/gemma4"
mkdir -p "$RESULT_DIR" "$SWEEP_CONFIGS"

INPUT_LEN=1024
OUTPUT_LEN=256
NUM_PROMPTS=20

make_config() {
    local SEQS=$1
    local BATCH=$2
    local OUT="$SWEEP_CONFIGS/adaptive_seqs${SEQS}_batch${BATCH}.yaml"
    cp "$BASE_CONFIG" "$OUT"
    sed -i "s/^max-num-seqs:.*/max-num-seqs: $SEQS/" "$OUT"
    sed -i "s/^max-num-batched-tokens:.*/max-num-batched-tokens: $BATCH/" "$OUT"
    # Build cudagraph_capture_sizes to cover all batch sizes up to SEQS
    # Powers of 2 from 1 to SEQS
    local SIZES="[1"
    local S=2
    while [ $S -le $SEQS ]; do
        SIZES="$SIZES,$S"
        S=$((S * 2))
    done
    SIZES="$SIZES]"
    sed -i "s/\"cudagraph_capture_sizes\": \[[^]]*\]/\"cudagraph_capture_sizes\": $SIZES/" "$OUT"
    echo "$OUT"
}

start_server() {
    local CONFIG=$1
    local NAME=$2
    local SEQS=${3:-4}  # pass seqs count for warmup
    docker stop vllm 2>/dev/null; docker rm vllm 2>/dev/null
    sleep 3
    docker run --rm -d \
      --name vllm --gpus all --ipc host --shm-size=16g --ulimit memlock=-1 \
      -v /data:/data -v /faststore:/faststore -v /data/cache:~/.cache \
      -e HF_HOME=/faststore/huggingface -e HF_HUB_OFFLINE=1 \
      -e VLLM_USE_FLASHINFER_MOE_FP4=1 -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass \
      -e PYTORCH_ALLOC_CONF=expandable_segments:True -e PYTHONUNBUFFERED=1 \
      -e PYTHONHASHSEED=0 -e PYTHONPATH=~/projects/vllm \
      vllm:runtime \
      bash -c "vllm serve --config $CONFIG 2>&1 | tee -a ~/projects/vllm/logs/vllm-sweep.log" \
      > /dev/null 2>&1

    for i in $(seq 1 120); do
        if docker exec vllm curl -s http://localhost:8000/health > /dev/null 2>&1; then
            # Warm up: trigger torch compile + CUDA graph capture at each batch size
            # Each new capture size triggers JIT compilation on first use
            # Without this, TTFT on first bench request includes compile overhead
            echo -n "  Warmup (torch compile" >&2
            # Trigger capture for each power-of-2 up to SEQS
            local CONC=1
            while [ $CONC -le $SEQS ]; do
                echo -n " bs=$CONC" >&2
                for j in $(seq 1 $CONC); do
                    docker exec vllm curl -s http://localhost:8000/v1/completions \
                        -H "Content-Type: application/json" \
                        -d '{"model":"default","prompt":"Warmup conc='$CONC' req='$j'","max_tokens":16}' > /dev/null 2>&1 &
                done
                wait
                CONC=$((CONC * 2))
            done
            # Extra stabilisation passes
            for w in 1 2 3; do
                docker exec vllm curl -s http://localhost:8000/v1/completions \
                    -H "Content-Type: application/json" \
                    -d '{"model":"default","prompt":"Stabilise pass '$w'","max_tokens":16}' > /dev/null 2>&1
            done
            echo ") done" >&2
            return 0
        fi
        if ! docker ps -q --filter name=vllm | grep -q .; then return 1; fi
        sleep 5
    done
    return 1
}

# Run one benchmark at rate=inf, return "throughput tpot_ms" or "FAIL"
bench_one() {
    local SEQS=$1
    local BATCH=$2
    local TAG="seqs${SEQS}_batch${BATCH}"
    local OUTDIR="$RESULT_DIR/$TAG"

    if [ -f "$OUTDIR/run=0.json" ]; then
        # Already have results
        python3 -c "
import json
d = json.load(open('$OUTDIR/run=0.json'))
print(f'{d.get(\"output_throughput\",0):.1f} {d.get(\"median_tpot_ms\",0):.2f} {d.get(\"median_ttft_ms\",0):.0f}')
" 2>/dev/null
        return
    fi

    local CFG=$(make_config $SEQS $BATCH)
    echo -n "  Starting seqs=$SEQS batch=$BATCH ... " >&2
    if ! start_server "$CFG" "$TAG" "$SEQS"; then
        echo "OOM/FAIL" >&2
        echo "FAIL"
        return
    fi
    echo -n "benchmarking ... " >&2
    mkdir -p "$OUTDIR"
    docker exec vllm vllm bench serve \
        --dataset-name random \
        --input-len $INPUT_LEN --output-len $OUTPUT_LEN \
        --num-prompts $NUM_PROMPTS \
        --request-rate inf --ignore-eos \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result --result-dir "$OUTDIR" --result-filename "run=0.json" \
        2>&1 > /dev/null

    if [ -f "$OUTDIR/run=0.json" ]; then
        local RES=$(python3 -c "
import json
d = json.load(open('$OUTDIR/run=0.json'))
done = d.get('completed',0)
if done == 0: print('FAIL')
else: print(f'{d.get(\"output_throughput\",0):.1f} {d.get(\"median_tpot_ms\",0):.2f} {d.get(\"median_ttft_ms\",0):.0f}')
" 2>/dev/null)
        echo "$RES" | awk '{printf "out=%s t/s  TPOT=%s ms\n", $1, $2}' >&2
        echo "$RES"
    else
        echo "FAIL" >&2
        echo "FAIL"
    fi
}

echo "============================================================"
echo "  Gemma 4 26B-A4B Adaptive Sweep"
echo "  Phase 1: Exponential probe (seqs doubling, rate=inf)"
echo "  Phase 2: Binary search around sweet spot"
echo "  Phase 3: Batch token sweep on winning seqs"
echo "============================================================"
echo ""

# ══════════════════════════════════════════════════════════════
# Phase 1: Exponential probe — find the throughput cliff
# ══════════════════════════════════════════════════════════════
echo "=== Phase 1: Exponential probe ==="
declare -A RESULTS
BEST_SEQS=1
BEST_THROUGHPUT=0
PREV_TPOT=0

for SEQS in 1 2 4 8 16 32 64; do
    # Scale batch with seqs
    BATCH=$((SEQS * 2048))
    [ $BATCH -lt 8192 ] && BATCH=8192
    [ $BATCH -gt 65536 ] && BATCH=65536

    RESULT=$(bench_one $SEQS $BATCH)
    if [ "$RESULT" = "FAIL" ]; then
        echo "  seqs=$SEQS: FAILED (OOM or crash)"
        echo "  Cliff found at seqs=$SEQS"
        CLIFF_SEQS=$SEQS
        break
    fi

    THROUGHPUT=$(echo "$RESULT" | awk '{print $1}')
    TPOT=$(echo "$RESULT" | awk '{print $2}')
    TTFT=$(echo "$RESULT" | awk '{print $3}')
    RESULTS[$SEQS]="$THROUGHPUT $TPOT $TTFT $BATCH"

    echo "  seqs=$SEQS batch=$BATCH: ${THROUGHPUT} t/s, TPOT=${TPOT}ms, TTFT=${TTFT}ms"

    # Check for throughput plateau or TPOT cliff (>2x degradation)
    if [ "$PREV_TPOT" != "0" ]; then
        RATIO=$(python3 -c "print(f'{$TPOT / $PREV_TPOT:.2f}')")
        if python3 -c "exit(0 if $TPOT / $PREV_TPOT > 2.0 else 1)"; then
            echo "  TPOT cliff detected: ${TPOT}ms vs ${PREV_TPOT}ms (${RATIO}x)"
            CLIFF_SEQS=$SEQS
            break
        fi
    fi

    if python3 -c "exit(0 if $THROUGHPUT > $BEST_THROUGHPUT else 1)"; then
        BEST_THROUGHPUT=$THROUGHPUT
        BEST_SEQS=$SEQS
    fi
    PREV_TPOT=$TPOT
    CLIFF_SEQS=$((SEQS * 2))  # no cliff yet
done

echo ""
echo "  Best so far: seqs=$BEST_SEQS at ${BEST_THROUGHPUT} t/s"
echo "  Cliff/limit at seqs=$CLIFF_SEQS"

# ══════════════════════════════════════════════════════════════
# Phase 2: Binary search between best and cliff
# ══════════════════════════════════════════════════════════════
if [ $CLIFF_SEQS -gt $((BEST_SEQS * 2)) ]; then
    echo ""
    echo "=== Phase 2: Binary search (seqs=$BEST_SEQS to $CLIFF_SEQS) ==="
    LO=$BEST_SEQS
    HI=$CLIFF_SEQS

    for ITER in 1 2 3; do
        MID=$(( (LO + HI) / 2 ))
        # Round to nearest power-friendly number
        [ $MID -eq $LO ] && break
        [ $MID -eq $HI ] && break

        BATCH=$((MID * 2048))
        [ $BATCH -lt 8192 ] && BATCH=8192
        [ $BATCH -gt 65536 ] && BATCH=65536

        RESULT=$(bench_one $MID $BATCH)
        if [ "$RESULT" = "FAIL" ]; then
            echo "  seqs=$MID: FAILED → search lower"
            HI=$MID
            continue
        fi

        THROUGHPUT=$(echo "$RESULT" | awk '{print $1}')
        TPOT=$(echo "$RESULT" | awk '{print $2}')
        TTFT=$(echo "$RESULT" | awk '{print $3}')
        echo "  seqs=$MID batch=$BATCH: ${THROUGHPUT} t/s, TPOT=${TPOT}ms"

        if python3 -c "exit(0 if $THROUGHPUT > $BEST_THROUGHPUT else 1)"; then
            BEST_THROUGHPUT=$THROUGHPUT
            BEST_SEQS=$MID
            LO=$MID
        else
            HI=$MID
        fi
    done
    echo "  Optimal seqs: $BEST_SEQS at ${BEST_THROUGHPUT} t/s"
fi

# ══════════════════════════════════════════════════════════════
# Phase 3: Batch token sweep on winning seqs
# ══════════════════════════════════════════════════════════════
echo ""
echo "=== Phase 3: Batch token sweep (seqs=$BEST_SEQS) ==="
BEST_BATCH=8192
BEST_BATCH_THROUGHPUT=0

for BATCH in 4096 8192 16384 32768 65536; do
    RESULT=$(bench_one $BEST_SEQS $BATCH)
    if [ "$RESULT" = "FAIL" ]; then
        echo "  batch=$BATCH: FAILED"
        continue
    fi
    THROUGHPUT=$(echo "$RESULT" | awk '{print $1}')
    TPOT=$(echo "$RESULT" | awk '{print $2}')
    echo "  seqs=$BEST_SEQS batch=$BATCH: ${THROUGHPUT} t/s, TPOT=${TPOT}ms"
    if python3 -c "exit(0 if $THROUGHPUT > $BEST_BATCH_THROUGHPUT else 1)"; then
        BEST_BATCH_THROUGHPUT=$THROUGHPUT
        BEST_BATCH=$BATCH
    fi
done

# ══════════════════════════════════════════════════════════════
# Restore and report
# ══════════════════════════════════════════════════════════════
echo ""
echo "=== Restoring original vLLM ==="
docker stop vllm 2>/dev/null; docker rm vllm 2>/dev/null; sleep 3
cd /data/docker && LLM_MODEL=gemma-4-26b-a4b docker compose up -d vllm 2>/dev/null
for i in $(seq 1 60); do
    docker exec vllm curl -s http://localhost:8000/health > /dev/null 2>&1 && echo "Restored" && break
    sleep 2
done

echo ""
echo "============================================================"
echo "  RESULTS"
echo "============================================================"
echo ""
echo "  Optimal: max-num-seqs=$BEST_SEQS  max-num-batched-tokens=$BEST_BATCH"
echo "  Throughput: ${BEST_BATCH_THROUGHPUT} t/s"
echo ""
printf "%-30s %8s %8s %8s\n" "Config" "Out_t/s" "TPOT_ms" "TTFT_ms"
echo "--------------------------------------------------------------"
for f in $(find "$RESULT_DIR" -name "run=0.json" | sort); do
    python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
parts = sys.argv[1].split('/')
config = parts[-3]
done = d.get('completed', 0)
if done == 0: sys.exit()
print(f'{config:<30} {d.get(\"output_throughput\",0):>8.1f} {d.get(\"median_tpot_ms\",0):>8.2f} {d.get(\"median_ttft_ms\",0):>8.0f}')
" "$f" 2>/dev/null
done
