#!/usr/bin/env bash
# Full evaluation + visualization pipeline for ProteinChameleon Stage II.
#
# Usage:
#   ./run_eval.sh [--n-gen N] [--n-gen-interleaved N] [--out-dir DIR] [--num-gpus N]
#
# Defaults: 500 alignment, 100 interleaved, out-dir=eval_results, num-gpus=4

set -euo pipefail
cd "$(dirname "$0")"

PROTEINCHAMALEON_PY="$HOME/miniconda3/envs/proteinchamaleon/bin/python"
GEOBPE_PY="$HOME/miniconda3/envs/geobpe/bin/python"

CKPT="/home/steven/checkpoints/stage2/final"
BPE="/home/steven/PT-BPE_ckpts/bpe_post_init.pkl"
OUT_DIR="/home/steven/eval_results"
N_GEN=500
N_GEN_INTERLEAVED=100
NUM_GPUS=4

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-gen)              N_GEN="$2";              shift 2 ;;
        --n-gen-interleaved)  N_GEN_INTERLEAVED="$2";  shift 2 ;;
        --out-dir)            OUT_DIR="$2";             shift 2 ;;
        --num-gpus)           NUM_GPUS="$2";            shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo " ProteinChameleon Evaluation Pipeline"
echo " ckpt:              $CKPT"
echo " out-dir:           $OUT_DIR"
echo " n-gen (align):     $N_GEN"
echo " n-gen (interleav): $N_GEN_INTERLEAVED"
echo " num-gpus:          $NUM_GPUS"
echo "============================================"

# ── Helper: launch N parallel shards on N GPUs, wait for all ─────────────────
run_parallel() {
    local TASK="$1"   # "alignment" or "interleaved"
    local N="$2"      # n-gen or n-gen-interleaved
    local LOG_PREFIX="$3"

    local PIDS=()
    for (( GPU=0; GPU<NUM_GPUS; GPU++ )); do
        echo "  Launching $TASK shard $GPU/$NUM_GPUS on GPU $GPU..."
        mkdir -p "$OUT_DIR"
        CUDA_VISIBLE_DEVICES=$GPU $PROTEINCHAMALEON_PY scripts/eval_stage2.py \
            --ckpt                "$CKPT" \
            --out-dir             "$OUT_DIR" \
            --n-gen               "$( [ "$TASK" = alignment ]    && echo "$N" || echo 0 )" \
            --n-gen-interleaved   "$( [ "$TASK" = interleaved ]  && echo "$N" || echo 0 )" \
            --shard               "$GPU" \
            --num-shards          "$NUM_GPUS" \
            --gen-only \
            > "$OUT_DIR/${LOG_PREFIX}_gpu${GPU}.log" 2>&1 &
        PIDS+=($!)
    done

    local FAILED=0
    for PID in "${PIDS[@]}"; do
        if ! wait "$PID"; then
            echo "  WARNING: a shard process failed (PID $PID)"
            FAILED=1
        fi
    done
    if [ "$FAILED" -eq 1 ]; then
        echo "Shards failed — check $OUT_DIR/${LOG_PREFIX}_gpu*.log"
        exit 1
    fi
    echo "  All $TASK shards done."
}

# ── Step 1a: Alignment inference ─────────────────────────────────────────────
echo ""
if [ "$N_GEN" -gt 0 ]; then
    echo "[1a/3] Running alignment inference on $NUM_GPUS GPUs..."
    run_parallel alignment "$N_GEN" align
else
    echo "[1a/3] Skipping alignment inference (n-gen=0)"
fi

# ── Step 1b: Interleaved inference ───────────────────────────────────────────
echo ""
if [ "$N_GEN_INTERLEAVED" -gt 0 ]; then
    echo "[1b/3] Running interleaved inference on $NUM_GPUS GPUs..."
    run_parallel interleaved "$N_GEN_INTERLEAVED" interleaved
else
    echo "[1b/3] Skipping interleaved inference (n-gen-interleaved=0)"
fi

# ── Helper: run visualize_eval in a watch loop until inference PIDs are done ──
watch_and_visualize() {
    local MODE="$1"
    local EXAMPLES_DIR="$2"
    local INFERENCE_PIDS=("${@:3}")

    echo "  Starting background PDF watcher for $MODE..."
    (
        while true; do
            $GEOBPE_PY scripts/visualize_eval.py \
                --mode         "$MODE" \
                --examples-dir "$EXAMPLES_DIR" \
                --bpe          "$BPE" 2>/dev/null
            # Check if all inference processes are done
            ALL_DONE=1
            for PID in "${INFERENCE_PIDS[@]}"; do
                if kill -0 "$PID" 2>/dev/null; then
                    ALL_DONE=0
                    break
                fi
            done
            [ "$ALL_DONE" -eq 1 ] && break
            sleep 30
        done
        # Final pass to catch any last examples
        $GEOBPE_PY scripts/visualize_eval.py \
            --mode         "$MODE" \
            --examples-dir "$EXAMPLES_DIR" \
            --bpe          "$BPE" 2>/dev/null
        echo "  PDF watcher done for $MODE."
    ) &
    echo $!
}

# ── Step 1a: Alignment inference ─────────────────────────────────────────────
echo ""
if [ "$N_GEN" -gt 0 ]; then
    echo "[1a/3] Running alignment inference on $NUM_GPUS GPUs..."
    ALIGN_PIDS=()
    for (( GPU=0; GPU<NUM_GPUS; GPU++ )); do
        mkdir -p "$OUT_DIR"
        CUDA_VISIBLE_DEVICES=$GPU $PROTEINCHAMALEON_PY scripts/eval_stage2.py \
            --ckpt              "$CKPT" \
            --out-dir           "$OUT_DIR" \
            --n-gen             "$N_GEN" \
            --n-gen-interleaved 0 \
            --shard             "$GPU" \
            --num-shards        "$NUM_GPUS" \
            --gen-only \
            > "$OUT_DIR/align_gpu${GPU}.log" 2>&1 &
        ALIGN_PIDS+=($!)
    done

    ALIGN_VIZ_PID=$(watch_and_visualize alignment "$OUT_DIR/alignment" "${ALIGN_PIDS[@]}")

    for PID in "${ALIGN_PIDS[@]}"; do wait "$PID" || true; done
    wait "$ALIGN_VIZ_PID" 2>/dev/null || true
    echo "  Alignment inference + PDFs done."
else
    echo "[1a/3] Skipping alignment (n-gen=0)"
fi

# ── Step 1b: Interleaved inference ───────────────────────────────────────────
echo ""
if [ "$N_GEN_INTERLEAVED" -gt 0 ]; then
    echo "[1b/3] Running interleaved inference on $NUM_GPUS GPUs..."
    INTER_PIDS=()
    for (( GPU=0; GPU<NUM_GPUS; GPU++ )); do
        mkdir -p "$OUT_DIR"
        CUDA_VISIBLE_DEVICES=$GPU $PROTEINCHAMALEON_PY scripts/eval_stage2.py \
            --ckpt              "$CKPT" \
            --out-dir           "$OUT_DIR" \
            --n-gen             0 \
            --n-gen-interleaved "$N_GEN_INTERLEAVED" \
            --shard             "$GPU" \
            --num-shards        "$NUM_GPUS" \
            --gen-only \
            > "$OUT_DIR/interleaved_gpu${GPU}.log" 2>&1 &
        INTER_PIDS+=($!)
    done

    INTER_VIZ_PID=$(watch_and_visualize interleaved "$OUT_DIR/interleaved" "${INTER_PIDS[@]}")

    for PID in "${INTER_PIDS[@]}"; do wait "$PID" || true; done
    wait "$INTER_VIZ_PID" 2>/dev/null || true
    echo "  Interleaved inference + PDFs done."
else
    echo "[1b/3] Skipping interleaved (n-gen-interleaved=0)"
fi

echo ""
echo "============================================"
echo " Done."
echo " Alignment:    $OUT_DIR/alignment/{accession}/{accession}.{pdf,json}"
echo " Interleaved:  $OUT_DIR/interleaved/{accession}/{accession}.{pdf,json}"
echo "============================================"
