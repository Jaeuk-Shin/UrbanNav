#!/bin/bash
# Multi-GPU DPVO batch processor (single server, no SLURM needed).
# Spawns one worker per GPU; each pulls videos from a shared queue.
# Safe to re-run: already-processed videos are skipped automatically.

set -e

# ── Configuration ─────────────────────────────────────────────────────
NUM_GPUS=4
VIDEO_DIR=dataset/citywalk_2min/videos
CALIB_FILE=calib/citywalk.txt
OUTPUT_DIR=dataset/citywalk_2min/poses
STRIDE=6
NETWORK=dpvo.pth
CONFIG=config/default.yaml

# ── Run ───────────────────────────────────────────────────────────────
mkdir -p logs

conda activate dpvo

python dpvo_slurm.py \
    --videodir "$VIDEO_DIR" \
    --calib "$CALIB_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --network "$NETWORK" \
    --config "$CONFIG" \
    --stride "$STRIDE" \
    --num_gpus "$NUM_GPUS" \
    --save_trajectory \
    2>&1 | tee logs/dpvo_$(date +%Y%m%d_%H%M%S).log
