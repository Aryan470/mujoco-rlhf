#!/bin/bash
set -euo pipefail

# ------------------------------------------------------------------
# Script: train.sh
# Purpose: Train the policy model for the next batch and log progress
# ------------------------------------------------------------------

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$BASE_DIR/logs/train"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +"%Y-%m-%d_%H-%M-%S")"
LOGFILE="$LOG_DIR/train_$TIMESTAMP.log"

echo "-----------------------------------------------------" | tee -a "$LOGFILE"
echo "TRAIN.SH STARTED  $(date)" | tee -a "$LOGFILE"
echo "Working directory: $BASE_DIR" | tee -a "$LOGFILE"
echo "Logfile:           $LOGFILE" | tee -a "$LOGFILE"
echo "-----------------------------------------------------" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# ---------------------------
# CUSTOM TRAINING COMMAND HERE
# ---------------------------
# Example placeholder:
# python train_model.py --config configs/latest.yaml

echo "[train.sh] Starting model training..." | tee -a "$LOGFILE"

# 🔥 REPLACE THIS with your actual training command:
python3 train_model.py 2>&1 | tee -a "$LOGFILE"

STATUS=$?
echo "" | tee -a "$LOGFILE"

if [[ $STATUS -eq 0 ]]; then
    echo "[train.sh] Training completed successfully." | tee -a "$LOGFILE"
else
    echo "[train.sh] ERROR: Training failed with exit code $STATUS" | tee -a "$LOGFILE"
    exit $STATUS
fi

echo "TRAIN.SH FINISHED $(date)" | tee -a "$LOGFILE"
echo "-----------------------------------------------------" | tee -a "$LOGFILE"
