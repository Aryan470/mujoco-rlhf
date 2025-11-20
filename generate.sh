#!/bin/bash
set -euo pipefail

# ------------------------------------------------------------------
# Script: generate.sh
# Purpose: Generate new clips using the latest trained policy
# ------------------------------------------------------------------

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$BASE_DIR/logs/generate"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +"%Y-%m-%d_%H-%M-%S")"
LOGFILE="$LOG_DIR/generate_$TIMESTAMP.log"

echo "-----------------------------------------------------" | tee -a "$LOGFILE"
echo "GENERATE.SH STARTED  $(date)" | tee -a "$LOGFILE"
echo "Working directory: $BASE_DIR" | tee -a "$LOGFILE"
echo "Logfile:           $LOGFILE" | tee -a "$LOGFILE"
echo "-----------------------------------------------------" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# --------------------------------
# CUSTOM GENERATE COMMAND GOES HERE
# --------------------------------
# Example placeholder:
# python generate_clips.py --policy models/checkpoints/policy.pt

echo "[generate.sh] Generating new clips..." | tee -a "$LOGFILE"

# 🔥 REPLACE THIS with your actual generation command:
python3 generate_clips.py 2>&1 | tee -a "$LOGFILE"

STATUS=$?
echo "" | tee -a "$LOGFILE"

if [[ $STATUS -eq 0 ]]; then
    echo "[generate.sh] Clip generation completed successfully." | tee -a "$LOGFILE"
else
    echo "[generate.sh] ERROR: Clip generation failed with exit code $STATUS" | tee -a "$LOGFILE"
    exit $STATUS
fi

echo "GENERATE.SH FINISHED $(date)" | tee -a "$LOGFILE"
echo "-----------------------------------------------------" | tee -a "$LOGFILE"
