#!/bin/bash
set -euo pipefail


BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

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


echo "[train.sh] Scanning for batch_*_results.json in data/metadata..." | tee -a "$LOGFILE"

max_i=-1
shopt -s nullglob
for f in "$BASE_DIR"/data/metadata/batch_*_results.json; do
    fname="$(basename "$f")"
    idx="${fname#batch_}"
    idx="${idx%_results.json}"

    if [[ "$idx" =~ ^[0-9]+$ ]]; then
        if (( idx > max_i )); then
            max_i=$idx
        fi
    fi
done
shopt -u nullglob

if (( max_i < 0 )); then
    echo "[train.sh] ERROR: No batch_*_results.json files found in data/metadata." | tee -a "$LOGFILE"
    exit 1
fi

phase=$((max_i + 1))

echo "[train.sh] Using latest batch index i=$max_i" | tee -a "$LOGFILE"
echo "[train.sh] Calling train_proc with phase_num=i+1=$phase" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"


echo "[train.sh] Starting retrain via train_proc..." | tee -a "$LOGFILE"

python3 -u - <<PY 2>&1 | tee -a "$LOGFILE"
from train_policy import train_proc

i = ${max_i}
phase_num = ${phase}

print(f"[python] Retraining with batch index i={i}, phase_num={phase_num}")
train_proc(phase_num)
PY

STATUS=${PIPESTATUS[0]}
echo "" | tee -a "$LOGFILE"

if [[ $STATUS -eq 0 ]]; then
    echo "[train.sh] Training completed successfully." | tee -a "$LOGFILE"
else
    echo "[train.sh] ERROR: Training failed with exit code $STATUS" | tee -a "$LOGFILE"
    exit $STATUS
fi

echo "TRAIN.SH FINISHED $(date)" | tee -a "$LOGFILE"
echo "-----------------------------------------------------" | tee -a "$LOGFILE"
