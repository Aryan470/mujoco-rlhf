#!/bin/bash
set -euo pipefail

# ------------------------------------------------------------------
# Script: generate.sh
# Purpose: Infer latest phase and generate new clips using generate_clips.py
# ------------------------------------------------------------------

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

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

# -----------------------------------------------------
# Detect highest i such that data/metadata/batch_{i}_results.json exists
# We will generate clips for phase = i + 1
# -----------------------------------------------------
echo "[generate.sh] Scanning for batch_*_results.json in data/metadata..." | tee -a "$LOGFILE"

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
    echo "[generate.sh] ERROR: No batch_*_results.json files found in data/metadata." | tee -a "$LOGFILE"
    echo "[generate.sh] You probably need to label an initial batch and/or run train.sh first." | tee -a "$LOGFILE"
    exit 1
fi

phase=$((max_i + 1))

echo "[generate.sh] Latest labeled batch index i=$max_i" | tee -a "$LOGFILE"
echo "[generate.sh] Will generate clips for phase (iteration_idx) = i+1 = $phase" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# Optional sanity check: make sure models for this phase exist
if [[ ! -f "$BASE_DIR/data/$phase/models/checkpoints/policy.pt" ]] \
   || [[ ! -f "$BASE_DIR/data/$phase/models/checkpoints/reward.pt" ]]; then
    echo "[generate.sh] ERROR: Expected trained models not found for phase $phase." | tee -a "$LOGFILE"
    echo "[generate.sh] Looked for:" | tee -a "$LOGFILE"
    echo "  data/$phase/models/checkpoints/policy.pt" | tee -a "$LOGFILE"
    echo "  data/$phase/models/checkpoints/reward.pt" | tee -a "$LOGFILE"
    echo "[generate.sh] Make sure train.sh successfully produced this phase before generating clips." | tee -a "$LOGFILE"
    exit 1
fi

# -----------------------------------------------------
# Run generate_clips.py for this phase
# generate_clips.py expects:
#   --base_path  (we use 'data')
#   --iteration_idx (the phase number)
# -----------------------------------------------------
echo "[generate.sh] Generating new clips with generate_clips.py --base_path data --iteration_idx $phase" | tee -a "$LOGFILE"

python3 -u generate_clips.py \
    --base_path data \
    --iteration_idx "$phase" \
    2>&1 | tee -a "$LOGFILE"

STATUS=${PIPESTATUS[0]}
echo "" | tee -a "$LOGFILE"

if [[ $STATUS -eq 0 ]]; then
    echo "[generate.sh] Clip generation completed successfully." | tee -a "$LOGFILE"
    echo "[generate.sh] Metadata should be in: data/metadata/batch_${phase}.json" | tee -a "$LOGFILE"
else
    echo "[generate.sh] ERROR: Clip generation failed with exit code $STATUS" | tee -a "$LOGFILE"
    exit $STATUS
fi

echo "GENERATE.SH FINISHED $(date)" | tee -a "$LOGFILE"
echo "-----------------------------------------------------" | tee -a "$LOGFILE"
