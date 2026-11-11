#!/usr/bin/env bash
# Re-run Scheme A (T5Gemma 2 reconstruction attacker) with fp32 master weights.
#   bash benchmark/threat/run_recon_fp32.sh [SEED]      (env: LR=1e-4 BS=2 ACC=8 EPOCHS=3)
# Background: the first pass trained bf16-stored weights with Adafactor; the model
# learned only the output prior and ignored the context (see results/logs and the
# design doc, section 7).  --fp32 keeps the 368M trainable weights in fp32.
set -u
SEED="${1:-42}"; LR="${LR:-1e-4}"; BS="${BS:-2}"; ACC="${ACC:-8}"; EPOCHS="${EPOCHS:-3}"
cd "$(dirname "$0")/../.." || exit 1
PY=".venv-threat/Scripts/python.exe"
RES="benchmark/threat/results"; LOGS="$RES/logs"; mkdir -p "$LOGS"
STATUS="$RES/run_all_status.txt"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_SYMLINKS_WARNING=1
echo "run_recon_fp32 started $(date '+%F %T') seed=$SEED lr=$LR bs=$BS accum=$ACC epochs=$EPOCHS" | tee -a "$STATUS"
step () {
  local name="$1"; shift
  local t0=$(date +%s)
  echo "[$(date '+%F %T')] START $name" | tee -a "$STATUS"
  "$@" > "$LOGS/$name.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] END   $name rc=$rc elapsed=$(( $(date +%s) - t0 ))s" | tee -a "$STATUS"
}
COMMON="--model t5gemma2 --seed $SEED --fp32 --tag fp32 --lr $LR --bs $BS --accum $ACC --epochs $EPOCHS"
# shellcheck disable=SC2086
step recon_t5gemma2_adaptive_fp32_s$SEED    $PY benchmark/threat/recon_t5.py --mode adaptive    $COMMON
# shellcheck disable=SC2086
step recon_t5gemma2_nonadaptive_fp32_s$SEED $PY benchmark/threat/recon_t5.py --mode nonadaptive $COMMON
step summarize $PY benchmark/threat/summarize.py
echo "run_recon_fp32 finished $(date '+%F %T')" | tee -a "$STATUS"
