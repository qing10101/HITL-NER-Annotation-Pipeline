#!/usr/bin/env bash
# Control run for Scheme A with t5-base (native <extra_id_k> span infilling).
#   bash benchmark/threat/run_recon_t5base.sh [SEED]   (env: LR=3e-4 BS=8 ACC=2 EPOCHS=3 WAIT=1)
# With WAIT=1 it first waits until run_recon_fp32.sh has written its "finished"
# line, so it can be queued behind the T5Gemma rerun on the single GPU.
set -u
SEED="${1:-42}"; LR="${LR:-3e-4}"; BS="${BS:-8}"; ACC="${ACC:-2}"; EPOCHS="${EPOCHS:-3}"; WAIT="${WAIT:-1}"
cd "$(dirname "$0")/../.." || exit 1
PY=".venv-threat/Scripts/python.exe"
RES="benchmark/threat/results"; LOGS="$RES/logs"; mkdir -p "$LOGS"
STATUS="$RES/run_all_status.txt"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_SYMLINKS_WARNING=1
if [ "$WAIT" = "1" ]; then
  while ! grep -q "run_recon_fp32 finished" "$STATUS"; do sleep 60; done
fi
echo "run_recon_t5base started $(date '+%F %T') seed=$SEED lr=$LR bs=$BS accum=$ACC epochs=$EPOCHS" | tee -a "$STATUS"
step () {
  local name="$1"; shift
  local t0=$(date +%s)
  echo "[$(date '+%F %T')] START $name" | tee -a "$STATUS"
  "$@" > "$LOGS/$name.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] END   $name rc=$rc elapsed=$(( $(date +%s) - t0 ))s" | tee -a "$STATUS"
}
COMMON="--model t5-base --seed $SEED --fp32 --tag fp32 --lr $LR --bs $BS --accum $ACC --epochs $EPOCHS"
# shellcheck disable=SC2086
step recon_t5-base_adaptive_fp32_s$SEED    $PY benchmark/threat/recon_t5.py --mode adaptive    $COMMON
# shellcheck disable=SC2086
step recon_t5-base_nonadaptive_fp32_s$SEED $PY benchmark/threat/recon_t5.py --mode nonadaptive $COMMON
step summarize $PY benchmark/threat/summarize.py
echo "run_recon_t5base finished $(date '+%F %T')" | tee -a "$STATUS"
