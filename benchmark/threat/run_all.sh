#!/usr/bin/env bash
# Sequential, unattended run of the threat experiments (one seed).
#   bash benchmark/threat/run_all.sh [SEED]
# GPU holds one job at a time, so steps run back to back; a failing step is
# logged and the chain continues.  Per-step logs: benchmark/threat/results/logs/.
# NeoBERT arguments are read from results/neobert_args.txt at step start so they
# can be adjusted while the T5Gemma steps are still running.
set -u
SEED="${1:-42}"
cd "$(dirname "$0")/../.." || exit 1
PY=".venv-threat/Scripts/python.exe"
RES="benchmark/threat/results"
LOGS="$RES/logs"
mkdir -p "$LOGS"
STATUS="$RES/run_all_status.txt"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo "run_all started $(date '+%F %T') seed=$SEED" | tee -a "$STATUS"

step () {  # step NAME CMD...
  local name="$1"; shift
  local t0=$(date +%s)
  echo "[$(date '+%F %T')] START $name" | tee -a "$STATUS"
  "$@" > "$LOGS/$name.log" 2>&1
  local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  echo "[$(date '+%F %T')] END   $name rc=$rc elapsed=${dt}s" | tee -a "$STATUS"
}

NEOBERT_ARGS_FILE="$RES/neobert_args.txt"
neobert_args () { [ -f "$NEOBERT_ARGS_FILE" ] && cat "$NEOBERT_ARGS_FILE" || echo ""; }

# Scheme A: T5Gemma 2 reconstruction attacker (pipeline validated in debug runs)
step recon_t5gemma2_adaptive_s$SEED    $PY benchmark/threat/recon_t5.py --model t5gemma2 --mode adaptive    --seed $SEED
step recon_t5gemma2_nonadaptive_s$SEED $PY benchmark/threat/recon_t5.py --model t5gemma2 --mode nonadaptive --seed $SEED

# Scheme B: NeoBERT presence attacker (args file lets us tune memory settings late)
# shellcheck disable=SC2046
step presence_neobert_raw_s$SEED      $PY benchmark/threat/presence_clf.py --model neobert --train-on raw      --seed $SEED $(neobert_args)
# shellcheck disable=SC2046
step presence_neobert_gold_del_s$SEED $PY benchmark/threat/presence_clf.py --model neobert --train-on gold_del --seed $SEED $(neobert_args)

# Scheme B: TF-IDF baseline for the same seed (seconds)
step presence_tfidf_raw_s$SEED      $PY benchmark/threat/presence_clf.py --model tfidf --train-on raw      --seed $SEED
step presence_tfidf_gold_del_s$SEED $PY benchmark/threat/presence_clf.py --model tfidf --train-on gold_del --seed $SEED

step summarize $PY benchmark/threat/summarize.py
echo "run_all finished $(date '+%F %T')" | tee -a "$STATUS"
