#!/usr/bin/env bash
set -euo pipefail
cd /workspace/projects/baseline_genyes8
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
RUN=/workspace/runs/baseline_genyes8/20261019_s8_finetune
echo "Started: $(date -Is)"
python train_server_s8.py --smoke > "$RUN/smoke.log" 2>&1
python train_server_s8.py > "$RUN/train.log" 2>&1
echo "Finished: $(date -Is)"
