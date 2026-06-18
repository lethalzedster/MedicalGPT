#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SESSION="${SESSION:-finalign_pipeline}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/finalign_pipeline.log}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" "bash -lc '
  set -euo pipefail
  cd \"$ROOT_DIR\"
  mkdir -p \"$LOG_DIR\"
  {
    echo \"[1/8] merge SFT adapter\"
    bash scripts/merge_finance_sft_adapter.sh

    echo \"[2/8] smoke inference\"
    bash scripts/run_finance_smoke_infer.sh

    echo \"[3/8] eval Base\"
    MAX_SAMPLES=\"$MAX_SAMPLES\" bash scripts/run_finance_eval_base.sh

    echo \"[4/8] eval SFT\"
    MAX_SAMPLES=\"$MAX_SAMPLES\" bash scripts/run_finance_eval_sft.sh

    echo \"[5/8] DPO smoke\"
    bash scripts/run_finance_dpo_smoke.sh

    echo \"[6/8] DPO formal\"
    bash scripts/run_finance_dpo.sh

    echo \"[7/8] merge DPO adapter\"
    bash scripts/merge_finance_dpo_adapter.sh

    echo \"[8/8] eval DPO and build A/B pairs\"
    MAX_SAMPLES=\"$MAX_SAMPLES\" bash scripts/run_finance_eval_dpo.sh
    bash scripts/run_finance_ab_judge.sh

    echo \"FinAlign pipeline finished at \$(date)\"
  } 2>&1 | tee -a \"$LOG_FILE\"
'"

tmux split-window -h -t "$SESSION" "bash -lc 'cd \"$ROOT_DIR\" && touch \"$LOG_FILE\" && tail -n 120 -F \"$LOG_FILE\"'"
tmux select-pane -t "$SESSION":0.0

echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
