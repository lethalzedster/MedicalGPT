#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-fin_sft}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  echo "attach with: tmux attach -t $SESSION_NAME"
  exit 0
fi

tmux new-session -d -s "$SESSION_NAME" -n train "cd '$ROOT_DIR' && bash scripts/run_finance_sft.sh"
tmux split-window -h -t "$SESSION_NAME:0" "cd '$ROOT_DIR' && touch logs/fin_sft.log && tail -n 100 -F logs/fin_sft.log"
tmux select-pane -t "$SESSION_NAME:0.0"
tmux select-layout -t "$SESSION_NAME:0" even-horizontal

echo "started tmux session: $SESSION_NAME"
echo "attach with: tmux attach -t $SESSION_NAME"
