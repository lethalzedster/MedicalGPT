#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-finalign_dpo_data}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${LOG_FILE:-logs/finalign_dpo_data.log}"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  echo "attach with: tmux attach -t $SESSION_NAME"
  exit 0
fi

tmux new-session -d -s "$SESSION_NAME" -n build "cd '$ROOT_DIR' && mkdir -p logs && bash scripts/run_deepseek_dpo_data.sh 2>&1 | tee -a '$LOG_FILE'"
tmux split-window -h -t "$SESSION_NAME:0" "cd '$ROOT_DIR' && touch '$LOG_FILE' && tail -n 120 -F '$LOG_FILE'"
tmux select-pane -t "$SESSION_NAME:0.0"
tmux select-layout -t "$SESSION_NAME:0" even-horizontal

echo "started tmux session: $SESSION_NAME"
echo "attach with: tmux attach -t $SESSION_NAME"
