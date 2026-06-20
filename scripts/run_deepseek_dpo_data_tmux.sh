#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-finalign_dpo_data}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${LOG_FILE:-logs/finalign_dpo_data.log}"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is required in the current shell environment."
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  echo "attach with: tmux attach -t $SESSION_NAME"
  exit 0
fi

tmux new-session -d -s "$SESSION_NAME" -n build \
  "cd '$ROOT_DIR' && mkdir -p logs && env \
    DEEPSEEK_API_KEY='$DEEPSEEK_API_KEY' \
    DEEPSEEK_MODEL='${DEEPSEEK_MODEL:-deepseek-v4-flash}' \
    DEEPSEEK_BASE_URL='${DEEPSEEK_BASE_URL:-https://api.deepseek.com}' \
    MAX_PROMPTS='${MAX_PROMPTS:-12000}' \
    MAX_JUDGE_SAMPLES='${MAX_JUDGE_SAMPLES:-10000}' \
    MAX_NEW_TOKENS='${MAX_NEW_TOKENS:-128}' \
    GEN_BATCH_SIZE='${GEN_BATCH_SIZE:-8}' \
    bash scripts/run_deepseek_dpo_data.sh 2>&1 | tee -a '$LOG_FILE'"
tmux split-window -h -t "$SESSION_NAME:0" "cd '$ROOT_DIR' && touch '$LOG_FILE' && tail -n 120 -F '$LOG_FILE'"
tmux select-pane -t "$SESSION_NAME:0.0"
tmux select-layout -t "$SESSION_NAME:0" even-horizontal

echo "started tmux session: $SESSION_NAME"
echo "attach with: tmux attach -t $SESSION_NAME"
