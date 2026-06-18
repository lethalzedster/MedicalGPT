#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-finalign_tb}"
PORT="${2:-16006}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  echo "attach with: tmux attach -t $SESSION_NAME"
  exit 0
fi

tmux new-session -d -s "$SESSION_NAME" "cd '$ROOT_DIR' && \
  set +u && \
  source /share/home/beiyou2/miniconda3/etc/profile.d/conda.sh && \
  conda activate rs_grpo && \
  set -u && \
  export PYTHONNOUSERSITE=1 && \
  tensorboard --logdir_spec SFT:outputs-finalign-sft-qwen25-7b/runs,DPO:outputs-finalign-dpo-qwen25-7b/runs --host 127.0.0.1 --port '$PORT'"

echo "started tensorboard tmux session: $SESSION_NAME"
echo "server port: $PORT"
echo "attach with: tmux attach -t $SESSION_NAME"
echo "local forward: ssh -L ${PORT}:127.0.0.1:${PORT} gpu4090"
