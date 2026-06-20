#!/usr/bin/env bash
set -euo pipefail

REWARD_DATA_DIR="${REWARD_DATA_DIR:-./data/finance_reward_deepseek}" \
OUTPUT_DIR="${OUTPUT_DIR:-outputs-finalign-dpo-deepseek-qwen25-7b}" \
MAX_STEPS="${MAX_STEPS:-1500}" \
bash scripts/run_finance_dpo.sh
