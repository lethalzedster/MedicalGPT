#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

set +u
source /share/home/beiyou2/miniconda3/etc/profile.d/conda.sh
conda activate rs_grpo
set -u
export PYTHONNOUSERSITE=1

MODEL_PATH="${MODEL_PATH:-outputs-finalign-dpo-qwen25-7b-merged}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"
OUT_DIR="${OUT_DIR:-reports/finalign/dpo}"
mkdir -p "$OUT_DIR"

for task in fineval finqa convfinqa sentiment; do
  case "$task" in
    fineval|sentiment) MAX_NEW_TOKENS="${MAX_NEW_TOKENS_CHOICE:-8}" ;;
    *) MAX_NEW_TOKENS="${MAX_NEW_TOKENS_QA:-32}" ;;
  esac
  python tools/finalign_generate.py \
    --model_name_or_path "$MODEL_PATH" \
    --input_file "data/finance_benchmarks/${task}/eval.jsonl" \
    --output_file "${OUT_DIR}/${task}_pred.jsonl" \
    --max_samples "$MAX_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --load_in_4bit
  python tools/finalign_eval.py \
    --prediction_file "${OUT_DIR}/${task}_pred.jsonl" \
    --output_file "${OUT_DIR}/${task}_metrics.json"
done
