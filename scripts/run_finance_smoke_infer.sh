#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

set +u
source /share/home/beiyou2/miniconda3/etc/profile.d/conda.sh
conda activate rs_grpo
set -u
export PYTHONNOUSERSITE=1

mkdir -p reports/finalign/generated

MODEL_PATH="${MODEL_PATH:-outputs-finalign-sft-qwen25-7b-merged}"
INPUT_FILE="${INPUT_FILE:-data/finance_benchmarks/finqa/eval.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-reports/finalign/smoke_sft_samples.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"

python tools/finalign_generate.py \
  --model_name_or_path "$MODEL_PATH" \
  --input_file "$INPUT_FILE" \
  --output_file "$OUTPUT_FILE" \
  --max_samples "$MAX_SAMPLES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --load_in_4bit
