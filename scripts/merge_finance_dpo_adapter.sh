#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

set +u
source /share/home/beiyou2/miniconda3/etc/profile.d/conda.sh
conda activate rs_grpo
set -u
export PYTHONNOUSERSITE=1

BASE_MODEL="${BASE_MODEL:-outputs-finalign-sft-qwen25-7b-merged}"
LORA_MODEL="${LORA_MODEL:-outputs-finalign-dpo-qwen25-7b}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs-finalign-dpo-qwen25-7b-merged}"

python tools/merge_peft_adapter.py \
  --base_model "$BASE_MODEL" \
  --tokenizer_path "$BASE_MODEL" \
  --lora_model "$LORA_MODEL" \
  --output_dir "$OUTPUT_DIR"
