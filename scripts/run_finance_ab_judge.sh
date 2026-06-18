#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p reports/finalign

set +u
source /share/home/beiyou2/miniconda3/etc/profile.d/conda.sh
conda activate rs_grpo
set -u
export PYTHONNOUSERSITE=1

echo "Using python: $(command -v python)"

LEFT_FILE="${LEFT_FILE:-reports/finalign/sft/finqa_pred.jsonl}"
RIGHT_FILE="${RIGHT_FILE:-reports/finalign/dpo/finqa_pred.jsonl}"
PAIRS_FILE="${PAIRS_FILE:-reports/finalign/ab_pairs_500.jsonl}"
JUDGE_FILE="${JUDGE_FILE:-reports/finalign/llm_judge_labels.jsonl}"
RESULT_FILE="${RESULT_FILE:-reports/finalign/llm_judge_results.json}"

python tools/build_finalign_ab_pairs.py \
  --left_file "$LEFT_FILE" \
  --right_file "$RIGHT_FILE" \
  --left_name sft \
  --right_name dpo \
  --output_file "$PAIRS_FILE" \
  --max_samples 500

if [[ -f "$JUDGE_FILE" ]]; then
  python tools/finalign_llm_judge.py \
    --judge_file "$JUDGE_FILE" \
    --target_model dpo \
    --baseline_model sft \
    --output_file "$RESULT_FILE"
else
  echo "A/B pairs written to $PAIRS_FILE"
  echo "Fill $JUDGE_FILE with judge labels containing: a_model, b_model, winner."
fi
