#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

set +u
source /share/home/beiyou2/miniconda3/etc/profile.d/conda.sh
conda activate rs_grpo
set -u
export PYTHONNOUSERSITE=1

PROMPT_FILE="${PROMPT_FILE:-reports/finalign/dpo_data/prompts_12k.jsonl}"
BASE_CANDIDATE_FILE="${BASE_CANDIDATE_FILE:-reports/finalign/dpo_data/base_candidates.jsonl}"
SFT_CANDIDATE_FILE="${SFT_CANDIDATE_FILE:-reports/finalign/dpo_data/sft_candidates.jsonl}"
SFT_SAMPLE_CANDIDATE_FILE="${SFT_SAMPLE_CANDIDATE_FILE:-reports/finalign/dpo_data/sft_sample_candidates.jsonl}"
JUDGE_FILE="${JUDGE_FILE:-reports/finalign/dpo_data/deepseek_judge.jsonl}"
DPO_OUTPUT_DIR="${DPO_OUTPUT_DIR:-data/finance_reward_deepseek}"
SUMMARY_FILE="${SUMMARY_FILE:-reports/finalign/data/finalign_dpo_deepseek_summary.json}"

MAX_PROMPTS="${MAX_PROMPTS:-12000}"
MAX_JUDGE_SAMPLES="${MAX_JUDGE_SAMPLES:-10000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-flash}"
DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"

mkdir -p reports/finalign/dpo_data reports/finalign/data

python tools/sample_finalign_prompts.py \
  --input_file data/finance_sft/train/train.jsonl \
  --input_file data/finance_sft/validation/validation.jsonl \
  --exclude_file data/finance_sft/test/test.jsonl \
  --output_file "$PROMPT_FILE" \
  --max_samples "$MAX_PROMPTS"

python tools/finalign_generate.py \
  --model_name_or_path /share/bupt/models/Qwen2.5-7B-Instruct \
  --input_file "$PROMPT_FILE" \
  --output_file "$BASE_CANDIDATE_FILE" \
  --max_samples "$MAX_PROMPTS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --batch_size "$GEN_BATCH_SIZE" \
  --resume \
  --load_in_4bit

python tools/finalign_generate.py \
  --model_name_or_path outputs-finalign-sft-qwen25-7b-merged \
  --input_file "$PROMPT_FILE" \
  --output_file "$SFT_CANDIDATE_FILE" \
  --max_samples "$MAX_PROMPTS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --batch_size "$GEN_BATCH_SIZE" \
  --resume \
  --load_in_4bit

python tools/finalign_generate.py \
  --model_name_or_path outputs-finalign-sft-qwen25-7b-merged \
  --input_file "$PROMPT_FILE" \
  --output_file "$SFT_SAMPLE_CANDIDATE_FILE" \
  --max_samples "$MAX_PROMPTS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 0.7 \
  --batch_size "$GEN_BATCH_SIZE" \
  --resume \
  --load_in_4bit

python tools/deepseek_pairwise_judge.py \
  --candidate_file "base=$BASE_CANDIDATE_FILE" \
  --candidate_file "sft=$SFT_CANDIDATE_FILE" \
  --candidate_file "sft_sample=$SFT_SAMPLE_CANDIDATE_FILE" \
  --output_file "$JUDGE_FILE" \
  --model "$DEEPSEEK_MODEL" \
  --base_url "$DEEPSEEK_BASE_URL" \
  --max_samples "$MAX_JUDGE_SAMPLES"

python tools/build_finalign_dpo_from_judge.py \
  --judge_file "$JUDGE_FILE" \
  --output_dir "$DPO_OUTPUT_DIR" \
  --summary_file "$SUMMARY_FILE" \
  --max_samples "$MAX_JUDGE_SAMPLES"
