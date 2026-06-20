#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

set +u
source /share/home/beiyou2/miniconda3/etc/profile.d/conda.sh
conda activate rs_grpo
set -u
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/share/home/beiyou2/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/share/home/beiyou2/.cache/huggingface/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/share/home/beiyou2/.cache/huggingface/transformers}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

echo "Using python: $(command -v python)" | tee -a logs/fin_dpo.log
echo "Using report_to: tensorboard" | tee -a logs/fin_dpo.log

MODEL_PATH="${MODEL_PATH:-outputs-finalign-sft-qwen25-7b-merged}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs-finalign-dpo-qwen25-7b}"
MAX_STEPS="${MAX_STEPS:-1000}"
REWARD_DATA_DIR="${REWARD_DATA_DIR:-./data/finance_reward}"
TRAIN_FILE_DIR="${TRAIN_FILE_DIR:-${REWARD_DATA_DIR}/train}"
VALIDATION_FILE_DIR="${VALIDATION_FILE_DIR:-${REWARD_DATA_DIR}/validation}"

echo "Using reward data: $REWARD_DATA_DIR" | tee -a logs/fin_dpo.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 training/dpo_training.py \
    --model_name_or_path "$MODEL_PATH" \
    --train_file_dir "$TRAIN_FILE_DIR" \
    --validation_file_dir "$VALIDATION_FILE_DIR" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_peft True \
    --qlora True \
    --load_in_4bit True \
    --max_train_samples -1 \
    --max_eval_samples 500 \
    --max_steps "$MAX_STEPS" \
    --eval_steps 100 \
    --save_steps 250 \
    --max_source_length 1536 \
    --max_target_length 512 \
    --output_dir "$OUTPUT_DIR" \
    --target_modules all \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 True \
    --fp16 False \
    --report_to tensorboard \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --cache_dir /share/home/beiyou2/.cache/huggingface 2>&1 | tee -a logs/fin_dpo.log
