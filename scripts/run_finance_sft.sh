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

echo "Using python: $(command -v python)" | tee -a logs/fin_sft.log
echo "Using torchrun: $(command -v torchrun)" | tee -a logs/fin_sft.log
echo "Using report_to: tensorboard" | tee -a logs/fin_sft.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 training/supervised_finetuning.py \
    --model_name_or_path /share/bupt/models/Qwen2.5-7B-Instruct \
    --train_file_dir ./data/finance_sft/train \
    --validation_file_dir ./data/finance_sft/validation \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_peft True \
    --qlora True \
    --load_in_4bit True \
    --max_train_samples -1 \
    --max_eval_samples 500 \
    --model_max_length 2048 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_steps 100 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 200 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 5 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 8 \
    --output_dir outputs-finalign-sft-qwen25-7b \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --cache_dir /share/home/beiyou2/.cache/huggingface \
    --flash_attn True \
    --deepspeed scripts/zero2.json 2>&1 | tee -a logs/fin_sft.log
