# FinAlign Evaluation And Alignment Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the scripts and data preparation needed after QLoRA-SFT: merge the SFT adapter, run smoke inference, prepare named finance benchmarks, evaluate Base/SFT/DPO, run DPO, and build LLM-as-Judge A/B reports.

**Architecture:** Keep raw finance data under `F:\FinAlign\datasets` locally and synchronized benchmark jsonl files under `/share/bupt/hgy/MedicalGPT/data/finance_benchmarks` remotely. Add focused Python tools for benchmark preparation, generation/evaluation, A/B pair building, and judge aggregation, with shell scripts as reproducible experiment entrypoints.

**Tech Stack:** Python, HuggingFace Transformers, PEFT, datasets, Qwen2.5-7B-Instruct, QLoRA, DeepSpeed ZeRO-2, TensorBoard.

---

### Task 1: Prepare Named Finance Benchmarks

**Files:**
- Create: `tools/prepare_finance_benchmarks.py`
- Create outputs: `data/finance_benchmarks/{fineval,finqa,convfinqa,sentiment}/eval.jsonl`

- [ ] Implement a scanner over the downloaded raw snapshots in `F:\FinAlign\datasets\raw`.
- [ ] Extract FinEval multiple-choice examples with labels.
- [ ] Extract FinQA and ConvFinQA numeric QA examples with answers.
- [ ] Extract financial sentiment examples with labels.
- [ ] Write a manifest with counts and source names.

### Task 2: Add SFT Merge And Smoke Inference Scripts

**Files:**
- Create: `scripts/merge_finance_sft_adapter.sh`
- Create: `scripts/run_finance_smoke_infer.sh`
- Create: `tools/finalign_generate.py`

- [ ] Merge `outputs-finalign-sft-qwen25-7b` into `outputs-finalign-sft-qwen25-7b-merged` using `tools/merge_peft_adapter.py`.
- [ ] Generate Base and SFT responses on a small benchmark subset.
- [ ] Save samples to `reports/finalign/smoke_sft_samples.jsonl`.

### Task 3: Add Automatic Benchmark Evaluation

**Files:**
- Create: `tools/finalign_eval.py`
- Create: `scripts/run_finance_eval_base.sh`
- Create: `scripts/run_finance_eval_sft.sh`
- Create: `scripts/run_finance_eval_dpo.sh`

- [ ] Run generation for FinEval, FinQA, ConvFinQA, and sentiment datasets.
- [ ] Score choice accuracy, numeric exact match, and sentiment accuracy/macro-F1.
- [ ] Save per-model JSON metrics to `reports/finalign`.

### Task 4: Fix DPO Continuation Entry Points

**Files:**
- Modify: `scripts/run_finance_dpo.sh`
- Create: `scripts/run_finance_dpo_smoke.sh`

- [ ] Use merged SFT model path as DPO initial model.
- [ ] Keep the existing tensorboard/logging/cache conventions.
- [ ] Provide a 100-step smoke run before formal DPO.

### Task 5: Add LLM-as-Judge A/B Flow

**Files:**
- Create: `tools/build_finalign_ab_pairs.py`
- Create: `tools/finalign_llm_judge.py`
- Create: `scripts/run_finance_ab_judge.sh`

- [ ] Build 500 A/B samples from Base, SFT, and DPO generation outputs.
- [ ] Support judge scoring from either an API-compatible endpoint or a local JSONL of labels.
- [ ] Aggregate pairwise win rate and bad-case categories.

### Task 6: Sync And Verify On gpu4090

**Files:**
- No new source files.

- [ ] Copy scripts/tools/data to `/share/bupt/hgy/MedicalGPT`.
- [ ] Run benchmark prep in dry/small mode.
- [ ] Run script syntax checks.
- [ ] Print next commands for merge, smoke inference, Base/SFT eval, DPO smoke, formal DPO, and A/B judge.
