# FinAlign: Chinese Financial LLM Post-training

FinAlign is a finance-domain adaptation of the MedicalGPT training pipeline. It keeps the upstream training code for SFT, DPO, RLOO/PPO-style alignment, and evaluation, while adding finance datasets and preprocessing scripts.

The project target is a reproducible `data -> training -> evaluation` loop for a Chinese financial assistant, not a full pretraining project.

## Scope

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Hardware target: `8 x RTX 4090`
- Main training path: QLoRA SFT, then DPO/RLOO preference alignment
- Domain tasks: financial QA, annual-report reasoning, financial sentiment, numeric calculation, and risk-compliance responses
- Excluded by default: large continued-pretraining corpora such as `BAAI/IndustryCorpus_finance`

## Raw Data

The downloader saves raw Hugging Face snapshots to `F:\FinAlign\datasets\raw` by default.

```powershell
.\scripts\download_finance_datasets.ps1 -OutputDir F:\FinAlign\datasets\raw
```

Default datasets:

- `BAAI/IndustryInstruction_Finance-Economics`
- `oieieio/Finance-Instruct-500k`
- `FinGPT/fingpt-sentiment-train`
- `ChanceFocus/flare-finqa`
- `ChanceFocus/flare-convfinqa`
- `ChanceFocus/flare-headlines`
- `SUFE-AIFLM-Lab/FinEval`
- `TongjiFinLab/CFBenchmark`
- `gandhiraketla277/finance-dpo-dataset`

## Data Preparation

Convert raw datasets into MedicalGPT-compatible local files:

```powershell
.\scripts\prepare_finance_data.ps1 `
  -RawDir F:\FinAlign\datasets\raw `
  -OutputDir .\data `
  -MaxSftSamples 300000 `
  -MaxDpoSamples 50000
```

For a quick smoke test:

```powershell
python tools/prepare_finance_data.py `
  --raw_dir F:\FinAlign\datasets\raw `
  --output_dir .\data\finance_smoke `
  --max_sft_samples 2000 `
  --max_dpo_samples 200 `
  --max_eval_samples 500 `
  --scan_limit_per_source 3000
```

Generated files:

- `data/finance_sft/train/train.jsonl`
- `data/finance_sft/validation/validation.jsonl`
- `data/finance_sft/test/test.jsonl`
- `data/finance_reward/train/train.jsonl`
- `data/finance_reward/validation/validation.jsonl`
- `data/finance_eval/finance_eval.jsonl`
- `data/finance_manifest.json`

SFT format:

```json
{"conversations":[{"from":"system","value":"你是一个专业、谨慎的金融助手。回答仅供学习和研究参考，不构成投资建议。"},{"from":"human","value":"请解释久期的含义。"},{"from":"gpt","value":"久期用于衡量债券价格对利率变化的敏感性..."}]}
```

DPO format:

```json
{"conversations":[{"from":"human","value":"某公司营收从100亿增长到125亿，同比增长率是多少？"}],"chosen":"同比增长率 = (125 - 100) / 100 = 25%。","rejected":"同比增长率大约是20%。"}
```

## Training

SFT:

```bash
bash scripts/run_finance_sft.sh
```

DPO:

```bash
bash scripts/run_finance_dpo.sh
```

The default scripts use QLoRA, bf16, gradient checkpointing, and all linear LoRA target modules. Adjust batch size, sequence length, and gradient accumulation after checking actual GPU memory.

## Resume-facing Evidence

Track these numbers after running experiments:

- SFT samples, DPO pairs, and evaluation samples
- Base vs SFT vs SFT+DPO win rate
- FinEval/CFinBench accuracy
- FinQA/ConvFinQA numeric accuracy
- financial sentiment F1
- safety/risk-compliance pass rate
- peak GPU memory and training time
