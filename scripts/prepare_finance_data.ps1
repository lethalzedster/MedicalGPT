param(
    [string]$RawDir = "F:\FinAlign\datasets\raw",
    [string]$OutputDir = ".\data",
    [int]$MaxSftSamples = 300000,
    [int]$MaxDpoSamples = 50000
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

python tools/prepare_finance_data.py `
    --raw_dir $RawDir `
    --output_dir $OutputDir `
    --max_sft_samples $MaxSftSamples `
    --max_dpo_samples $MaxDpoSamples
