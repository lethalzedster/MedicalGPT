param(
    [string]$OutputDir = "F:\FinAlign\datasets\raw"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

python tools/download_finance_datasets.py --output_dir $OutputDir
