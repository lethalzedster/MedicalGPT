#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MAX_STEPS="${MAX_STEPS:-100}" OUTPUT_DIR="${OUTPUT_DIR:-outputs-finalign-dpo-qwen25-7b-smoke}" bash scripts/run_finance_dpo.sh
