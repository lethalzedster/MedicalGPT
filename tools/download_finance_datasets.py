"""Download FinAlign finance datasets from Hugging Face.

The script downloads raw datasets only. It intentionally skips large
pretraining corpora such as BAAI/IndustryCorpus_finance.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_DATASETS = [
    "BAAI/IndustryInstruction_Finance-Economics",
    "oieieio/Finance-Instruct-500k",
    "FinGPT/fingpt-sentiment-train",
    "ChanceFocus/flare-finqa",
    "ChanceFocus/flare-convfinqa",
    "ChanceFocus/flare-headlines",
    "SUFE-AIFLM-Lab/FinEval",
    "TongjiFinLab/CFBenchmark",
    "gandhiraketla277/finance-dpo-dataset",
]


def repo_dir_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download FinAlign raw finance datasets.")
    parser.add_argument(
        "--output_dir",
        default=r"F:\FinAlign\datasets\raw",
        help="Directory for raw dataset snapshots.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset repo id to download. Can be passed multiple times. Defaults to the FinAlign set.",
    )
    parser.add_argument(
        "--log_file",
        default=None,
        help="Download log file. Defaults to <output_dir>/../download.log.",
    )
    parser.add_argument(
        "--ignore_patterns",
        nargs="*",
        default=["*.md5", "*.lock"],
        help="Patterns ignored by snapshot_download.",
    )
    return parser.parse_args()


def append_log(log_file: Path, line: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file) if args.log_file else output_dir.parent / "download.log"
    datasets = args.datasets or DEFAULT_DATASETS

    append_log(log_file, f"=== FinAlign dataset download start {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    append_log(log_file, f"output_dir={output_dir}")

    for repo_id in datasets:
        local_dir = output_dir / repo_dir_name(repo_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        append_log(log_file, f"START {repo_id}")
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                ignore_patterns=args.ignore_patterns,
            )
        except Exception as exc:  # keep downloading the rest
            append_log(log_file, f"FAIL  {repo_id}: {exc!r}")
        else:
            append_log(log_file, f"DONE  {repo_id}")

    append_log(log_file, f"=== FinAlign dataset download end {time.strftime('%Y-%m-%d %H:%M:%S')} ===")


if __name__ == "__main__":
    main()
