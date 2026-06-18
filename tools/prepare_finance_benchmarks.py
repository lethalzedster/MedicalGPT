"""Prepare named FinAlign benchmark jsonl files.

This script consumes the raw Hugging Face snapshots downloaded by
tools/download_finance_datasets.py and writes compact, explicit benchmark
files for later Base/SFT/DPO evaluation.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


DEFAULT_RAW_DIR = r"F:\FinAlign\datasets\raw"
DEFAULT_OUTPUT_DIR = "./data/finance_benchmarks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FinAlign benchmark datasets.")
    parser.add_argument("--raw_dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_fineval", type=int, default=2000)
    parser.add_argument("--max_finqa", type=int, default=2000)
    parser.add_argument("--max_convfinqa", type=int, default=2000)
    parser.add_argument("--max_sentiment", type=int, default=5000)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return re.sub(r"\s+", " ", str(value)).strip()


def iter_parquet_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.parquet")):
        if ".cache" not in path.parts:
            yield path


def read_parquet(path: Path) -> Iterator[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files=str(path), split="train")
    for row in ds:
        yield dict(row)


def read_fineval_zip(path: Path) -> Iterator[Dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".csv"):
                continue
            split = name.split("/", 1)[0]
            subject = Path(name).stem
            with zf.open(name) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                reader = csv.DictReader(text)
                for row in reader:
                    item = dict(row)
                    item["_split"] = split
                    item["_subject"] = subject
                    yield item


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def limit_rows(rows: Iterable[Dict[str, Any]], limit: int) -> Iterator[Dict[str, Any]]:
    for i, row in enumerate(rows):
        if limit > 0 and i >= limit:
            break
        yield row


def fineval_rows(raw_dir: Path) -> Iterator[Dict[str, Any]]:
    root = raw_dir / "SUFE-AIFLM-Lab__FinEval"
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    parquet_files = list(iter_parquet_files(root))
    iterators: List[Iterator[Dict[str, Any]]] = [read_parquet(path) for path in parquet_files]
    zip_path = root / "FinEval.zip"
    if zip_path.exists():
        iterators.append(read_fineval_zip(zip_path))
    for iterator in iterators:
        for row in iterator:
            split = clean_text(row.get("_split")) or "unknown"
            subject = clean_text(row.get("_subject"))
            question = clean_text(row.get("question") or row.get("query") or row.get("input"))
            options: List[str] = []
            raw_options = row.get("options") or row.get("choices")
            if isinstance(raw_options, list):
                options = [clean_text(x) for x in raw_options]
            elif isinstance(raw_options, dict):
                options = [f"{k}. {clean_text(v)}" for k, v in sorted(raw_options.items())]
            else:
                for key in ["A", "B", "C", "D", "E"]:
                    text = clean_text(row.get(key))
                    if text:
                        options.append(f"{key}. {text}")
            answer = clean_text(row.get("answer") or row.get("label") or row.get("gold"))
            if answer.isdigit() and options:
                idx = int(answer)
                if 0 <= idx < len(options):
                    answer = labels[idx]
            if question and options and answer:
                prompt = question + "\n" + "\n".join(options) + "\n请只输出正确选项的字母。"
                yield {
                    "id": clean_text(row.get("id")) or f"fineval-{abs(hash(prompt))}",
                    "task": "fineval",
                    "split": split,
                    "subject": subject,
                    "prompt": prompt,
                    "answer": answer[:1].upper(),
                    "choices": options,
                    "metric": "accuracy",
                    "source": "SUFE-AIFLM-Lab/FinEval",
                }


def qa_rows(raw_dir: Path, repo_dir: str, task: str, source: str) -> Iterator[Dict[str, Any]]:
    root = raw_dir / repo_dir
    for path in iter_parquet_files(root):
        split = path.parent.name if path.parent.name != root.name else path.stem
        for row in read_parquet(path):
            prompt = clean_text(row.get("query") or row.get("question") or row.get("input"))
            answer = clean_text(row.get("answer") or row.get("output"))
            if prompt and answer:
                yield {
                    "id": clean_text(row.get("id")) or f"{task}-{abs(hash(prompt))}",
                    "task": task,
                    "split": split,
                    "prompt": prompt,
                    "answer": answer,
                    "metric": "numeric_em",
                    "source": source,
                }


def sentiment_rows(raw_dir: Path) -> Iterator[Dict[str, Any]]:
    roots = [
        (raw_dir / "FinGPT__fingpt-sentiment-train", "FinGPT/fingpt-sentiment-train"),
        (raw_dir / "ChanceFocus__flare-headlines", "ChanceFocus/flare-headlines"),
    ]
    allowed = {"negative", "neutral", "positive"}
    for root, source in roots:
        for path in iter_parquet_files(root):
            split = path.parent.name if path.parent.name != root.name else path.stem
            for row in read_parquet(path):
                instruction = clean_text(row.get("instruction"))
                text = clean_text(row.get("input") or row.get("query") or row.get("text"))
                answer = clean_text(row.get("output") or row.get("answer") or row.get("label"))
                answer_l = answer.lower()
                if answer_l not in allowed:
                    continue
                prompt = instruction + "\n" + text if instruction else text
                if prompt:
                    yield {
                        "id": clean_text(row.get("id")) or f"sentiment-{abs(hash(prompt))}",
                        "task": "sentiment",
                        "split": split,
                        "prompt": prompt + "\n请只输出 negative、neutral 或 positive。",
                        "answer": answer_l,
                        "choices": ["negative", "neutral", "positive"],
                        "metric": "macro_f1",
                        "source": source,
                    }


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("fineval", fineval_rows(raw_dir), args.max_fineval),
        ("finqa", qa_rows(raw_dir, "ChanceFocus__flare-finqa", "finqa", "ChanceFocus/flare-finqa"), args.max_finqa),
        (
            "convfinqa",
            qa_rows(raw_dir, "ChanceFocus__flare-convfinqa", "convfinqa", "ChanceFocus/flare-convfinqa"),
            args.max_convfinqa,
        ),
        ("sentiment", sentiment_rows(raw_dir), args.max_sentiment),
    ]

    manifest: Dict[str, Any] = {"raw_dir": str(raw_dir), "output_dir": str(output_dir), "benchmarks": {}}
    for name, iterator, limit in specs:
        rows = list(limit_rows(iterator, limit))
        count = write_jsonl(output_dir / name / "eval.jsonl", rows)
        manifest["benchmarks"][name] = {
            "count": count,
            "sources": dict(Counter(row["source"] for row in rows)),
            "metrics": sorted(set(row["metric"] for row in rows)),
        }
        print(f"{name}: {count}")

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
