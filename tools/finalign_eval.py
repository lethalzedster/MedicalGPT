"""Score FinAlign generation files."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


SENTIMENT_LABELS = ["negative", "neutral", "positive"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score FinAlign benchmark predictions.")
    parser.add_argument("--prediction_file", required=True)
    parser.add_argument("--output_file", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def first_choice(text: str) -> str:
    match = re.search(r"\b([A-E])\b", text.upper())
    return match.group(1) if match else text.strip()[:1].upper()


def extract_number(text: str) -> Optional[float]:
    cleaned = text.replace(",", "")
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def numeric_equal(pred: str, gold: str) -> bool:
    pred_num = extract_number(pred)
    gold_num = extract_number(gold)
    if pred_num is None or gold_num is None:
        return pred.strip().lower() == gold.strip().lower()
    if gold_num == 0:
        return abs(pred_num) < 1e-4
    return abs(pred_num - gold_num) / max(abs(gold_num), 1e-8) <= 1e-4


def normalize_sentiment(text: str) -> str:
    text_l = text.lower()
    for label in SENTIMENT_LABELS:
        if re.search(rf"\b{label}\b", text_l):
            return label
    if "负" in text or "消极" in text:
        return "negative"
    if "正" in text or "积极" in text:
        return "positive"
    if "中" in text:
        return "neutral"
    return text_l.strip()


def macro_f1(pairs: Iterable[tuple[str, str]]) -> float:
    pairs = list(pairs)
    scores = []
    for label in SENTIMENT_LABELS:
        tp = sum(1 for gold, pred in pairs if gold == label and pred == label)
        fp = sum(1 for gold, pred in pairs if gold != label and pred == label)
        fn = sum(1 for gold, pred in pairs if gold == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def score_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row.get("task", "unknown")].append(row)

    metrics: Dict[str, Any] = {"total": len(rows), "tasks": {}}
    for task, task_rows in sorted(by_task.items()):
        if task == "fineval":
            correct = [
                first_choice(row.get("prediction", "")) == str(row.get("answer", "")).strip().upper()[:1]
                for row in task_rows
            ]
            metrics["tasks"][task] = {"count": len(task_rows), "accuracy": sum(correct) / len(correct)}
        elif task in {"finqa", "convfinqa"}:
            correct = [numeric_equal(row.get("prediction", ""), str(row.get("answer", ""))) for row in task_rows]
            metrics["tasks"][task] = {"count": len(task_rows), "numeric_em": sum(correct) / len(correct)}
        elif task == "sentiment":
            pairs = [
                (str(row.get("answer", "")).lower().strip(), normalize_sentiment(row.get("prediction", "")))
                for row in task_rows
            ]
            correct = [gold == pred for gold, pred in pairs]
            metrics["tasks"][task] = {
                "count": len(task_rows),
                "accuracy": sum(correct) / len(correct),
                "macro_f1": macro_f1(pairs),
                "pred_distribution": dict(Counter(pred for _, pred in pairs)),
            }
        else:
            metrics["tasks"][task] = {"count": len(task_rows)}
    return metrics


def main() -> None:
    args = parse_args()
    rows = list(read_jsonl(Path(args.prediction_file)))
    metrics = score_rows(rows)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
