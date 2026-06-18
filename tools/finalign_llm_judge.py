"""Aggregate FinAlign LLM-as-Judge pairwise labels.

Input rows should contain at least:
  a_model, b_model, winner

winner can be "A", "B", "tie", or the model name.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate FinAlign LLM-as-Judge labels.")
    parser.add_argument("--judge_file", required=True)
    parser.add_argument("--target_model", default="dpo")
    parser.add_argument("--baseline_model", default="sft")
    parser.add_argument("--output_file", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def winner_model(row: Dict[str, Any]) -> str:
    winner = str(row.get("winner", "")).strip().lower()
    a_model = str(row.get("a_model", "")).strip()
    b_model = str(row.get("b_model", "")).strip()
    if winner in {"a", "response_a", "回答a"}:
        return a_model
    if winner in {"b", "response_b", "回答b"}:
        return b_model
    if winner in {"tie", "draw", "平局", "both", "none"}:
        return "tie"
    if winner == a_model.lower():
        return a_model
    if winner == b_model.lower():
        return b_model
    return "unknown"


def main() -> None:
    args = parse_args()
    rows = list(read_jsonl(Path(args.judge_file)))
    wins = Counter()
    by_task: Dict[str, Counter] = defaultdict(Counter)
    bad_cases = Counter()
    for row in rows:
        winner = winner_model(row)
        wins[winner] += 1
        by_task[str(row.get("task", "unknown"))][winner] += 1
        for item in row.get("bad_case_tags", []) or []:
            bad_cases[str(item)] += 1

    target_wins = wins[args.target_model]
    baseline_wins = wins[args.baseline_model]
    comparable = target_wins + baseline_wins
    result = {
        "total": len(rows),
        "target_model": args.target_model,
        "baseline_model": args.baseline_model,
        "wins": dict(wins),
        "pairwise_win_rate_excluding_ties": target_wins / comparable if comparable else 0.0,
        "pairwise_win_rate_counting_ties_half": (target_wins + 0.5 * wins["tie"]) / len(rows) if rows else 0.0,
        "by_task": {task: dict(counter) for task, counter in by_task.items()},
        "bad_case_tags": dict(bad_cases),
    }

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
