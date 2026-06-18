"""Build A/B pairs from two FinAlign generation files."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FinAlign A/B judge pairs.")
    parser.add_argument("--left_file", required=True, help="Usually SFT predictions.")
    parser.add_argument("--right_file", required=True, help="Usually DPO predictions.")
    parser.add_argument("--left_name", default="sft")
    parser.add_argument("--right_name", default="dpo")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    args = parse_args()
    left_rows = {row.get("id"): row for row in read_jsonl(Path(args.left_file)) if row.get("id")}
    right_rows = {row.get("id"): row for row in read_jsonl(Path(args.right_file)) if row.get("id")}
    ids = sorted(set(left_rows) & set(right_rows))
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    if args.max_samples > 0:
        ids = ids[: args.max_samples]

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for item_id in ids:
            left = left_rows[item_id]
            right = right_rows[item_id]
            swap = rng.random() < 0.5
            a_model, a_row = (args.right_name, right) if swap else (args.left_name, left)
            b_model, b_row = (args.left_name, left) if swap else (args.right_name, right)
            out = {
                "id": item_id,
                "task": left.get("task"),
                "prompt": left.get("prompt"),
                "answer": left.get("answer"),
                "a_model": a_model,
                "b_model": b_model,
                "response_a": a_row.get("prediction", ""),
                "response_b": b_row.get("prediction", ""),
                "judge_dimensions": [
                    "professional_accuracy",
                    "numeric_consistency",
                    "faithfulness",
                    "risk_disclosure",
                    "compliance",
                ],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"wrote {len(ids)} pairs to {output_file}")


if __name__ == "__main__":
    main()
