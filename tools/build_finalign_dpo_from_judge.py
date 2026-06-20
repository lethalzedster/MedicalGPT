"""Build MedicalGPT DPO files from DeepSeek judge rows."""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple


SYSTEM_PROMPT = "你是一个专业、谨慎的中文金融助手。回答仅供学习和研究参考，不构成投资建议。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert judge labels into FinAlign DPO data.")
    parser.add_argument("--judge_file", required=True)
    parser.add_argument("--output_dir", default="data/finance_reward_deepseek")
    parser.add_argument("--summary_file", default="reports/finalign/data/finalign_dpo_deepseek_summary.json")
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--validation_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_confidence", type=float, default=0.7)
    parser.add_argument("--min_score_gap", type=float, default=1.0)
    parser.add_argument("--max_length_ratio", type=float, default=3.0)
    parser.add_argument("--max_similarity", type=float, default=0.92)
    parser.add_argument("--min_answer_chars", type=int, default=8)
    parser.add_argument("--max_answer_chars", type=int, default=6000)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def valid_pair(row: Dict[str, Any], args: argparse.Namespace, reject_reasons: Counter[str]) -> bool:
    prompt = clean_text(row.get("prompt"))
    chosen = clean_text(row.get("chosen"))
    rejected = clean_text(row.get("rejected"))
    if not prompt or not chosen or not rejected:
        reject_reasons["missing_text"] += 1
        return False
    if chosen == rejected:
        reject_reasons["same_answer"] += 1
        return False
    if as_float(row.get("confidence")) < args.min_confidence:
        reject_reasons["low_confidence"] += 1
        return False
    if as_float(row.get("score_gap")) < args.min_score_gap:
        reject_reasons["low_score_gap"] += 1
        return False
    if not (args.min_answer_chars <= len(chosen) <= args.max_answer_chars):
        reject_reasons["chosen_length"] += 1
        return False
    if not (args.min_answer_chars <= len(rejected) <= args.max_answer_chars):
        reject_reasons["rejected_length"] += 1
        return False
    length_ratio = max(len(chosen), len(rejected)) / max(min(len(chosen), len(rejected)), 1)
    if length_ratio > args.max_length_ratio:
        reject_reasons["length_ratio"] += 1
        return False
    similarity = difflib.SequenceMatcher(None, chosen, rejected).ratio()
    if similarity > args.max_similarity:
        reject_reasons["too_similar"] += 1
        return False
    return True


def to_dpo(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": clean_text(row.get("prompt"))},
        ],
        "chosen": clean_text(row.get("chosen")),
        "rejected": clean_text(row.get("rejected")),
        "meta": {
            "id": row.get("id"),
            "winner_model": row.get("winner_model"),
            "loser_model": row.get("loser_model"),
            "confidence": row.get("confidence"),
            "score_gap": row.get("score_gap"),
            "bad_case_tags": row.get("bad_case_tags", []),
            "judge_model": row.get("judge_model"),
        },
    }


def split_rows(rows: List[Dict[str, Any]], validation_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    val_n = int(len(rows) * validation_ratio)
    return rows[val_n:], rows[:val_n]


def main() -> None:
    args = parse_args()
    reject_reasons: Counter[str] = Counter()
    accepted: List[Dict[str, Any]] = []
    seen_prompts = set()
    model_pairs: Counter[str] = Counter()

    for row in read_jsonl(Path(args.judge_file)):
        prompt_key = clean_text(row.get("prompt")).lower()
        if prompt_key in seen_prompts:
            reject_reasons["duplicate_prompt"] += 1
            continue
        if not valid_pair(row, args, reject_reasons):
            continue
        seen_prompts.add(prompt_key)
        model_pairs[f"{row.get('winner_model')} > {row.get('loser_model')}"] += 1
        accepted.append(to_dpo(row))
        if args.max_samples > 0 and len(accepted) >= args.max_samples:
            break

    train, validation = split_rows(accepted, args.validation_ratio, args.seed)
    output_dir = Path(args.output_dir)
    counts = {
        "train": write_jsonl(output_dir / "train" / "train.jsonl", train),
        "validation": write_jsonl(output_dir / "validation" / "validation.jsonl", validation),
    }
    summary = {
        "judge_file": args.judge_file,
        "output_dir": str(output_dir),
        "counts": counts,
        "accepted": len(accepted),
        "rejected": dict(reject_reasons),
        "model_pairs": dict(model_pairs),
        "filters": {
            "min_confidence": args.min_confidence,
            "min_score_gap": args.min_score_gap,
            "max_length_ratio": args.max_length_ratio,
            "max_similarity": args.max_similarity,
            "max_samples": args.max_samples,
        },
    }
    summary_file = Path(args.summary_file)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
