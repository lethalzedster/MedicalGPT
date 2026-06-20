"""Sample FinAlign prompts for preference-data generation.

The output is intentionally simple so it can be reused by local generation
scripts and external LLM-as-Judge jobs:
  {"id": "...", "prompt": "...", "reference": "...", "source_file": "..."}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample FinAlign prompts from SFT data.")
    parser.add_argument("--input_file", action="append", required=True)
    parser.add_argument("--exclude_file", action="append", default=[])
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_samples", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_prompt_chars", type=int, default=8)
    parser.add_argument("--max_prompt_chars", type=int, default=4000)
    parser.add_argument("--min_reference_chars", type=int, default=4)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\u0000", " ")
    return re.sub(r"\s+", " ", text).strip()


def prompt_key(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def extract_from_conversations(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
    messages = row.get("conversations")
    if not isinstance(messages, list):
        return None

    last_human = ""
    first_answer_after_last_human = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = clean_text(message.get("from") or message.get("role")).lower()
        value = clean_text(message.get("value") or message.get("content"))
        if role in {"human", "user"} and value:
            last_human = value
            first_answer_after_last_human = ""
        elif role in {"gpt", "assistant"} and last_human and not first_answer_after_last_human:
            first_answer_after_last_human = value

    if not last_human:
        return None
    return {"prompt": last_human, "reference": first_answer_after_last_human}


def extract_prompt(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
    from_conv = extract_from_conversations(row)
    if from_conv:
        return from_conv
    prompt = clean_text(row.get("prompt") or row.get("question") or row.get("query") or row.get("instruction"))
    reference = clean_text(row.get("answer") or row.get("output") or row.get("response") or row.get("label"))
    if not prompt:
        return None
    return {"prompt": prompt, "reference": reference}


def collect_excluded(paths: Iterable[str]) -> Set[str]:
    excluded: Set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        for row in read_jsonl(path):
            item = extract_prompt(row)
            if item:
                excluded.add(prompt_key(item["prompt"]))
    return excluded


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    excluded = collect_excluded(args.exclude_file)

    items: List[Dict[str, str]] = []
    seen: Set[str] = set(excluded)
    for raw_path in args.input_file:
        path = Path(raw_path)
        for row in read_jsonl(path):
            item = extract_prompt(row)
            if not item:
                continue
            prompt = item["prompt"]
            reference = item.get("reference", "")
            if not (args.min_prompt_chars <= len(prompt) <= args.max_prompt_chars):
                continue
            if reference and len(reference) < args.min_reference_chars:
                continue
            key = prompt_key(prompt)
            if key in seen:
                continue
            seen.add(key)
            items.append(
                {
                    "id": f"prompt-{len(items):08d}",
                    "prompt": prompt,
                    "reference": reference,
                    "source_file": str(path),
                }
            )

    rng.shuffle(items)
    if args.max_samples > 0:
        items = items[: args.max_samples]

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for index, item in enumerate(items):
            item = dict(item)
            item["id"] = f"prompt-{index:08d}"
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} prompts to {output_file}")


if __name__ == "__main__":
    main()
