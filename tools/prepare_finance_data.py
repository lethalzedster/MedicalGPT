"""Prepare FinAlign finance datasets for SFT and DPO training.

Input:
  Raw Hugging Face dataset snapshots, typically downloaded by
  tools/download_finance_datasets.py.

Output:
  data/finance_sft/train/train.jsonl
  data/finance_sft/validation/validation.jsonl
  data/finance_reward/train/train.jsonl
  data/finance_reward/validation/validation.jsonl
  data/finance_eval/finance_eval.jsonl

The SFT output uses MedicalGPT's ShareGPT format:
  {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}

The DPO output uses MedicalGPT's preference format:
  {"conversations": [...], "chosen": "...", "rejected": "..."}
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


SYSTEM_PROMPT = "你是一个专业、谨慎的金融助手。回答仅供学习和研究参考，不构成投资建议。"
RAW_EXTENSIONS = {".jsonl", ".json", ".csv", ".parquet"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FinAlign SFT/DPO/eval jsonl files.")
    parser.add_argument("--raw_dir", default=r"F:\FinAlign\datasets\raw", help="Raw dataset snapshot directory.")
    parser.add_argument("--output_dir", default="./data", help="MedicalGPT data output root.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_ratio", type=float, default=0.01)
    parser.add_argument("--test_ratio", type=float, default=0.01)
    parser.add_argument("--max_sft_samples", type=int, default=300000)
    parser.add_argument("--max_dpo_samples", type=int, default=50000)
    parser.add_argument("--max_eval_samples", type=int, default=30000)
    parser.add_argument("--max_per_source", type=int, default=120000)
    parser.add_argument("--min_prompt_chars", type=int, default=4)
    parser.add_argument("--min_answer_chars", type=int, default=4)
    parser.add_argument("--max_prompt_chars", type=int, default=12000)
    parser.add_argument("--max_answer_chars", type=int, default=12000)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def read_json(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
    elif isinstance(obj, dict):
        for key in ("data", "train", "validation", "test", "examples"):
            value = obj.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return
        yield obj


def read_csv(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield dict(row)


def read_with_datasets(path: Path) -> Iterator[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            f"{path.suffix} requires the `datasets` package. Install project requirements first."
        ) from exc

    file_type = "parquet" if path.suffix == ".parquet" else path.suffix.lstrip(".")
    ds = load_dataset(file_type, data_files=str(path), split="train")
    for row in ds:
        yield dict(row)


def iter_records(raw_dir: Path) -> Iterator[Tuple[str, Dict[str, Any]]]:
    files = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in RAW_EXTENSIONS]
    for path in sorted(files):
        source = source_name(path, raw_dir)
        try:
            if path.suffix == ".jsonl":
                iterator = read_jsonl(path)
            elif path.suffix == ".json":
                iterator = read_json(path)
            elif path.suffix == ".csv":
                iterator = read_csv(path)
            else:
                iterator = read_with_datasets(path)
            for record in iterator:
                yield source, record
        except Exception as exc:
            print(f"[warn] skip {path}: {exc}")


def source_name(path: Path, raw_dir: Path) -> str:
    rel = path.relative_to(raw_dir)
    return rel.parts[0] if rel.parts else path.stem


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    text = str(value).replace("\u0000", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_role(role: Any) -> str:
    role = clean_text(role).lower()
    if role in {"human", "user", "question", "input"}:
        return "human"
    if role in {"gpt", "assistant", "answer", "output", "response"}:
        return "gpt"
    if role == "system":
        return "system"
    return role


def normalize_messages(messages: Any) -> List[Dict[str, str]]:
    if not isinstance(messages, list):
        return []
    result = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = normalize_role(message.get("from", message.get("role", "")))
        content = clean_text(message.get("value", message.get("content", "")))
        if role in {"system", "human", "gpt"} and content:
            result.append({"from": role, "value": content})
    return result


def first_nonempty(record: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = clean_text(record.get(key))
        if value:
            return value
    return ""


def stringify_answer(value: Any) -> str:
    if isinstance(value, list):
        messages = normalize_messages(value)
        if messages:
            return "\n".join(m["value"] for m in messages if m["from"] == "gpt") or clean_text(value)
    if isinstance(value, dict):
        for key in ("content", "value", "answer", "output", "response"):
            text = clean_text(value.get(key))
            if text:
                return text
    return clean_text(value)


def build_prompt(record: Dict[str, Any]) -> str:
    instruction = first_nonempty(record, ("instruction", "prompt", "query", "question", "input"))
    context = first_nonempty(record, ("context", "text", "passage", "article"))
    if context and instruction and context not in instruction:
        return f"{context}\n\n问题：{instruction}"
    return instruction or context


def to_sft(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    conversations = normalize_messages(record.get("conversations") or record.get("messages"))
    if conversations:
        roles = {m["from"] for m in conversations}
        if "human" in roles and "gpt" in roles:
            if conversations[0]["from"] != "system":
                conversations = [{"from": "system", "value": SYSTEM_PROMPT}] + conversations
            return {"conversations": conversations}

    prompt = build_prompt(record)
    answer = first_nonempty(record, ("output", "answer", "response", "completion", "label"))
    if not answer and "gold" in record and "choices" in record:
        choices = record.get("choices")
        gold = record.get("gold")
        answer = choice_answer(choices, gold)
    if not prompt or not answer:
        return None
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": answer},
        ]
    }


def choice_answer(choices: Any, gold: Any) -> str:
    if isinstance(choices, list):
        try:
            idx = int(gold)
            if 0 <= idx < len(choices):
                return clean_text(choices[idx])
        except Exception:
            pass
        return clean_text(gold)
    return clean_text(gold)


def to_dpo(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "chosen" not in record or "rejected" not in record:
        return None
    conversations = normalize_messages(record.get("conversations") or record.get("messages"))
    if not conversations:
        prompt = build_prompt(record)
        if not prompt:
            return None
        conversations = [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": prompt},
        ]
    chosen = stringify_answer(record.get("chosen"))
    rejected = stringify_answer(record.get("rejected"))
    if not chosen or not rejected or chosen == rejected:
        return None
    return {"conversations": conversations, "chosen": chosen, "rejected": rejected}


def to_eval(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prompt = build_prompt(record)
    if not prompt:
        return None
    answer = first_nonempty(record, ("answer", "output", "label"))
    if not answer and "gold" in record:
        answer = choice_answer(record.get("choices"), record.get("gold"))
    if not answer:
        return None
    item = {"question": prompt, "answer": answer}
    if "choices" in record:
        item["choices"] = record["choices"]
    if "gold" in record:
        item["gold"] = record["gold"]
    return item


def valid_sft(item: Dict[str, Any], args: argparse.Namespace) -> bool:
    messages = item["conversations"]
    prompt = "\n".join(m["value"] for m in messages if m["from"] == "human")
    answer = "\n".join(m["value"] for m in messages if m["from"] == "gpt")
    return valid_lengths(prompt, answer, args)


def valid_dpo(item: Dict[str, Any], args: argparse.Namespace) -> bool:
    prompt = "\n".join(m["value"] for m in item["conversations"] if m["from"] == "human")
    if not valid_lengths(prompt, item["chosen"], args):
        return False
    if not valid_lengths(prompt, item["rejected"], args):
        return False
    chosen_len = max(len(item["chosen"]), 1)
    rejected_len = max(len(item["rejected"]), 1)
    return max(chosen_len, rejected_len) / min(chosen_len, rejected_len) <= 3.0


def valid_lengths(prompt: str, answer: str, args: argparse.Namespace) -> bool:
    return (
        args.min_prompt_chars <= len(prompt) <= args.max_prompt_chars
        and args.min_answer_chars <= len(answer) <= args.max_answer_chars
    )


def stable_key(item: Dict[str, Any]) -> str:
    if "conversations" in item:
        text = "\n".join(m["value"] for m in item["conversations"])
    else:
        text = json.dumps(item, ensure_ascii=False, sort_keys=True)
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def dedupe_and_limit(
    items: List[Tuple[str, Dict[str, Any]]],
    max_total: int,
    max_per_source: int,
) -> List[Tuple[str, Dict[str, Any]]]:
    seen = set()
    source_counts: Counter[str] = Counter()
    result = []
    for source, item in items:
        key = stable_key(item)
        if key in seen:
            continue
        if source_counts[source] >= max_per_source:
            continue
        seen.add(key)
        source_counts[source] += 1
        result.append((source, item))
        if max_total > 0 and len(result) >= max_total:
            break
    return result


def split_items(
    items: List[Tuple[str, Dict[str, Any]]],
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    total = len(shuffled)
    val_n = int(total * validation_ratio)
    test_n = int(total * test_ratio)
    val = [item for _, item in shuffled[:val_n]]
    test = [item for _, item in shuffled[val_n : val_n + test_n]]
    train = [item for _, item in shuffled[val_n + test_n :]]
    return train, val, test


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir not found: {raw_dir}")

    sft_items: List[Tuple[str, Dict[str, Any]]] = []
    dpo_items: List[Tuple[str, Dict[str, Any]]] = []
    eval_items: List[Tuple[str, Dict[str, Any]]] = []
    scanned_by_source: Counter[str] = Counter()
    kept_by_source: Dict[str, Counter[str]] = defaultdict(Counter)

    for source, record in iter_records(raw_dir):
        scanned_by_source[source] += 1
        sft = to_sft(record)
        if sft and valid_sft(sft, args):
            sft_items.append((source, sft))
            kept_by_source[source]["sft"] += 1
        dpo = to_dpo(record)
        if dpo and valid_dpo(dpo, args):
            dpo_items.append((source, dpo))
            kept_by_source[source]["dpo"] += 1
        ev = to_eval(record)
        if ev:
            eval_items.append((source, ev))
            kept_by_source[source]["eval"] += 1

    sft_items = dedupe_and_limit(sft_items, args.max_sft_samples, args.max_per_source)
    dpo_items = dedupe_and_limit(dpo_items, args.max_dpo_samples, args.max_per_source)
    eval_items = dedupe_and_limit(eval_items, args.max_eval_samples, args.max_per_source)

    sft_train, sft_val, sft_test = split_items(sft_items, args.validation_ratio, args.test_ratio, args.seed)
    dpo_train, dpo_val, _ = split_items(dpo_items, args.validation_ratio, 0.0, args.seed)

    counts = {
        "sft_train": write_jsonl(output_dir / "finance_sft" / "train" / "train.jsonl", sft_train),
        "sft_validation": write_jsonl(output_dir / "finance_sft" / "validation" / "validation.jsonl", sft_val),
        "sft_test": write_jsonl(output_dir / "finance_sft" / "test" / "test.jsonl", sft_test),
        "dpo_train": write_jsonl(output_dir / "finance_reward" / "train" / "train.jsonl", dpo_train),
        "dpo_validation": write_jsonl(output_dir / "finance_reward" / "validation" / "validation.jsonl", dpo_val),
        "eval": write_jsonl(output_dir / "finance_eval" / "finance_eval.jsonl", [item for _, item in eval_items]),
    }

    manifest = {
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "counts": counts,
        "scanned_by_source": dict(scanned_by_source),
        "kept_by_source": {source: dict(counter) for source, counter in kept_by_source.items()},
        "system_prompt": SYSTEM_PROMPT,
    }
    manifest_path = output_dir / "finance_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
