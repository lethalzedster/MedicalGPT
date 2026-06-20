"""Use DeepSeek as an OpenAI-compatible LLM-as-Judge for FinAlign DPO data.

The API key must be provided with DEEPSEEK_API_KEY. This script is resumable:
existing ids in --output_file are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


JUDGE_SYSTEM_PROMPT = (
    "你是严谨的中文金融问答偏好评审员。你只根据给定问题、参考答案和候选回答进行判断，"
    "不要补充外部事实。请优先选择更准确、更忠实、更合规、更有风险提示的回答。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepSeek pairwise/multi-candidate judge.")
    parser.add_argument("--candidate_file", action="append", required=True, help="Format: model_name=path.jsonl")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--base_url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--sleep_seconds", type=float, default=0.2)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for row in read_jsonl(path):
        if row.get("id"):
            ids.add(str(row["id"]))
    return ids


def parse_candidate_arg(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--candidate_file must be model_name=path, got: {value}")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"empty model name in --candidate_file: {value}")
    return name, Path(path)


def load_candidates(candidate_args: Iterable[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    by_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for arg in candidate_args:
        model_name, path = parse_candidate_arg(arg)
        rows = {}
        for row in read_jsonl(path):
            item_id = row.get("id")
            if item_id:
                rows[str(item_id)] = row
        by_model[model_name] = rows
    return by_model


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def common_ids(by_model: Dict[str, Dict[str, Dict[str, Any]]]) -> List[str]:
    id_sets = [set(rows) for rows in by_model.values()]
    if not id_sets:
        return []
    return sorted(set.intersection(*id_sets))


def build_user_prompt(prompt: str, reference: str, candidates: List[Dict[str, str]]) -> str:
    candidate_text = "\n\n".join(
        f"候选 {idx + 1} | model={candidate['model']}\n{candidate['response']}"
        for idx, candidate in enumerate(candidates)
    )
    reference_block = reference if reference else "无"
    return f"""请评审以下金融问答候选回答，并选出最适合作为 DPO chosen 的回答和最适合作为 rejected 的回答。

问题：
{prompt}

参考答案/材料：
{reference_block}

候选回答：
{candidate_text}

评分维度：
1. 专业准确性
2. 数值一致性
3. 材料忠实度
4. 风险提示充分性
5. 合规性
6. 表达清晰度

只输出 JSON，不要输出 Markdown。格式如下：
{{
  "winner_model": "候选回答对应的 model",
  "loser_model": "候选回答对应的 model",
  "confidence": 0.0,
  "score_gap": 0,
  "dimension_scores": {{
    "professional_accuracy": {{"winner": 0, "loser": 0}},
    "numeric_consistency": {{"winner": 0, "loser": 0}},
    "faithfulness": {{"winner": 0, "loser": 0}},
    "risk_disclosure": {{"winner": 0, "loser": 0}},
    "compliance": {{"winner": 0, "loser": 0}},
    "clarity": {{"winner": 0, "loser": 0}}
  }},
  "bad_case_tags": ["numeric_error", "unfaithful", "missing_risk_disclosure"],
  "reason": "一句话说明原因"
}}"""


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def call_deepseek(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def main() -> None:
    args = parse_args()
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key and not args.dry_run:
        raise RuntimeError("DEEPSEEK_API_KEY is required")

    by_model = load_candidates(args.candidate_file)
    ids = common_ids(by_model)
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    if args.max_samples > 0:
        ids = ids[: args.max_samples]

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    done_ids = read_existing_ids(output_file)
    written = 0

    with output_file.open("a", encoding="utf-8") as f:
        for item_id in ids:
            if item_id in done_ids:
                continue
            rows = [rows_by_id[item_id] for rows_by_id in by_model.values()]
            prompt = clean_text(rows[0].get("prompt") or rows[0].get("question"))
            reference = clean_text(rows[0].get("reference") or rows[0].get("answer"))
            candidates = []
            for model_name, rows_by_id in by_model.items():
                row = rows_by_id[item_id]
                response = clean_text(row.get("prediction") or row.get("response"))
                if response:
                    candidates.append({"model": model_name, "response": response})
            if len(candidates) < 2 or not prompt:
                continue

            user_prompt = build_user_prompt(prompt, reference, candidates)
            if args.dry_run:
                print(user_prompt[:2000])
                break

            raw_content = ""
            parsed: Dict[str, Any] = {}
            last_error = ""
            for attempt in range(args.max_retries):
                try:
                    raw_content = call_deepseek(
                        api_key=api_key,
                        base_url=args.base_url,
                        model=args.model,
                        system_prompt=JUDGE_SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        timeout=args.timeout,
                    )
                    parsed = extract_json(raw_content)
                    break
                except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError) as exc:
                    last_error = str(exc)
                    time.sleep(min(2 ** attempt, 8))
            if not parsed:
                parsed = {"error": last_error, "raw_content": raw_content}

            winner_model = clean_text(parsed.get("winner_model"))
            loser_model = clean_text(parsed.get("loser_model"))
            candidate_by_model = {candidate["model"]: candidate["response"] for candidate in candidates}
            out = {
                "id": item_id,
                "prompt": prompt,
                "reference": reference,
                "candidates": candidates,
                "winner_model": winner_model,
                "loser_model": loser_model,
                "chosen": candidate_by_model.get(winner_model, ""),
                "rejected": candidate_by_model.get(loser_model, ""),
                "confidence": parsed.get("confidence", 0),
                "score_gap": parsed.get("score_gap", 0),
                "dimension_scores": parsed.get("dimension_scores", {}),
                "bad_case_tags": parsed.get("bad_case_tags", []),
                "reason": parsed.get("reason", ""),
                "judge_model": args.model,
                "judge_raw": parsed,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    print(f"wrote {written} new judge rows to {output_file}")


if __name__ == "__main__":
    main()
