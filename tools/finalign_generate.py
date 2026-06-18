"""Generate model responses for FinAlign prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate responses for FinAlign prompts.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--load_in_4bit", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_prompt(tokenizer: AutoTokenizer, text: str) -> str:
    messages = [{"role": "user", "content": text}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return text


def main() -> None:
    args = parse_args()
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant_config,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, device_map="auto")
    model.eval()

    rows: List[Dict[str, Any]] = list(read_jsonl(input_file))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    do_sample = args.temperature > 0
    with output_file.open("w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="generate"):
            prompt_text = row.get("prompt") or row.get("question") or ""
            prompt = build_prompt(tokenizer, prompt_text)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            generate_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                generate_kwargs["temperature"] = args.temperature
            with torch.no_grad():
                generated = model.generate(**inputs, **generate_kwargs)
            new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            out = dict(row)
            out["prediction"] = response
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
