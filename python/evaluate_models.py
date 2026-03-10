#!/usr/bin/env python
"""
Compare a base causal LM against the same model with a LoRA adapter.

Expected dataset format (JSONL or JSON):
- Each sample should include a prompt-like field:
  prompt | input | instruction | question | text
- Optional reference field for scoring:
  reference | target | answer | output | response
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
EXPECTED_PROJECT_NAME = "llm-finetuning"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate base model vs base+LoRA on the same prompts."
    )
    parser.add_argument("--base-model", required=True, help="HF model id or local path")
    parser.add_argument(
        "--lora-adapter", required=True, help="Path or HF repo for the LoRA adapter"
    )
    parser.add_argument("--dataset", required=True, help="Path to .json or .jsonl data")
    parser.add_argument(
        "--output-report",
        default="lora_eval_report.json",
        help="Output JSON report path",
    )
    parser.add_argument(
        "--output-predictions",
        default="",
        help="Optional output JSONL with per-sample predictions",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Max samples to evaluate (after optional shuffle)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle samples before truncating to max-samples",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading model/tokenizer",
    )
    parser.add_argument(
        "--prompt-key",
        default="",
        help="Explicit prompt key. If omitted, a default key search is used.",
    )
    parser.add_argument(
        "--reference-key",
        default="",
        help="Explicit reference key. If omitted, a default key search is used.",
    )
    return parser.parse_args()


def ensure_llm_finetuning_project() -> Path:
    if PROJECT_ROOT.name != EXPECTED_PROJECT_NAME:
        raise SystemExit(
            f"This script only runs inside '{EXPECTED_PROJECT_NAME}'. "
            f"Detected project root: {PROJECT_ROOT}"
        )

    required_paths = [
        PROJECT_ROOT / "python" / "train_model.py",
        PROJECT_ROOT / "requirements.txt",
    ]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required_paths if not path.exists()]
    if missing:
        raise SystemExit(
            "Project validation failed. Missing required files: "
            + ", ".join(missing)
        )

    return PROJECT_ROOT


def resolve_project_path(path_value: str, project_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {i}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL line {i} is not an object.")
                records.append(row)
        return records

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError("JSON list dataset must contain objects.")
        return payload
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            if not all(isinstance(item, dict) for item in payload["data"]):
                raise ValueError("JSON data list must contain objects.")
            return payload["data"]
    raise ValueError("Unsupported dataset format. Use JSONL or JSON list/object with 'data'.")


def first_present_key(record: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        if key in record:
            return key
    return None


def extract_chat_prompt_and_reference(
    record: dict[str, Any],
) -> tuple[list[dict[str, str]], str | None] | None:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    normalized_messages = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role is None or content is None:
            continue
        normalized_messages.append({"role": str(role), "content": str(content)})

    if not normalized_messages:
        return None

    reference = None
    prompt_messages = normalized_messages
    if normalized_messages[-1]["role"] == "assistant":
        reference = normalized_messages[-1]["content"]
        prompt_messages = normalized_messages[:-1]

    if not prompt_messages:
        return None

    return prompt_messages, reference


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    common = 0
    for token, pred_count in pred_counts.items():
        common += min(pred_count, ref_counts.get(token, 0))

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_examples(
    records: list[dict[str, Any]],
    prompt_key: str,
    reference_key: str,
) -> tuple[list[dict[str, Any]], str, str | None]:
    prompt_candidates = ["prompt", "input", "instruction", "question", "text"]
    reference_candidates = ["reference", "target", "answer", "output", "response"]

    resolved_prompt_key = prompt_key
    resolved_reference_key = reference_key or None

    if not resolved_prompt_key:
        for record in records:
            found = first_present_key(record, prompt_candidates)
            if found:
                resolved_prompt_key = found
                break

    if not resolved_prompt_key:
        for record in records:
            parsed = extract_chat_prompt_and_reference(record)
            if parsed is not None:
                resolved_prompt_key = "messages"
                break

    if not resolved_prompt_key:
        raise ValueError(
            "Could not infer prompt key. Pass --prompt-key explicitly."
        )

    if not resolved_reference_key:
        for record in records:
            found = first_present_key(record, reference_candidates)
            if found:
                resolved_reference_key = found
                break

    examples = []
    for i, record in enumerate(records):
        if resolved_prompt_key == "messages":
            parsed = extract_chat_prompt_and_reference(record)
            if parsed is None:
                continue
            prompt_messages, inferred_reference = parsed
            prompt = "\n".join(
                f"{item['role']}: {item['content']}" for item in prompt_messages
            )
            reference = inferred_reference
            if resolved_reference_key and resolved_reference_key in record:
                ref_value = record.get(resolved_reference_key)
                if ref_value is not None:
                    reference = str(ref_value)

            examples.append(
                {
                    "id": i,
                    "prompt": prompt,
                    "chat_messages": prompt_messages,
                    "reference": reference,
                    "raw": record,
                }
            )
            continue

        if resolved_prompt_key not in record:
            continue
        prompt_value = record.get(resolved_prompt_key, "")
        if prompt_value is None:
            continue
        prompt = str(prompt_value)
        if not prompt.strip():
            continue

        reference = None
        if resolved_reference_key and resolved_reference_key in record:
            ref_value = record.get(resolved_reference_key)
            if ref_value is not None:
                reference = str(ref_value)

        examples.append({"id": i, "prompt": prompt, "reference": reference, "raw": record})

    if not examples:
        raise ValueError(
            f"No usable examples found with prompt key '{resolved_prompt_key}'."
        )

    if resolved_prompt_key == "messages" and resolved_reference_key is None:
        resolved_reference_key = "messages[-1].assistant"

    return examples, resolved_prompt_key, resolved_reference_key


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_tokenizer(model_name_or_path: str, trust_remote_code: bool):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_name_or_path: str, trust_remote_code: bool):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
    )
    return model


def apply_lora(model, adapter_path: str):
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_path)


def generate_one(
    model,
    tokenizer,
    example: dict[str, Any],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> tuple[str, float]:
    import torch

    prompt_text = example["prompt"]
    chat_messages = example.get("chat_messages")
    if chat_messages is not None and hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": temperature > 0.0,
    }
    if temperature > 0.0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    elapsed = time.perf_counter() - start

    completion_ids = output_ids[0][input_len:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return completion, elapsed


def evaluate_model(
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    results = []
    latencies = []
    em_scores = []
    f1_scores = []

    for ex in examples:
        prediction, latency = generate_one(
            model=model,
            tokenizer=tokenizer,
            example=ex,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )
        latencies.append(latency)
        item = {
            "id": ex["id"],
            "prompt": ex["prompt"],
            "prediction": prediction,
            "latency_sec": latency,
        }

        reference = ex.get("reference")
        if reference is not None:
            em = int(normalize_text(prediction) == normalize_text(reference))
            f1 = token_f1(prediction, reference)
            item["reference"] = reference
            item["exact_match"] = em
            item["token_f1"] = f1
            em_scores.append(em)
            f1_scores.append(f1)

        results.append(item)

    summary = {
        "num_examples": len(results),
        "avg_latency_sec": round(statistics.mean(latencies), 4) if latencies else None,
        "median_latency_sec": round(statistics.median(latencies), 4) if latencies else None,
    }
    if em_scores:
        summary["exact_match"] = round(sum(em_scores) / len(em_scores), 4)
        summary["token_f1"] = round(sum(f1_scores) / len(f1_scores), 4)

    return {"summary": summary, "predictions": results}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def main() -> None:
    args = parse_args()
    project_root = ensure_llm_finetuning_project()
    set_seed(args.seed)

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import peft  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "Missing dependencies. Install: pip install torch transformers peft"
        ) from exc

    dataset_path = resolve_project_path(args.dataset, project_root)
    records = load_records(dataset_path)
    examples, prompt_key, reference_key = extract_examples(
        records=records,
        prompt_key=args.prompt_key,
        reference_key=args.reference_key,
    )

    if args.shuffle:
        random.shuffle(examples)
    if args.max_samples > 0:
        examples = examples[: args.max_samples]

    tokenizer = load_tokenizer(args.base_model, trust_remote_code=args.trust_remote_code)

    # Evaluate base model first.
    base_model = load_base_model(args.base_model, trust_remote_code=args.trust_remote_code)
    base_eval = evaluate_model(
        model=base_model,
        tokenizer=tokenizer,
        examples=examples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Free memory before loading LoRA variant.
    try:
        import torch

        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    lora_base = load_base_model(args.base_model, trust_remote_code=args.trust_remote_code)
    lora_model = apply_lora(lora_base, args.lora_adapter)
    lora_eval = evaluate_model(
        model=lora_model,
        tokenizer=tokenizer,
        examples=examples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    report = {
        "config": {
            "base_model": args.base_model,
            "lora_adapter": args.lora_adapter,
            "dataset": str(dataset_path),
            "prompt_key": prompt_key,
            "reference_key": reference_key,
            "max_samples": args.max_samples,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        "base_model": base_eval["summary"],
        "base_plus_lora": lora_eval["summary"],
    }

    output_report_path = resolve_project_path(args.output_report, project_root)
    write_json(output_report_path, report)

    if args.output_predictions:
        output_predictions_path = resolve_project_path(args.output_predictions, project_root)
        combined_rows = []
        for base_row, lora_row in zip(base_eval["predictions"], lora_eval["predictions"]):
            row = {
                "id": base_row["id"],
                "prompt": base_row["prompt"],
                "reference": base_row.get("reference"),
                "base_prediction": base_row["prediction"],
                "lora_prediction": lora_row["prediction"],
                "base_latency_sec": base_row["latency_sec"],
                "lora_latency_sec": lora_row["latency_sec"],
            }
            if "exact_match" in base_row:
                row["base_exact_match"] = base_row["exact_match"]
                row["lora_exact_match"] = lora_row.get("exact_match")
                row["base_token_f1"] = base_row.get("token_f1")
                row["lora_token_f1"] = lora_row.get("token_f1")
            combined_rows.append(row)
        write_jsonl(output_predictions_path, combined_rows)

    print("\n=== Evaluation Complete ===")
    print(json.dumps(report, indent=2))
    if args.output_predictions:
        print(f"\nWrote per-sample predictions to: {output_predictions_path}")
    print(f"Wrote summary report to: {output_report_path}")


if __name__ == "__main__":
    main()
