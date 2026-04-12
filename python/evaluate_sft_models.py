#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def get_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Tokenizer has neither a pad token nor eos token.")


def resolve_path(path_like: str) -> Path:
    return Path(path_like).expanduser().resolve()


def first_json_obj(text: str) -> str | None:
    text = text.strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def parse_obj(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    stripped = text.strip()
    if not stripped:
        return None, ["empty_output"]

    frag = first_json_obj(stripped)
    if frag is None:
        return None, ["invalid_json"]

    if frag != stripped:
        reasons.append("wrapped_json")

    try:
        obj = json.loads(frag)
    except Exception:
        return None, ["invalid_json"]

    if not isinstance(obj, dict):
        return None, ["non_object_json"]

    return obj, reasons


def norm(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    return value


def infer_schema_from_refs(references: list[str]) -> tuple[list[str], dict[str, tuple[float, float]], dict[str, set[Any]]]:
    parsed = []
    for ref in references:
        obj, _ = parse_obj(ref)
        if obj is not None:
            parsed.append(obj)

    if not parsed:
        raise ValueError("Could not infer schema from references.")

    keys = sorted(set.intersection(*(set(obj.keys()) for obj in parsed)))

    numeric_ranges: dict[str, tuple[float, float]] = {}
    categorical_values: dict[str, set[Any]] = {}

    for key in keys:
        vals = [obj[key] for obj in parsed]
        is_numeric = True
        numeric_vals = []
        for v in vals:
            try:
                numeric_vals.append(float(v))
            except Exception:
                is_numeric = False
                break
        if is_numeric:
            numeric_ranges[key] = (min(numeric_vals), max(numeric_vals))
        else:
            categorical_values[key] = {norm(v) for v in vals}

    return keys, numeric_ranges, categorical_values


def score_struct(prediction: str, reference: str, schema) -> dict[str, Any]:
    keys, numeric_ranges, categorical_values = schema

    pred_obj, parse_reasons = parse_obj(prediction)
    ref_obj, _ = parse_obj(reference)

    if ref_obj is None:
        raise ValueError("Reference JSON is invalid.")

    reasons = list(parse_reasons)
    if pred_obj is None:
        return {
            "accuracy": 0.0,
            "hallucinated": 1,
            "a1_score": 0.0,
            "exact_json_match": 0,
            "reasons": reasons or ["invalid_json"],
        }

    missing_keys = [k for k in keys if k not in pred_obj]
    extra_keys = [k for k in pred_obj if k not in keys]

    reasons.extend(["missing_keys"] if missing_keys else [])
    reasons.extend(["extra_keys"] if extra_keys else [])

    field_scores = []
    for key in keys:
        if key not in pred_obj or key not in ref_obj:
            field_scores.append(0.0)
            continue

        pred_val = pred_obj[key]
        ref_val = ref_obj[key]

        if key in numeric_ranges:
            try:
                pred_num = float(pred_val)
                ref_num = float(ref_val)
                lo, hi = numeric_ranges[key]
                denom = max(hi - lo, 1.0)
                score = max(0.0, 1.0 - abs(pred_num - ref_num) / denom)
                field_scores.append(score)
            except Exception:
                reasons.append(f"non_numeric:{key}")
                field_scores.append(0.0)
        else:
            pred_norm = norm(pred_val)
            ref_norm = norm(ref_val)
            allowed = categorical_values.get(key, set())
            if allowed and pred_norm not in allowed:
                reasons.append(f"out_of_domain:{key}")
            field_scores.append(1.0 if pred_norm == ref_norm else 0.0)

    accuracy = sum(field_scores) / max(len(keys), 1)
    hallucinated = 1 if reasons else 0
    exact_json_match = int(
        pred_obj == ref_obj and not missing_keys and not extra_keys and not parse_reasons
    )
    a1_score = accuracy * (1 - hallucinated)

    return {
        "accuracy": accuracy,
        "hallucinated": hallucinated,
        "a1_score": a1_score,
        "exact_json_match": exact_json_match,
        "reasons": sorted(set(reasons)),
    }


@dataclass
class PromptExample:
    sample_id: int
    prompt_text: str
    reference_text: str


def load_examples(dataset_path: Path, tokenizer, max_samples: int = 0) -> list[PromptExample]:
    examples: list[PromptExample] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            messages = row["messages"]
            assistant = [m for m in messages if m["role"] == "assistant"]
            prompt_messages = [m for m in messages if m["role"] != "assistant"]
            if len(assistant) != 1:
                continue

            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            reference_text = assistant[0]["content"]
            examples.append(PromptExample(idx, prompt_text, reference_text))

            if max_samples and len(examples) >= max_samples:
                break

    if not examples:
        raise ValueError("No valid examples loaded.")
    return examples


def load_causal_lm(model_path: str, load_mode: str, trust_remote_code: bool):
    compute_dtype = get_compute_dtype()
    if load_mode == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
    else:
        kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if compute_dtype != torch.float32:
            kwargs["torch_dtype"] = compute_dtype
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if torch.cuda.is_available():
            model.to("cuda")

    model.eval()
    model.config.use_cache = True
    return model


def attach_adapter(model, adapter_path: str | None):
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        model.eval()
    return model


def compute_eval_loss(model, tokenizer, prompt_text: str, reference_text: str, max_length: int) -> float:
    eos = tokenizer.eos_token or ""
    prompt_tokens = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    full_tokens = tokenizer(
        f"{prompt_text}{reference_text}{eos}",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]
    labels = input_ids.clone()

    prompt_len = prompt_tokens["input_ids"].shape[1]
    labels[:, :prompt_len] = -100

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return float(out.loss.item())


def evaluate_model(
    label: str,
    model,
    tokenizer,
    examples: list[PromptExample],
    schema,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_length_for_loss: int,
):
    rows = []
    losses = []

    for ex in examples:
        inputs = tokenizer(ex.prompt_text, return_tensors="pt").to(next(model.parameters()).device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": temperature > 0.0,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        latency = time.perf_counter() - start

        prompt_len = inputs["input_ids"].shape[1]
        pred_ids = outputs[0][prompt_len:]
        prediction = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()

        metrics = score_struct(prediction, ex.reference_text, schema)
        loss_val = compute_eval_loss(model, tokenizer, ex.prompt_text, ex.reference_text, max_length_for_loss)
        losses.append(loss_val)

        rows.append(
            {
                "id": ex.sample_id,
                "prompt": ex.prompt_text,
                "reference": ex.reference_text,
                "prediction": prediction,
                "latency_sec": latency,
                "loss": loss_val,
                **metrics,
            }
        )

    n = len(rows)
    summary = {
        "num_examples": n,
        "avg_latency_sec": round(sum(r["latency_sec"] for r in rows) / n, 4),
        "median_latency_sec": round(sorted(r["latency_sec"] for r in rows)[n // 2], 4),
        "accuracy": round(sum(r["accuracy"] for r in rows) / n, 4),
        "hallucination_rate": round(sum(r["hallucinated"] for r in rows) / n, 4),
        "a1_score": round(sum(r["a1_score"] for r in rows) / n, 4),
        "exact_json_match": round(sum(r["exact_json_match"] for r in rows) / n, 4),
        "eval_loss": round(sum(losses) / len(losses), 4),
        "hallucination_reason_counts": {},
    }

    reason_counts: dict[str, int] = {}
    for r in rows:
        for reason in r["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    summary["hallucination_reason_counts"] = reason_counts

    return summary, rows


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate base vs LoRA vs QLoRA locally.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-load-mode", choices=["4bit", "16bit"], default="4bit")
    parser.add_argument("--lora-adapter", default="")
    parser.add_argument("--lora-load-mode", choices=["4bit", "16bit"], default="16bit")
    parser.add_argument("--qlora-adapter", default="")
    parser.add_argument("--qlora-load-mode", choices=["4bit", "16bit"], default="4bit")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-length-for-loss", type=int, default=768)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-report", default="reports/local_eval.json")
    parser.add_argument("--output-predictions", default="reports/local_eval_predictions.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=args.trust_remote_code)
    ensure_pad_token(tokenizer)

    examples = load_examples(resolve_path(args.dataset), tokenizer, args.max_samples)
    schema = infer_schema_from_refs([ex.reference_text for ex in examples])

    models_to_run = [
        ("base", "", args.base_load_mode),
    ]
    if args.lora_adapter:
        models_to_run.append(("sft_lora", args.lora_adapter, args.lora_load_mode))
    if args.qlora_adapter:
        models_to_run.append(("sft_qlora", args.qlora_adapter, args.qlora_load_mode))

    report = {
        "config": {
            "base_model": args.base_model,
            "dataset": args.dataset,
            "max_samples": args.max_samples,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "load_modes": {},
        },
        "metric_definitions": {
            "accuracy": "Mean per-field score against the structured weather reference JSON.",
            "hallucination_rate": "Fraction of generations with formatting or schema violations.",
            "a1_score": "Mean(accuracy * (1 - hallucinated)).",
            "exact_json_match": "Fraction of samples whose JSON matches the reference exactly.",
            "eval_loss": "Mean teacher-forced loss on the gold completion tokens only.",
        },
        "models": {},
        "comparisons_vs_base": {},
    }

    merged_rows: dict[int, dict[str, Any]] = {
        ex.sample_id: {"id": ex.sample_id, "prompt": ex.prompt_text, "reference": ex.reference_text}
        for ex in examples
    }

    base_summary = None
    for label, adapter_path, load_mode in models_to_run:
        model = load_causal_lm(args.base_model, load_mode, args.trust_remote_code)
        model = attach_adapter(model, adapter_path if adapter_path else None)

        summary, rows = evaluate_model(
            label=label,
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            schema=schema,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_length_for_loss=args.max_length_for_loss,
        )
        report["models"][label] = summary
        report["config"]["load_modes"][label] = load_mode

        if label == "base":
            base_summary = summary

        for row in rows:
            merged_rows[row["id"]].update(
                {
                    f"{label}_prediction": row["prediction"],
                    f"{label}_latency_sec": row["latency_sec"],
                    f"{label}_loss": row["loss"],
                    f"{label}_accuracy": row["accuracy"],
                    f"{label}_hallucinated": row["hallucinated"],
                    f"{label}_a1_score": row["a1_score"],
                    f"{label}_exact_json_match": row["exact_json_match"],
                    f"{label}_hallucination_reasons": row["reasons"],
                }
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if base_summary is not None:
        for label, summary in report["models"].items():
            if label == "base":
                continue
            report["comparisons_vs_base"][label] = {
                "accuracy": round(summary["accuracy"] - base_summary["accuracy"], 4),
                "hallucination_rate": round(summary["hallucination_rate"] - base_summary["hallucination_rate"], 4),
                "a1_score": round(summary["a1_score"] - base_summary["a1_score"], 4),
                "exact_json_match": round(summary["exact_json_match"] - base_summary["exact_json_match"], 4),
                "eval_loss": round(summary["eval_loss"] - base_summary["eval_loss"], 4),
            }

    out_report = resolve_path(args.output_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    out_preds = resolve_path(args.output_predictions)
    out_preds.parent.mkdir(parents=True, exist_ok=True)
    with out_preds.open("w", encoding="utf-8") as f:
        for _, row in sorted(merged_rows.items()):
            f.write(json.dumps(row) + "\n")

    print(json.dumps(report, indent=2))
