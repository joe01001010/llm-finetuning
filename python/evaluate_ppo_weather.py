#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ppo_config import RewardConfig, add_reward_config_args, reward_config_from_args
from ppo_data import WeatherPromptExample, load_weather_prompt_examples
from ppo_reward import compute_weather_reward
from ppo_utils import ensure_pad_token, resolve_path, save_json, write_jsonl
from weather_json_metrics import infer_schema, norm, token_f1


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate base, SFT, and PPO weather models")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--output-predictions", default="")
    parser.add_argument("--sft-lora-adapter", default="")
    parser.add_argument("--sft-qlora-adapter", default="")
    parser.add_argument("--ppo-base-adapter", default="")
    parser.add_argument("--ppo-lora-adapter", default="")
    parser.add_argument("--ppo-qlora-adapter", default="")
    parser.add_argument("--reward-config-path", default="")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--base-load-mode", choices=["4bit", "16bit", "32bit"], default="16bit")
    parser.add_argument("--trust-remote-code", action="store_true")
    add_reward_config_args(parser)
    return parser.parse_args()


def load_runtime_config(adapter_path: Path) -> dict[str, Any]:
    for file_name in ("adapter_runtime_config.json", "ppo_runtime_config.json"):
        config_path = adapter_path / file_name
        if config_path.exists():
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
    return {}


def infer_load_mode(label: str, adapter_path: Path | None, base_load_mode: str) -> str:
    if label == "base":
        return base_load_mode
    if adapter_path is not None:
        runtime_config = load_runtime_config(adapter_path)
        runtime_mode = runtime_config.get("base_model_load_mode")
        if runtime_mode in {"4bit", "16bit", "32bit"}:
            return runtime_mode
    if "qlora" in label:
        return "4bit"
    return "16bit"


def load_tokenizer(tokenizer_path: str, base_model_path: str, trust_remote_code: bool):
    load_path = tokenizer_path or base_model_path
    tokenizer = AutoTokenizer.from_pretrained(load_path, use_fast=True, trust_remote_code=trust_remote_code)
    ensure_pad_token(tokenizer)
    tokenizer.padding_side = "left"
    return tokenizer


def get_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_base_model(model_path: str, load_mode: str, trust_remote_code: bool):
    compute_dtype = get_compute_dtype()
    if load_mode == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )

    load_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if load_mode == "16bit" and compute_dtype != torch.float32:
        load_kwargs["torch_dtype"] = compute_dtype
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
    return model


def load_model_variant(base_model_path: str, adapter_path: Path | None, load_mode: str, trust_remote_code: bool):
    model = load_base_model(base_model_path, load_mode, trust_remote_code)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model


def infer_model_device(model) -> torch.device:
    for parameter in model.parameters():
        return parameter.device
    raise RuntimeError("Unable to infer model device for generation.")


def load_reward_config(args: argparse.Namespace) -> RewardConfig:
    if args.reward_config_path:
        payload = json.loads(resolve_path(args.reward_config_path, PROJECT_ROOT).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Reward config JSON must contain an object.")
        return RewardConfig(**payload)
    return reward_config_from_args(args)


def generate_prediction(model, tokenizer, prompt_text: str, args: argparse.Namespace) -> tuple[str, float]:
    inputs = tokenizer(prompt_text, return_tensors="pt")
    device = infer_model_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": args.temperature > 0.0,
    }
    if args.temperature > 0.0:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    prompt_width = int(inputs["input_ids"].shape[1])
    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    latency = time.perf_counter() - start_time
    prediction = tokenizer.decode(outputs[0][prompt_width:], skip_special_tokens=True).strip()
    return prediction, latency


def evaluate_model(
    label: str,
    model,
    tokenizer,
    examples: list[WeatherPromptExample],
    schema,
    reward_config: RewardConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    exact_match_values: list[int] = []
    token_f1_values: list[float] = []
    latency_values: list[float] = []
    reward_values: list[float] = []
    accuracy_values: list[float] = []
    hallucination_values: list[float] = []
    a1_values: list[float] = []
    exact_json_values: list[float] = []
    reason_counts: dict[str, int] = {}

    for example in examples:
        prediction, latency = generate_prediction(model, tokenizer, example.prompt_text, args)
        reward_value, breakdown = compute_weather_reward(
            prediction=prediction,
            reference=example.reference_text,
            schema=schema,
            config=reward_config,
        )

        exact_match = int(norm(prediction) == norm(example.reference_text))
        token_f1_value = token_f1(prediction, example.reference_text)

        exact_match_values.append(exact_match)
        token_f1_values.append(token_f1_value)
        latency_values.append(latency)
        reward_values.append(reward_value)
        accuracy_values.append(float(breakdown["accuracy"]))
        hallucination_values.append(float(breakdown["hallucinated"]))
        a1_values.append(float(breakdown["a1_score"]))
        exact_json_values.append(float(breakdown["exact_json_match"]))
        for reason in breakdown["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        rows.append(
            {
                "id": example.sample_id,
                "prompt": example.prompt_text,
                "reference": example.reference_text,
                "prediction": prediction,
                "latency_sec": round(latency, 6),
                "reward": round(reward_value, 6),
                "exact_match": exact_match,
                "token_f1": round(token_f1_value, 6),
                "accuracy": round(float(breakdown["accuracy"]), 6),
                "hallucinated": int(breakdown["hallucinated"]),
                "a1_score": round(float(breakdown["a1_score"]), 6),
                "exact_json_match": int(breakdown["exact_json_match"]),
                "hallucination_reasons": breakdown["reasons"],
            }
        )

    summary = {
        "num_examples": len(rows),
        "avg_latency_sec": round(statistics.mean(latency_values), 4) if latency_values else None,
        "median_latency_sec": round(statistics.median(latency_values), 4) if latency_values else None,
        "exact_match": round(sum(exact_match_values) / len(exact_match_values), 4) if exact_match_values else None,
        "token_f1": round(sum(token_f1_values) / len(token_f1_values), 4) if token_f1_values else None,
        "accuracy": round(statistics.mean(accuracy_values), 4) if accuracy_values else None,
        "hallucination_rate": round(statistics.mean(hallucination_values), 4) if hallucination_values else None,
        "a1_score": round(statistics.mean(a1_values), 4) if a1_values else None,
        "exact_json_match": round(statistics.mean(exact_json_values), 4) if exact_json_values else None,
        "avg_reward": round(statistics.mean(reward_values), 4) if reward_values else None,
        "hallucination_reason_counts": dict(sorted(reason_counts.items())),
    }
    return {"summary": summary, "predictions": rows}


def build_model_specs(args: argparse.Namespace) -> list[tuple[str, Path | None]]:
    specs: list[tuple[str, Path | None]] = [("base", None)]
    optional_paths = {
        "sft_lora": args.sft_lora_adapter,
        "sft_qlora": args.sft_qlora_adapter,
        "ppo_base": args.ppo_base_adapter,
        "ppo_lora": args.ppo_lora_adapter,
        "ppo_qlora": args.ppo_qlora_adapter,
    }
    for label, adapter_value in optional_paths.items():
        if adapter_value:
            specs.append((label, resolve_path(adapter_value, PROJECT_ROOT)))
    return specs


def merge_prediction_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    labels = list(results.keys())
    base_rows = results[labels[0]]["predictions"]
    merged_rows: list[dict[str, Any]] = []
    for row_index, base_row in enumerate(base_rows):
        merged = {
            "id": base_row["id"],
            "prompt": base_row["prompt"],
            "reference": base_row["reference"],
        }
        for label in labels:
            source = results[label]["predictions"][row_index]
            merged[f"{label}_prediction"] = source["prediction"]
            merged[f"{label}_latency_sec"] = source["latency_sec"]
            merged[f"{label}_reward"] = source["reward"]
            merged[f"{label}_accuracy"] = source["accuracy"]
            merged[f"{label}_hallucinated"] = source["hallucinated"]
            merged[f"{label}_a1_score"] = source["a1_score"]
            merged[f"{label}_exact_json_match"] = source["exact_json_match"]
            merged[f"{label}_hallucination_reasons"] = source["hallucination_reasons"]
        merged_rows.append(merged)
    return merged_rows


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    reward_config = load_reward_config(args)
    tokenizer = load_tokenizer(args.tokenizer_path, args.base_model, args.trust_remote_code)
    examples = load_weather_prompt_examples(
        dataset_path=resolve_path(args.dataset, PROJECT_ROOT),
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    schema = infer_schema([{"reference": example.reference_text} for example in examples])

    results: dict[str, dict[str, Any]] = {}
    load_modes: dict[str, str] = {}
    model_specs = build_model_specs(args)
    for label, adapter_path in model_specs:
        load_mode = infer_load_mode(label, adapter_path, args.base_load_mode)
        load_modes[label] = load_mode
        model = load_model_variant(
            base_model_path=args.base_model,
            adapter_path=adapter_path,
            load_mode=load_mode,
            trust_remote_code=args.trust_remote_code,
        )
        results[label] = evaluate_model(
            label=label,
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            schema=schema,
            reward_config=reward_config,
            args=args,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "config": {
            "base_model": args.base_model,
            "tokenizer_path": args.tokenizer_path or args.base_model,
            "dataset": str(resolve_path(args.dataset, PROJECT_ROOT)),
            "max_samples": args.max_samples,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "load_modes": load_modes,
        },
        "metric_definitions": {
            "accuracy": "Mean per-field score against the structured weather reference JSON.",
            "hallucination_rate": "Fraction of generations with formatting or schema violations.",
            "a1_score": "Mean(accuracy * (1 - hallucinated)).",
            "exact_json_match": "Fraction of samples whose JSON matches the reference exactly.",
            "avg_reward": "Average PPO reward under the configured structured weather reward function.",
        },
        "models": {label: payload["summary"] for label, payload in results.items()},
        "comparisons_vs_base": {},
    }

    base_summary = results["base"]["summary"]
    for label, payload in results.items():
        if label == "base":
            continue
        comparison = {}
        for metric_name in ("accuracy", "hallucination_rate", "a1_score", "exact_json_match", "avg_reward", "exact_match", "token_f1"):
            base_value = base_summary.get(metric_name)
            candidate_value = payload["summary"].get(metric_name)
            if isinstance(base_value, (int, float)) and isinstance(candidate_value, (int, float)):
                comparison[metric_name] = round(candidate_value - base_value, 4)
            else:
                comparison[metric_name] = None
        report["comparisons_vs_base"][label] = comparison

    report_path = resolve_path(args.output_report, PROJECT_ROOT)
    save_json(report_path, report)

    if args.output_predictions:
        prediction_rows = merge_prediction_rows(results)
        write_jsonl(resolve_path(args.output_predictions, PROJECT_ROOT), prediction_rows)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
