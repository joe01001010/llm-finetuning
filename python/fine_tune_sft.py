#!/usr/bin/env python
"""
Supervised LoRA / QLoRA training for the Seattle weather chat dataset.

This script keeps the original adapter-based SFT workflow intact so that PPO
post-training can be run as an optional second stage afterwards.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from ppo_data import WeatherPromptExample, load_weather_prompt_examples
from ppo_utils import (
    model_device,
    resolve_path,
    save_json,
    set_seed,
)
from weather_agent import WeatherModelAgent
from weather_json_metrics import infer_schema, score_struct


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL_PATH = "/local-containers/Qwen2-7B-Instruct"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "seattle_weather_chat_train.jsonl"
DEFAULT_EVAL_DATA_PATH = PROJECT_ROOT / "data" / "seattle_weather_chat_eval.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "adapter-weights"


@dataclass
class SFTFeature:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


class SupervisedWeatherDataset(Dataset):
    """
    Tokenized chat-style SFT dataset where the loss is only applied to the
    assistant response tokens.
    """

    def __init__(
        self,
        examples: list[WeatherPromptExample],
        tokenizer,
        max_length: int,
    ) -> None:
        self.features: list[SFTFeature] = []
        eos_suffix = tokenizer.eos_token or ""

        for example in examples:
            prompt_text = example.prompt_text
            completion_text = f"{example.reference_text}{eos_suffix}"
            prompt_tokens = tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            full_tokens = tokenizer(
                f"{prompt_text}{completion_text}",
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )

            input_ids = list(full_tokens["input_ids"])
            attention_mask = list(full_tokens["attention_mask"])
            prompt_length = len(prompt_tokens["input_ids"])
            if not input_ids or prompt_length >= len(input_ids):
                continue

            labels = input_ids.copy()
            labels[:prompt_length] = [-100] * prompt_length
            if all(label == -100 for label in labels):
                continue

            self.features.append(
                SFTFeature(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            )

        if not self.features:
            raise ValueError("No usable SFT training examples remained after tokenization.")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        feature = self.features[index]
        return {
            "input_ids": feature.input_ids,
            "attention_mask": feature.attention_mask,
            "labels": feature.labels,
        }


class SupervisedDataCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        input_ids = pad_sequence(
            [torch.tensor(feature["input_ids"], dtype=torch.long) for feature in features],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = pad_sequence(
            [torch.tensor(feature["attention_mask"], dtype=torch.long) for feature in features],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            [torch.tensor(feature["labels"], dtype=torch.long) for feature in features],
            batch_first=True,
            padding_value=-100,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LoRA / QLoRA weather adapters with supervised fine-tuning.",
    )
    parser.add_argument("--mode", choices=["lora", "qlora"], default="qlora")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory where adapter weights will be saved. Defaults to ./adapter-weights/<mode>.",
    )
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--eval-data-path", default=str(DEFAULT_EVAL_DATA_PATH))
    parser.add_argument("--sanity-check-samples", type=int, default=32)
    parser.add_argument("--sanity-check-max-new-tokens", type=int, default=48)
    parser.add_argument("--sanity-check-repetition-penalty", type=float, default=1.05)
    parser.add_argument("--skip-post-train-sanity-check", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--disable-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return resolve_path(args.output_dir, PROJECT_ROOT)
    return DEFAULT_OUTPUT_ROOT / args.mode


def prepare_model_for_training(agent: WeatherModelAgent, model, tokenizer, args: argparse.Namespace):
    model = agent.create_lora_adapter(
        model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    if tokenizer.pad_token_id is not None and hasattr(model.config, "pad_token_id"):
        model.config.pad_token_id = tokenizer.pad_token_id
    return model


def build_training_args(
    args: argparse.Namespace,
    output_dir: Path,
    mode: str,
) -> TrainingArguments:
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    use_fp16 = torch.cuda.is_available() and not use_bf16
    optim = "paged_adamw_8bit" if mode == "qlora" else "adamw_torch"

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=use_bf16,
        fp16=use_fp16,
        optim=optim,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        lr_scheduler_type="cosine",
        dataloader_pin_memory=torch.cuda.is_available(),
    )


def warn_if_risky_configuration(args: argparse.Namespace, dataset_size: int) -> list[str]:
    warnings: list[str] = []
    if args.mode == "qlora" and args.learning_rate > 1e-4:
        warnings.append(
            "QLoRA learning rates above 1e-4 can destabilize structured JSON outputs for this project."
        )
    if args.num_train_epochs > 5:
        warnings.append(
            "More than 5 SFT epochs on this weather dataset often overfits formatting and can corrupt JSON generation."
        )
    if dataset_size > 0 and args.batch_size == 1 and args.gradient_accumulation_steps == 1:
        warnings.append(
            "A global batch size of 1 is noisy for SFT. Consider increasing --gradient-accumulation-steps."
        )
    for warning in warnings:
        print(f"WARNING: {warning}")
    return warnings

def run_post_train_sanity_check(
    agent: WeatherModelAgent,
    model,
    tokenizer,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any] | None:
    eval_path = resolve_path(args.eval_data_path, PROJECT_ROOT)
    if not eval_path.exists():
        return None

    examples = load_weather_prompt_examples(
        dataset_path=eval_path,
        tokenizer=tokenizer,
        max_samples=args.sanity_check_samples,
        shuffle=False,
        seed=args.seed,
    )
    schema = infer_schema([{"reference": example.reference_text} for example in examples])
    model.eval()

    rows: list[dict[str, Any]] = []
    accuracy_values: list[float] = []
    hallucination_values: list[float] = []
    json_valid_values: list[float] = []
    recoverable_values: list[float] = []
    for example in examples:
        prediction, _latency = agent.generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt_text=example.prompt_text,
            max_new_tokens=args.sanity_check_max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=args.sanity_check_repetition_penalty,
            do_sample=False,
        )
        breakdown = score_struct(prediction, example.reference_text, schema)
        rows.append(
            {
                "id": example.sample_id,
                "prediction": prediction,
                "reference": example.reference_text,
                "accuracy": breakdown["accuracy"],
                "hallucinated": breakdown["hallucinated"],
                "json_valid": breakdown.get("json_valid", 0),
                "recoverable_json": breakdown.get("recoverable_json", 0),
                "reasons": breakdown["reasons"],
            }
        )
        accuracy_values.append(float(breakdown["accuracy"]))
        hallucination_values.append(float(breakdown["hallucinated"]))
        json_valid_values.append(float(breakdown.get("json_valid", 0)))
        recoverable_values.append(float(breakdown.get("recoverable_json", 0)))

    summary = {
        "num_examples": len(rows),
        "accuracy": round(sum(accuracy_values) / len(accuracy_values), 4) if accuracy_values else None,
        "hallucination_rate": round(sum(hallucination_values) / len(hallucination_values), 4) if hallucination_values else None,
        "json_valid_rate": round(sum(json_valid_values) / len(json_valid_values), 4) if json_valid_values else None,
        "recoverable_json_rate": round(sum(recoverable_values) / len(recoverable_values), 4) if recoverable_values else None,
        "warning": (
            "Post-train sanity check indicates collapsed formatting. PPO should not start from this adapter."
            if hallucination_values and (sum(hallucination_values) / len(hallucination_values)) > 0.5
            else ""
        ),
    }
    payload = {"summary": summary, "samples": rows}
    save_json(output_dir / "sft_generation_sanity_check.json", payload)
    if summary["warning"]:
        print(f"WARNING: {summary['warning']}")
    return payload


def save_runtime_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    tokenizer_path: str,
    dataset_size: int,
    base_model_load_mode: str,
    compute_dtype: torch.dtype,
    train_result,
    warnings: list[str],
    sanity_check: dict[str, Any] | None,
) -> None:
    runtime_config = {
        "training_algorithm": "sft",
        "adapter_mode": args.mode,
        "base_model_path": args.model_path,
        "tokenizer_path": tokenizer_path or args.model_path,
        "dataset_path": str(resolve_path(args.data_path, PROJECT_ROOT)),
        "dataset_size": dataset_size,
        "base_model_load_mode": base_model_load_mode,
        "compute_dtype": str(compute_dtype).replace("torch.", ""),
        "seed": args.seed,
        "warnings": warnings,
        "sft": {
            "num_train_epochs": args.num_train_epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        },
    }
    summary = {
        "training_algorithm": "sft",
        "mode": args.mode,
        "dataset_size": dataset_size,
        "train_loss": getattr(train_result, "training_loss", None),
        "global_step": getattr(train_result, "global_step", None),
        "metrics": getattr(train_result, "metrics", {}),
        "warnings": warnings,
        "post_train_sanity_check": sanity_check["summary"] if sanity_check else None,
    }
    save_json(output_dir / "adapter_runtime_config.json", runtime_config)
    save_json(output_dir / "sft_training_summary.json", summary)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    agent = WeatherModelAgent(
        base_model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        trust_remote_code=args.trust_remote_code,
        project_root=PROJECT_ROOT,
    )
    tokenizer = agent.load_tokenizer(padding_side="right")
    examples = load_weather_prompt_examples(
        dataset_path=resolve_path(args.data_path, PROJECT_ROOT),
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        shuffle=False,
        seed=args.seed,
    )
    dataset = SupervisedWeatherDataset(examples=examples, tokenizer=tokenizer, max_length=args.max_length)
    warnings = warn_if_risky_configuration(args, len(dataset))

    model, base_model_load_mode, compute_dtype = agent.load_training_backbone(args.mode)
    model = prepare_model_for_training(agent, model, tokenizer, args)

    training_args = build_training_args(args, output_dir, args.mode)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=SupervisedDataCollator(tokenizer),
    )

    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    sanity_check = None
    if not args.skip_post_train_sanity_check:
        sanity_check = run_post_train_sanity_check(
            agent=agent,
            model=model,
            tokenizer=tokenizer,
            args=args,
            output_dir=output_dir,
        )
    save_runtime_metadata(
        output_dir=output_dir,
        args=args,
        tokenizer_path=args.tokenizer_path,
        dataset_size=len(dataset),
        base_model_load_mode=base_model_load_mode,
        compute_dtype=compute_dtype,
        train_result=train_result,
        warnings=warnings,
        sanity_check=sanity_check,
    )

    metrics = {
        "training_loss": getattr(train_result, "training_loss", None),
        "global_step": getattr(train_result, "global_step", None),
        "train_samples": len(dataset),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
