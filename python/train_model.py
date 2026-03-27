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
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

from ppo_data import WeatherPromptExample, load_weather_prompt_examples
from ppo_utils import (
    ensure_pad_token,
    find_lora_target_modules,
    resolve_path,
    resolve_pretrained_source,
    save_json,
    set_seed,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL_PATH = "/local-containers/Qwen2-7B-Instruct"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "seattle_weather_chat_train.jsonl"
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
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-samples", type=int, default=0)
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


def get_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_base_model(model_path: str, mode: str, trust_remote_code: bool):
    model_source, is_local = resolve_pretrained_source(model_path, kind="model")
    compute_dtype = get_compute_dtype()
    if mode == "qlora":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            local_files_only=is_local,
        )
        model = prepare_model_for_kbit_training(model)
        base_model_load_mode = "4bit"
    else:
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": is_local,
        }
        if compute_dtype != torch.float32:
            load_kwargs["torch_dtype"] = compute_dtype
        model = AutoModelForCausalLM.from_pretrained(model_source, **load_kwargs)
        base_model_load_mode = "16bit" if compute_dtype != torch.float32 else "32bit"

    model.config.use_cache = False
    return model, base_model_load_mode, compute_dtype


def enable_model_input_grads(model) -> None:
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return

    embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if embeddings is None:
        return

    def hook(_module, _inputs, output):
        output.requires_grad_(True)

    embeddings.register_forward_hook(hook)


def prepare_model_for_training(model, tokenizer, args: argparse.Namespace):
    target_modules = find_lora_target_modules(model)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    enable_model_input_grads(model)
    if tokenizer.pad_token_id is not None and hasattr(model.config, "pad_token_id"):
        model.config.pad_token_id = tokenizer.pad_token_id
    return model


def load_tokenizer(tokenizer_path: str, fallback_model_path: str, trust_remote_code: bool):
    load_path = tokenizer_path or fallback_model_path
    tokenizer_source, is_local = resolve_pretrained_source(load_path, kind="tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=True,
        trust_remote_code=trust_remote_code,
        local_files_only=is_local,
    )
    ensure_pad_token(tokenizer)
    tokenizer.padding_side = "right"
    return tokenizer


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


def save_runtime_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    tokenizer_path: str,
    dataset_size: int,
    base_model_load_mode: str,
    compute_dtype: torch.dtype,
    train_result,
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
    }
    save_json(output_dir / "adapter_runtime_config.json", runtime_config)
    save_json(output_dir / "sft_training_summary.json", summary)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer_path, args.model_path, args.trust_remote_code)
    examples = load_weather_prompt_examples(
        dataset_path=resolve_path(args.data_path, PROJECT_ROOT),
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        shuffle=False,
        seed=args.seed,
    )
    dataset = SupervisedWeatherDataset(examples=examples, tokenizer=tokenizer, max_length=args.max_length)

    model, base_model_load_mode, compute_dtype = load_base_model(
        model_path=args.model_path,
        mode=args.mode,
        trust_remote_code=args.trust_remote_code,
    )
    model = prepare_model_for_training(model, tokenizer, args)

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
    save_runtime_metadata(
        output_dir=output_dir,
        args=args,
        tokenizer_path=args.tokenizer_path,
        dataset_size=len(dataset),
        base_model_load_mode=base_model_load_mode,
        compute_dtype=compute_dtype,
        train_result=train_result,
    )

    metrics = {
        "training_loss": getattr(train_result, "training_loss", None),
        "global_step": getattr(train_result, "global_step", None),
        "train_samples": len(dataset),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
