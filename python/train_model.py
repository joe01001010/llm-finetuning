#!/usr/bin/env python
"""
Train LoRA or QLoRA adapters with supervised fine-tuning.
"""

from __future__ import annotations

import argparse
import json

from weather_agent import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH, SFTTrainingConfig, WeatherAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train weather adapters with SFT using either LoRA or QLoRA.",
    )
    parser.add_argument("--mode", choices=["lora", "qlora"], default="qlora")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument(
        "--output-dir",
        default="",
        help="Defaults to ./adapter-weights/sft-<mode>.",
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


def main() -> None:
    args = parse_args()
    agent = WeatherAgent()
    summary = agent.train_sft(
        SFTTrainingConfig(
            mode=args.mode,
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            dataset_path=args.data_path,
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            save_total_limit=args.save_total_limit,
            max_length=args.max_length,
            max_samples=args.max_samples,
            seed=args.seed,
            trust_remote_code=args.trust_remote_code,
            gradient_checkpointing=args.gradient_checkpointing,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
