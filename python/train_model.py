"""
train_model.py
PPO-based adapter training for the Seattle weather chat dataset.
"""
#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model

from weather_json_metrics import infer_schema, reward_from_prediction


MODEL_PATH = "/local-containers/Qwen2-7B-Instruct"
DATA_PATH = f"{Path(__file__).resolve().parent.parent}/data/seattle_weather_chat_train.jsonl"
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "adapter-weights"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train weather adapters with PPO using prompt-only rollouts and structured rewards."
    )
    parser.add_argument("--mode", choices=["lora", "qlora"], default="qlora")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument(
        "--output-dir",
        default="",
        help="Adapter output directory. Defaults to ./adapter-weights/<mode>.",
    )
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mini-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--init-kl-coef", type=float, default=0.2)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_16bit_dtype():
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def resolve_output_dir(mode: str, output_dir: str) -> Path:
    if output_dir:
        return Path(output_dir)
    return OUTPUT_ROOT / mode


def load_full_precision_model(model_path: str, dtype):
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            low_cpu_mem_usage=False,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )


def create_backbone(mode: str, model_path: str, compute_dtype):
    if mode == "qlora":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        backbone = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
        )
        backbone = prepare_model_for_kbit_training(backbone)
        base_model_load_mode = "4bit"
    else:
        backbone = load_full_precision_model(model_path, compute_dtype)
        base_model_load_mode = "16bit" if compute_dtype != torch.float32 else "32bit"

    adapter_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    backbone = get_peft_model(backbone, adapter_cfg)
    backbone.config.use_cache = False
    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()
    backbone.print_trainable_parameters()
    return backbone, base_model_load_mode


def create_policy_and_reference(mode: str, model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = get_16bit_dtype()
    backbone, base_model_load_mode = create_backbone(mode, model_path, compute_dtype)
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(backbone)
    ref_model = create_reference_model(policy_model)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("GPU is unavailable")
        sys.exit(67)

    return policy_model, ref_model, tokenizer, base_model_load_mode, compute_dtype


def prepare_dataset(data_path: str, tokenizer, max_prompt_length: int):
    dataset = load_dataset("json", data_files=data_path, split="train")

    def to_prompt_record(example):
        messages = example.get("messages")
        if not isinstance(messages, list) or not messages:
            return {"prompt": "", "reference": "", "input_ids": []}

        clean = []
        for message in messages:
            if isinstance(message, dict) and message.get("role") is not None and message.get("content") is not None:
                clean.append({
                    "role": str(message["role"]),
                    "content": str(message["content"]),
                })
        if not clean:
            return {"prompt": "", "reference": "", "input_ids": []}

        reference = ""
        if clean[-1]["role"] == "assistant":
            reference = clean[-1]["content"]
            clean = clean[:-1]
        if not clean or not reference:
            return {"prompt": "", "reference": "", "input_ids": []}

        prompt_text = tokenizer.apply_chat_template(
            clean,
            tokenize=False,
            add_generation_prompt=True,
        )
        tokenized = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_prompt_length,
            padding=False,
        )
        return {
            "prompt": prompt_text,
            "reference": reference,
            "input_ids": tokenized["input_ids"],
        }

    dataset = dataset.map(to_prompt_record, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda example: bool(example["prompt"]) and bool(example["reference"]) and bool(example["input_ids"]))
    schema = infer_schema([{"reference": reference} for reference in dataset["reference"]])
    return dataset, schema


def collate_prompt_batch(features):
    return {
        "input_ids": [torch.tensor(feature["input_ids"], dtype=torch.long) for feature in features],
        "prompt": [feature["prompt"] for feature in features],
        "reference": [feature["reference"] for feature in features],
    }


def get_policy_device(model) -> torch.device:
    for parameter in model.parameters():
        return parameter.device
    raise RuntimeError("Unable to determine model device.")


def build_generation_kwargs(tokenizer, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


def write_runtime_config(output_dir: Path, mode: str, base_model_load_mode: str, compute_dtype, args: argparse.Namespace) -> None:
    runtime_config = {
        "training_algorithm": "ppo",
        "adapter_mode": mode,
        "base_model_load_mode": base_model_load_mode,
        "compute_dtype": str(compute_dtype).replace("torch.", ""),
        "reward_function": "structured_json_accuracy_with_hallucination_penalty",
        "ppo": {
            "batch_size": args.batch_size,
            "mini_batch_size": args.mini_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "ppo_epochs": args.ppo_epochs,
            "init_kl_coef": args.init_kl_coef,
            "max_prompt_length": args.max_prompt_length,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    }
    (output_dir / "adapter_runtime_config.json").write_text(
        json.dumps(runtime_config, indent=2),
        encoding="utf-8",
    )


def save_policy_artifacts(policy_model, tokenizer, output_dir: Path, mode: str, base_model_load_mode: str, compute_dtype, args: argparse.Namespace, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_model.pretrained_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(policy_model.v_head.state_dict(), output_dir / "ppo_value_head.pt")
    write_runtime_config(output_dir, mode, base_model_load_mode, compute_dtype, args)
    (output_dir / "ppo_training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def train_with_ppo(
    policy_model,
    ref_model,
    tokenizer,
    dataset,
    schema,
    output_dir: Path,
    mode: str,
    base_model_load_mode: str,
    compute_dtype,
    args: argparse.Namespace,
):
    ppo_config = PPOConfig(
        model_name=args.model_path,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ppo_epochs=args.ppo_epochs,
        init_kl_coef=args.init_kl_coef,
        seed=args.seed,
        log_with=None,
    )
    ppo_trainer = PPOTrainer(
        ppo_config,
        policy_model,
        ref_model,
        tokenizer,
        dataset=dataset,
        data_collator=collate_prompt_batch,
    )

    device = get_policy_device(policy_model)
    generation_kwargs = build_generation_kwargs(tokenizer, args)
    reward_history: list[float] = []
    accuracy_history: list[float] = []
    a1_history: list[float] = []
    exact_json_history: list[float] = []
    reward_reason_counts: dict[str, int] = defaultdict(int)
    ppo_stats: dict[str, list[float]] = defaultdict(list)
    global_step = 0

    for epoch in range(args.num_train_epochs):
        print(f"Starting PPO epoch {epoch + 1}/{args.num_train_epochs}")
        for batch in ppo_trainer.dataloader:
            global_step += 1
            query_tensors = [query_tensor.to(device) for query_tensor in batch["input_ids"]]
            response_tensors = ppo_trainer.generate(
                query_tensors,
                return_prompt=False,
                **generation_kwargs,
            )
            if isinstance(response_tensors, torch.Tensor):
                response_tensors = [response for response in response_tensors]

            response_texts = [
                tokenizer.decode(response_tensor.squeeze(), skip_special_tokens=True).strip()
                for response_tensor in response_tensors
            ]

            rewards = []
            batch_rewards = []
            batch_accuracies = []
            batch_a1_scores = []
            for response_text, reference_text in zip(response_texts, batch["reference"]):
                reward_value, breakdown = reward_from_prediction(response_text, reference_text, schema)
                rewards.append(torch.tensor(reward_value, device=device))
                batch_rewards.append(reward_value)
                batch_accuracies.append(float(breakdown["accuracy"]))
                batch_a1_scores.append(float(breakdown["a1_score"]))
                reward_history.append(reward_value)
                accuracy_history.append(float(breakdown["accuracy"]))
                a1_history.append(float(breakdown["a1_score"]))
                exact_json_history.append(float(breakdown["exact_json_match"]))
                for reason in breakdown["reasons"]:
                    reward_reason_counts[reason] += 1

            step_stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            for key, value in step_stats.items():
                if isinstance(value, (int, float)):
                    ppo_stats[key].append(float(value))

            if global_step % args.logging_steps == 0:
                print(
                    f"PPO step {global_step}: "
                    f"avg_reward={sum(batch_rewards) / len(batch_rewards):.4f}, "
                    f"avg_accuracy={sum(batch_accuracies) / len(batch_accuracies):.4f}, "
                    f"avg_a1={sum(batch_a1_scores) / len(batch_a1_scores):.4f}"
                )

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                checkpoint_summary = {
                    "global_step": global_step,
                    "avg_reward": mean_or_none(reward_history),
                    "avg_accuracy": mean_or_none(accuracy_history),
                    "avg_a1_score": mean_or_none(a1_history),
                    "avg_exact_json_match": mean_or_none(exact_json_history),
                    "reason_counts": dict(sorted(reward_reason_counts.items())),
                }
                save_policy_artifacts(
                    policy_model,
                    tokenizer,
                    checkpoint_dir,
                    mode,
                    base_model_load_mode,
                    compute_dtype,
                    args,
                    checkpoint_summary,
                )

    summary = {
        "training_algorithm": "ppo",
        "mode": mode,
        "base_model_load_mode": base_model_load_mode,
        "num_train_epochs": args.num_train_epochs,
        "ppo_steps": global_step,
        "num_rollouts": len(reward_history),
        "avg_reward": mean_or_none(reward_history),
        "avg_accuracy": mean_or_none(accuracy_history),
        "avg_a1_score": mean_or_none(a1_history),
        "avg_exact_json_match": mean_or_none(exact_json_history),
        "reason_counts": dict(sorted(reward_reason_counts.items())),
        "ppo_stats": {key: mean_or_none(values) for key, values in sorted(ppo_stats.items())},
    }
    save_policy_artifacts(
        policy_model,
        tokenizer,
        output_dir,
        mode,
        base_model_load_mode,
        compute_dtype,
        args,
        summary,
    )
    print(json.dumps(summary, indent=2))


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = resolve_output_dir(args.mode, args.output_dir)

    try:
        policy_model, ref_model, tokenizer, base_model_load_mode, compute_dtype = create_policy_and_reference(
            mode=args.mode,
            model_path=args.model_path,
        )
        dataset, schema = prepare_dataset(
            data_path=args.data_path,
            tokenizer=tokenizer,
            max_prompt_length=args.max_prompt_length,
        )
        if len(dataset) == 0:
            raise SystemExit("No usable prompt/reference examples found for PPO training.")
        train_with_ppo(
            policy_model=policy_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=dataset,
            schema=schema,
            output_dir=output_dir,
            mode=args.mode,
            base_model_load_mode=base_model_load_mode,
            compute_dtype=compute_dtype,
            args=args,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise SystemExit(
                "CUDA OOM during PPO training.\n"
                "Try lowering --batch-size, --mini-batch-size, or --max-new-tokens."
            ) from exc
        raise


if __name__ == '__main__':
    main()
