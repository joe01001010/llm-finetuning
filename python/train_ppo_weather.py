#!/usr/bin/env python
"""
Compatibility entrypoint for PPO weather post-training.

The implementation now lives in `fine_tune_ppo.py` so the PPO stage stays
separate from SFT and evaluation while preserving the original CLI command.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ppo_config import PPOHyperParameters, add_reward_config_args, reward_config_from_args
from ppo_data import WeatherPromptExample, load_weather_prompt_examples
from ppo_memory import RolloutBatch
from ppo_reward import compute_weather_reward
from ppo_utils import (
    build_response_mask,
    compute_gae,
    decode_responses,
    ensure_pad_token,
    entropy_from_logits,
    find_lora_target_modules,
    get_compute_dtype,
    logprobs_from_logits,
    masked_mean,
    masked_whiten,
    model_device,
    resolve_path,
    resolve_pretrained_source,
    save_json,
    set_seed,
    tensor_dict_to_device,
)
from ppo_value_head import CausalLMWithValueHead
from weather_json_metrics import infer_schema


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL_PATH = "/local-containers/Qwen2-7B-Instruct"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "seattle_weather_chat_train.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "adapter-weights"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PPO post-training for weather JSON generation using LoRA/QLoRA adapters.",
    )
    parser.add_argument("--mode", choices=["base", "lora", "qlora"], default="qlora")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--adapter-path", default="", help="Optional SFT or PPO adapter checkpoint to initialize from.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", default="", help="Defaults to ./adapter-weights/ppo-<mode>.")
    parser.add_argument("--reference-load-mode", choices=["auto", "4bit", "16bit", "32bit"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--num-train-epochs", type=int, default=PPOHyperParameters.num_train_epochs)
    parser.add_argument("--rollout-batch-size", type=int, default=PPOHyperParameters.rollout_batch_size)
    parser.add_argument("--mini-batch-size", type=int, default=PPOHyperParameters.mini_batch_size)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=PPOHyperParameters.gradient_accumulation_steps)
    parser.add_argument("--ppo-epochs", type=int, default=PPOHyperParameters.ppo_epochs)
    parser.add_argument("--learning-rate", type=float, default=PPOHyperParameters.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=PPOHyperParameters.weight_decay)
    parser.add_argument("--gamma", type=float, default=PPOHyperParameters.gamma)
    parser.add_argument("--gae-lambda", type=float, default=PPOHyperParameters.gae_lambda)
    parser.add_argument("--clip-range", type=float, default=PPOHyperParameters.clip_range)
    parser.add_argument("--clip-range-value", type=float, default=PPOHyperParameters.clip_range_value)
    parser.add_argument("--value-loss-coef", type=float, default=PPOHyperParameters.value_loss_coef)
    parser.add_argument("--entropy-coef", type=float, default=PPOHyperParameters.entropy_coef)
    parser.add_argument("--kl-coef", type=float, default=PPOHyperParameters.kl_coef)
    parser.add_argument("--max-grad-norm", type=float, default=PPOHyperParameters.max_grad_norm)
    parser.add_argument("--max-prompt-length", type=int, default=PPOHyperParameters.max_prompt_length)
    parser.add_argument("--max-new-tokens", type=int, default=PPOHyperParameters.max_new_tokens)
    parser.add_argument("--temperature", type=float, default=PPOHyperParameters.temperature)
    parser.add_argument("--top-p", type=float, default=PPOHyperParameters.top_p)
    parser.add_argument("--max-prompt-samples", type=int, default=PPOHyperParameters.max_prompt_samples)
    parser.add_argument("--save-steps", type=int, default=PPOHyperParameters.save_steps)
    parser.add_argument("--logging-steps", type=int, default=PPOHyperParameters.logging_steps)
    parser.add_argument("--seed", type=int, default=PPOHyperParameters.seed)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--value-head-dropout", type=float, default=0.1)
    add_reward_config_args(parser)
    return parser.parse_args()


def collate_examples(examples: list[WeatherPromptExample]) -> list[WeatherPromptExample]:
    return examples


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return resolve_path(args.output_dir, PROJECT_ROOT)
    return DEFAULT_OUTPUT_ROOT / f"ppo-{args.mode}"


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
    tokenizer.padding_side = "left"
    return tokenizer


def load_lm_backbone(
    model_path: str,
    load_mode: str,
    trust_remote_code: bool,
    device_map: str | None,
):
    model_source, is_local = resolve_pretrained_source(model_path, kind="model")
    compute_dtype = get_compute_dtype()
    if load_mode == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            quantization_config=quant_config,
            device_map=device_map or "auto",
            trust_remote_code=trust_remote_code,
            local_files_only=is_local,
        )
        return model, "4bit", compute_dtype

    load_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code, "local_files_only": is_local}
    if compute_dtype != torch.float32 and load_mode == "16bit":
        load_kwargs["torch_dtype"] = compute_dtype
    model = AutoModelForCausalLM.from_pretrained(model_source, **load_kwargs)
    if device_map is None and torch.cuda.is_available():
        model.to(torch.device("cuda"))
    resolved_load_mode = "16bit" if load_mode == "16bit" and compute_dtype != torch.float32 else load_mode
    return model, resolved_load_mode, compute_dtype


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


def make_fresh_adapter(model, args: argparse.Namespace):
    target_modules = find_lora_target_modules(model)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    return get_peft_model(model, lora_config)


def determine_policy_load_mode(mode: str) -> str:
    if mode == "qlora":
        return "4bit"
    return "16bit"


def determine_reference_load_mode(mode: str, requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    if mode == "qlora":
        return "4bit"
    return "4bit" if torch.cuda.is_available() else "16bit"


def prepare_policy_model(args: argparse.Namespace) -> tuple[CausalLMWithValueHead, str, torch.dtype, str]:
    load_mode = determine_policy_load_mode(args.mode)
    backbone, resolved_load_mode, compute_dtype = load_lm_backbone(
        model_path=args.model_path,
        load_mode=load_mode,
        trust_remote_code=args.trust_remote_code,
        device_map="auto" if load_mode == "4bit" else None,
    )
    if load_mode == "4bit":
        backbone = prepare_model_for_kbit_training(backbone)

    init_source = "base_model"
    if args.adapter_path:
        adapter_path = resolve_path(args.adapter_path, PROJECT_ROOT)
        backbone = PeftModel.from_pretrained(backbone, str(adapter_path), is_trainable=True)
        init_source = str(adapter_path)
    else:
        backbone = make_fresh_adapter(backbone, args)

    backbone.config.use_cache = False
    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    enable_model_input_grads(backbone)

    policy_model = CausalLMWithValueHead(backbone, value_head_dropout=args.value_head_dropout)
    value_head_path = resolve_path(args.adapter_path, PROJECT_ROOT) / "ppo_value_head.pt" if args.adapter_path else None
    if value_head_path is not None and value_head_path.exists():
        policy_model.load_value_head(value_head_path, strict=False)

    device = model_device(policy_model.pretrained_model)
    policy_model.value_head.to(device)
    return policy_model, resolved_load_mode, compute_dtype, init_source


def prepare_reference_model(args: argparse.Namespace):
    reference_load_mode = determine_reference_load_mode(args.mode, args.reference_load_mode)
    reference_model, resolved_load_mode, _compute_dtype = load_lm_backbone(
        model_path=args.model_path,
        load_mode=reference_load_mode,
        trust_remote_code=args.trust_remote_code,
        device_map="auto" if reference_load_mode == "4bit" else None,
    )
    if args.adapter_path:
        reference_model = PeftModel.from_pretrained(
            reference_model,
            str(resolve_path(args.adapter_path, PROJECT_ROOT)),
            is_trainable=False,
        )
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    return reference_model, resolved_load_mode


def get_response_stats(policy_model, sequence_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_width: int, response_width: int):
    outputs = policy_model(input_ids=sequence_ids, attention_mask=attention_mask)
    token_logprobs = logprobs_from_logits(outputs["logits"], sequence_ids)
    token_values = torch.nan_to_num(outputs["values"][:, :-1].float(), nan=0.0, posinf=0.0, neginf=0.0)
    start_index = max(prompt_width - 1, 0)
    end_index = start_index + response_width
    response_logprobs = torch.nan_to_num(token_logprobs[:, start_index:end_index], nan=0.0, posinf=0.0, neginf=0.0)
    response_values = torch.nan_to_num(token_values[:, start_index:end_index], nan=0.0, posinf=0.0, neginf=0.0)
    response_logits = torch.nan_to_num(outputs["logits"][:, start_index:end_index, :].float(), nan=0.0, posinf=0.0, neginf=0.0)
    return response_logprobs, response_values, response_logits


def get_reference_logprobs(reference_model, sequence_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_width: int, response_width: int) -> torch.Tensor:
    with torch.no_grad():
        outputs = reference_model(input_ids=sequence_ids, attention_mask=attention_mask, return_dict=True)
    token_logprobs = logprobs_from_logits(outputs.logits, sequence_ids)
    start_index = max(prompt_width - 1, 0)
    end_index = start_index + response_width
    return torch.nan_to_num(token_logprobs[:, start_index:end_index], nan=0.0, posinf=0.0, neginf=0.0)


def collect_rollout(
    batch_examples: list[WeatherPromptExample],
    tokenizer,
    policy_model: CausalLMWithValueHead,
    reference_model,
    schema,
    reward_config,
    args: argparse.Namespace,
) -> RolloutBatch:
    device = model_device(policy_model.pretrained_model)
    prompt_texts = [example.prompt_text for example in batch_examples]
    reference_texts = [example.reference_text for example in batch_examples]
    prompt_batch = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_prompt_length,
    )
    prompt_batch = tensor_dict_to_device(prompt_batch, device)
    prompt_width = int(prompt_batch["input_ids"].size(1))

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": args.temperature > 0.0,
        "renormalize_logits": True,
        "remove_invalid_values": True,
    }
    if args.temperature > 0.0:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    policy_model.eval()
    with torch.no_grad():
        sequence_ids = policy_model.generate(**prompt_batch, **generation_kwargs)

    response_ids = sequence_ids[:, prompt_width:]
    response_mask = build_response_mask(response_ids, tokenizer.pad_token_id, tokenizer.eos_token_id)
    response_texts, response_lengths, eos_flags = decode_responses(tokenizer, response_ids, response_mask)
    response_mask_long = response_mask.long()
    sequence_attention_mask = torch.cat([prompt_batch["attention_mask"], response_mask_long], dim=1)

    with torch.no_grad():
        old_logprobs, old_values, _response_logits = get_response_stats(
            policy_model=policy_model,
            sequence_ids=sequence_ids,
            attention_mask=sequence_attention_mask,
            prompt_width=prompt_width,
            response_width=response_ids.size(1),
        )
        reference_device = model_device(reference_model)
        ref_logprobs = get_reference_logprobs(
            reference_model=reference_model,
            sequence_ids=sequence_ids.to(reference_device),
            attention_mask=sequence_attention_mask.to(reference_device),
            prompt_width=prompt_width,
            response_width=response_ids.size(1),
        ).to(device)

    score_rewards: list[float] = []
    reward_breakdowns: list[dict[str, Any]] = []
    accuracy_values: list[float] = []
    a1_values: list[float] = []
    exact_json_values: list[float] = []
    for prediction_text, reference_text in zip(response_texts, reference_texts):
        reward_value, breakdown = compute_weather_reward(
            prediction=prediction_text,
            reference=reference_text,
            schema=schema,
            config=reward_config,
        )
        score_rewards.append(reward_value)
        reward_breakdowns.append(breakdown)
        accuracy_values.append(float(breakdown["accuracy"]))
        a1_values.append(float(breakdown["a1_score"]))
        exact_json_values.append(float(breakdown["exact_json_match"]))

    score_reward_tensor = torch.tensor(score_rewards, device=device, dtype=torch.float32)
    rewards = -args.kl_coef * torch.nan_to_num(old_logprobs - ref_logprobs, nan=0.0, posinf=0.0, neginf=0.0)
    rewards = rewards * response_mask
    last_token_indices = response_mask.long().sum(dim=1) - 1
    for row_index, last_index in enumerate(last_token_indices.tolist()):
        if last_index >= 0:
            rewards[row_index, last_index] += score_reward_tensor[row_index]

    advantages, returns = compute_gae(
        rewards=rewards,
        values=old_values * response_mask,
        mask=response_mask,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )
    advantages = masked_whiten(advantages, response_mask)
    kl_divergence = masked_mean(
        torch.nan_to_num(old_logprobs - ref_logprobs, nan=0.0, posinf=0.0, neginf=0.0),
        response_mask,
        dim=1,
    )

    return RolloutBatch(
        prompt_width=prompt_width,
        sample_ids=[example.sample_id for example in batch_examples],
        prompt_texts=prompt_texts,
        reference_texts=reference_texts,
        response_texts=response_texts,
        reward_breakdowns=reward_breakdowns,
        prompt_input_ids=prompt_batch["input_ids"],
        prompt_attention_mask=prompt_batch["attention_mask"],
        sequence_ids=sequence_ids,
        sequence_attention_mask=sequence_attention_mask,
        response_ids=response_ids,
        response_mask=response_mask,
        old_logprobs=old_logprobs * response_mask,
        ref_logprobs=ref_logprobs * response_mask,
        old_values=old_values * response_mask,
        rewards=rewards,
        advantages=advantages,
        returns=returns,
        score_rewards=score_reward_tensor,
        kl_divergence=kl_divergence,
        response_lengths=response_lengths,
        eos_flags=eos_flags,
    )


def optimize_policy(
    policy_model: CausalLMWithValueHead,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    args: argparse.Namespace,
) -> dict[str, float]:
    policy_model.train()
    stats: defaultdict[str, list[float]] = defaultdict(list)
    step_count = 0
    trainable_parameters = [parameter for parameter in policy_model.parameters() if parameter.requires_grad]
    optimizer.zero_grad(set_to_none=True)

    for _ppo_epoch in range(args.ppo_epochs):
        for minibatch in rollout.iter_minibatches(args.mini_batch_size, shuffle=True):
            new_logprobs, new_values, response_logits = get_response_stats(
                policy_model=policy_model,
                sequence_ids=minibatch["sequence_ids"],
                attention_mask=minibatch["sequence_attention_mask"],
                prompt_width=rollout.prompt_width,
                response_width=rollout.response_length,
            )
            response_mask = minibatch["response_mask"]
            new_logprobs = new_logprobs * response_mask
            new_values = new_values * response_mask

            log_ratio = torch.clamp(
                torch.nan_to_num(new_logprobs - minibatch["old_logprobs"], nan=0.0, posinf=0.0, neginf=0.0),
                min=-20.0,
                max=20.0,
            )
            ratio = torch.exp(log_ratio)
            unclipped = ratio * minibatch["advantages"]
            clipped = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * minibatch["advantages"]
            policy_loss = -masked_mean(torch.minimum(unclipped, clipped), response_mask)

            value_delta = new_values - minibatch["old_values"]
            clipped_values = minibatch["old_values"] + torch.clamp(value_delta, -args.clip_range_value, args.clip_range_value)
            value_loss_unclipped = (new_values - minibatch["returns"]) ** 2
            value_loss_clipped = (clipped_values - minibatch["returns"]) ** 2
            value_loss = 0.5 * masked_mean(torch.maximum(value_loss_unclipped, value_loss_clipped), response_mask)

            response_entropy = entropy_from_logits(response_logits)
            entropy_bonus = masked_mean(response_entropy, response_mask)

            loss = policy_loss + args.value_loss_coef * value_loss - args.entropy_coef * entropy_bonus
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                stats["skipped_non_finite_loss"].append(1.0)
                continue
            (loss / args.gradient_accumulation_steps).backward()

            step_count += 1
            if step_count % args.gradient_accumulation_steps == 0:
                has_non_finite_grad = any(
                    parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                    for parameter in trainable_parameters
                )
                if has_non_finite_grad:
                    optimizer.zero_grad(set_to_none=True)
                    stats["skipped_non_finite_grad"].append(1.0)
                    continue
                clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                stats["skipped_non_finite_grad"].append(0.0)

            approx_kl = masked_mean(
                torch.nan_to_num(minibatch["old_logprobs"] - new_logprobs, nan=0.0, posinf=0.0, neginf=0.0),
                response_mask,
            )
            clip_fraction = masked_mean((torch.abs(ratio - 1.0) > args.clip_range).to(new_logprobs.dtype), response_mask)

            stats["loss"].append(float(loss.detach().item()))
            stats["policy_loss"].append(float(policy_loss.detach().item()))
            stats["value_loss"].append(float(value_loss.detach().item()))
            stats["entropy"].append(float(entropy_bonus.detach().item()))
            stats["approx_kl"].append(float(approx_kl.detach().item()))
            stats["clip_fraction"].append(float(clip_fraction.detach().item()))
            stats["skipped_non_finite_loss"].append(0.0)

    if step_count % args.gradient_accumulation_steps != 0:
        has_non_finite_grad = any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in trainable_parameters
        )
        if not has_non_finite_grad:
            clip_grad_norm_(trainable_parameters, args.max_grad_norm)
            optimizer.step()
        else:
            stats["skipped_non_finite_grad"].append(1.0)
        optimizer.zero_grad(set_to_none=True)

    return {
        key: round(sum(values) / len(values), 6)
        for key, values in stats.items()
        if values
    }


def save_ppo_artifacts(
    output_dir: Path,
    policy_model: CausalLMWithValueHead,
    tokenizer,
    runtime_config: dict[str, Any],
    reward_config,
    metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_model.pretrained_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    policy_model.save_value_head(output_dir / "ppo_value_head.pt")
    save_json(output_dir / "adapter_runtime_config.json", runtime_config)
    save_json(output_dir / "ppo_runtime_config.json", runtime_config)
    save_json(output_dir / "ppo_reward_config.json", reward_config.to_dict())
    save_json(output_dir / "ppo_training_summary.json", metrics)


def summarize_rollout(rollout: RolloutBatch) -> dict[str, float]:
    accuracy_values = [float(item["accuracy"]) for item in rollout.reward_breakdowns]
    a1_values = [float(item["a1_score"]) for item in rollout.reward_breakdowns]
    exact_json_values = [float(item["exact_json_match"]) for item in rollout.reward_breakdowns]
    hallucination_values = [float(item["hallucinated"]) for item in rollout.reward_breakdowns]
    return {
        "reward": round(float(rollout.score_rewards.mean().item()), 6),
        "accuracy": round(sum(accuracy_values) / len(accuracy_values), 6),
        "a1_score": round(sum(a1_values) / len(a1_values), 6),
        "exact_json_match": round(sum(exact_json_values) / len(exact_json_values), 6),
        "hallucination_rate": round(sum(hallucination_values) / len(hallucination_values), 6),
        "kl_divergence": round(float(rollout.kl_divergence.mean().item()), 6),
        "mean_response_length": round(float(rollout.response_lengths.float().mean().item()), 6),
        "eos_rate": round(float(rollout.eos_flags.float().mean().item()), 6),
    }


def build_runtime_config(
    args: argparse.Namespace,
    init_source: str,
    base_model_load_mode: str,
    reference_load_mode: str,
    compute_dtype: torch.dtype,
    reward_config,
) -> dict[str, Any]:
    return {
        "training_algorithm": "ppo",
        "mode": args.mode,
        "base_model_path": args.model_path,
        "tokenizer_path": args.tokenizer_path or args.model_path,
        "adapter_init_path": args.adapter_path or "",
        "initialization_source": init_source,
        "base_model_load_mode": base_model_load_mode,
        "reference_load_mode": reference_load_mode,
        "compute_dtype": str(compute_dtype).replace("torch.", ""),
        "dataset_path": str(resolve_path(args.data_path, PROJECT_ROOT)),
        "seed": args.seed,
        "ppo": {
            "num_train_epochs": args.num_train_epochs,
            "rollout_batch_size": args.rollout_batch_size,
            "mini_batch_size": args.mini_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ppo_epochs": args.ppo_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
            "clip_range_value": args.clip_range_value,
            "value_loss_coef": args.value_loss_coef,
            "entropy_coef": args.entropy_coef,
            "kl_coef": args.kl_coef,
            "max_grad_norm": args.max_grad_norm,
            "max_prompt_length": args.max_prompt_length,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_prompt_samples": args.max_prompt_samples,
            "save_steps": args.save_steps,
            "logging_steps": args.logging_steps,
        },
        "reward": reward_config.to_dict(),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    reward_config = reward_config_from_args(args)
    tokenizer = load_tokenizer(args.tokenizer_path, args.model_path, args.trust_remote_code)
    examples = load_weather_prompt_examples(
        dataset_path=resolve_path(args.data_path, PROJECT_ROOT),
        tokenizer=tokenizer,
        max_samples=args.max_prompt_samples,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    schema = infer_schema([{"reference": example.reference_text} for example in examples])

    policy_model, base_model_load_mode, compute_dtype, init_source = prepare_policy_model(args)
    reference_model, reference_load_mode = prepare_reference_model(args)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy_model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    runtime_config = build_runtime_config(
        args=args,
        init_source=init_source,
        base_model_load_mode=base_model_load_mode,
        reference_load_mode=reference_load_mode,
        compute_dtype=compute_dtype,
        reward_config=reward_config,
    )

    dataloader = DataLoader(
        examples,
        batch_size=args.rollout_batch_size,
        shuffle=args.shuffle,
        collate_fn=collate_examples,
    )

    global_step = 0
    history: defaultdict[str, list[float]] = defaultdict(list)
    total_steps = len(dataloader) * max(args.num_train_epochs, 1)
    progress = tqdm(total=total_steps, desc="PPO weather training", dynamic_ncols=True)

    for epoch_index in range(args.num_train_epochs):
        for batch_examples in dataloader:
            global_step += 1
            rollout = collect_rollout(
                batch_examples=batch_examples,
                tokenizer=tokenizer,
                policy_model=policy_model,
                reference_model=reference_model,
                schema=schema,
                reward_config=reward_config,
                args=args,
            )
            rollout_stats = summarize_rollout(rollout)
            optimize_stats = optimize_policy(
                policy_model=policy_model,
                optimizer=optimizer,
                rollout=rollout,
                args=args,
            )

            for key, value in {**rollout_stats, **optimize_stats}.items():
                history[key].append(value)

            progress.update(1)
            progress.set_postfix(
                reward=f"{rollout_stats['reward']:.3f}",
                acc=f"{rollout_stats['accuracy']:.3f}",
                a1=f"{rollout_stats['a1_score']:.3f}",
                kl=f"{rollout_stats['kl_divergence']:.3f}",
            )

            if global_step % args.logging_steps == 0:
                progress.write(
                    f"step={global_step} "
                    f"reward={rollout_stats['reward']:.4f} "
                    f"accuracy={rollout_stats['accuracy']:.4f} "
                    f"a1={rollout_stats['a1_score']:.4f} "
                    f"kl={rollout_stats['kl_divergence']:.4f}"
                )

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                checkpoint_metrics = {
                    "global_step": global_step,
                    "epoch": epoch_index + 1,
                    "history_means": {
                        key: round(sum(values) / len(values), 6)
                        for key, values in history.items()
                        if values
                    },
                }
                save_ppo_artifacts(
                    output_dir=checkpoint_dir,
                    policy_model=policy_model,
                    tokenizer=tokenizer,
                    runtime_config=runtime_config,
                    reward_config=reward_config,
                    metrics=checkpoint_metrics,
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    progress.close()

    final_metrics = {
        "training_algorithm": "ppo",
        "global_step": global_step,
        "num_examples": len(examples),
        "history_means": {
            key: round(sum(values) / len(values), 6)
            for key, values in history.items()
            if values
        },
    }
    save_ppo_artifacts(
        output_dir=output_dir,
        policy_model=policy_model,
        tokenizer=tokenizer,
        runtime_config=runtime_config,
        reward_config=reward_config,
        metrics=final_metrics,
    )
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
