#!/usr/bin/env python
"""
Run PPO post-training starting from a saved SFT LoRA or QLoRA adapter.
"""

from __future__ import annotations

import argparse
import json

from weather_agent import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    PPOHyperParameters,
    PPOTrainingConfig,
    WeatherAgent,
    add_reward_config_args,
    reward_config_from_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PPO post-training on top of a saved weather SFT adapter.",
    )
    parser.add_argument("--mode", choices=["lora", "qlora"], default="qlora")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="Path to the saved SFT adapter that PPO should continue training from.",
    )
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument(
        "--output-dir",
        default="",
        help="Defaults to ./adapter-weights/ppo-<mode>.",
    )
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


def main() -> None:
    args = parse_args()
    agent = WeatherAgent()
    summary = agent.train_ppo(
        PPOTrainingConfig(
            mode=args.mode,
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            adapter_path=args.adapter_path,
            dataset_path=args.data_path,
            output_dir=args.output_dir,
            reference_load_mode=args.reference_load_mode,
            trust_remote_code=args.trust_remote_code,
            shuffle=args.shuffle,
            num_train_epochs=args.num_train_epochs,
            rollout_batch_size=args.rollout_batch_size,
            mini_batch_size=args.mini_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            ppo_epochs=args.ppo_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            clip_range_value=args.clip_range_value,
            value_loss_coef=args.value_loss_coef,
            entropy_coef=args.entropy_coef,
            kl_coef=args.kl_coef,
            max_grad_norm=args.max_grad_norm,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_prompt_samples=args.max_prompt_samples,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            seed=args.seed,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            value_head_dropout=args.value_head_dropout,
            reward_config=reward_config_from_args(args),
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
