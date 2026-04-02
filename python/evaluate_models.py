#!/usr/bin/env python
"""
Evaluate the base model against SFT and PPO weather adapter variants.
"""

from __future__ import annotations

import argparse
import json

from weather_agent import (
    DEFAULT_EVAL_DATA_PATH,
    DEFAULT_MODEL_PATH,
    EvaluationConfig,
    WeatherAgent,
    add_reward_config_args,
    reward_config_from_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate base, SFT, and PPO weather models")
    parser.add_argument("--base-model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--dataset", default=str(DEFAULT_EVAL_DATA_PATH))
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--output-predictions", default="")
    parser.add_argument("--sft-lora-adapter", default="")
    parser.add_argument("--sft-qlora-adapter", default="")
    parser.add_argument("--ppo-lora-adapter", default="")
    parser.add_argument("--ppo-qlora-adapter", default="")
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


def main() -> None:
    args = parse_args()
    agent = WeatherAgent()
    report = agent.evaluate_variants(
        EvaluationConfig(
            base_model_path=args.base_model,
            tokenizer_path=args.tokenizer_path,
            dataset_path=args.dataset,
            output_report=args.output_report,
            output_predictions=args.output_predictions,
            sft_lora_adapter=args.sft_lora_adapter,
            sft_qlora_adapter=args.sft_qlora_adapter,
            ppo_lora_adapter=args.ppo_lora_adapter,
            ppo_qlora_adapter=args.ppo_qlora_adapter,
            max_samples=args.max_samples,
            shuffle=args.shuffle,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            base_load_mode=args.base_load_mode,
            trust_remote_code=args.trust_remote_code,
            reward_config=reward_config_from_args(args),
        )
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
