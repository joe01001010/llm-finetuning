from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RewardConfig:
    json_valid_weight: float = 0.2
    recoverable_json_weight: float = 0.08
    schema_weight: float = 0.15
    field_accuracy_weight: float = 0.55
    exact_json_bonus: float = 0.15
    hallucination_penalty: float = 0.2
    malformed_json_penalty: float = 0.4
    empty_response_penalty: float = 0.5
    verbosity_penalty: float = 0.1
    extra_field_penalty: float = 0.12
    missing_field_penalty: float = 0.12
    wrapped_json_penalty: float = 0.03
    out_of_domain_penalty: float = 0.05
    non_numeric_penalty: float = 0.05
    repeated_symbol_penalty: float = 0.2
    forbidden_character_penalty: float = 0.15
    reward_clip_min: float = -1.5
    reward_clip_max: float = 1.5
    verbosity_char_threshold: int = 12

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass
class PPOHyperParameters:
    num_train_epochs: int = 1
    rollout_batch_size: int = 1
    mini_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    ppo_epochs: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_value: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.05
    max_grad_norm: float = 1.0
    max_prompt_length: int = 512
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    max_prompt_samples: int = 0
    save_steps: int = 100
    logging_steps: int = 10
    seed: int = 42

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def add_reward_config_args(parser) -> None:
    parser.add_argument("--json-valid-weight", type=float, default=RewardConfig.json_valid_weight)
    parser.add_argument("--recoverable-json-weight", type=float, default=RewardConfig.recoverable_json_weight)
    parser.add_argument("--schema-weight", type=float, default=RewardConfig.schema_weight)
    parser.add_argument("--field-accuracy-weight", type=float, default=RewardConfig.field_accuracy_weight)
    parser.add_argument("--exact-json-bonus", type=float, default=RewardConfig.exact_json_bonus)
    parser.add_argument("--hallucination-penalty", type=float, default=RewardConfig.hallucination_penalty)
    parser.add_argument("--malformed-json-penalty", type=float, default=RewardConfig.malformed_json_penalty)
    parser.add_argument("--empty-response-penalty", type=float, default=RewardConfig.empty_response_penalty)
    parser.add_argument("--verbosity-penalty", type=float, default=RewardConfig.verbosity_penalty)
    parser.add_argument("--extra-field-penalty", type=float, default=RewardConfig.extra_field_penalty)
    parser.add_argument("--missing-field-penalty", type=float, default=RewardConfig.missing_field_penalty)
    parser.add_argument("--wrapped-json-penalty", type=float, default=RewardConfig.wrapped_json_penalty)
    parser.add_argument("--out-of-domain-penalty", type=float, default=RewardConfig.out_of_domain_penalty)
    parser.add_argument("--non-numeric-penalty", type=float, default=RewardConfig.non_numeric_penalty)
    parser.add_argument("--repeated-symbol-penalty", type=float, default=RewardConfig.repeated_symbol_penalty)
    parser.add_argument("--forbidden-character-penalty", type=float, default=RewardConfig.forbidden_character_penalty)
    parser.add_argument("--reward-clip-min", type=float, default=RewardConfig.reward_clip_min)
    parser.add_argument("--reward-clip-max", type=float, default=RewardConfig.reward_clip_max)
    parser.add_argument("--verbosity-char-threshold", type=int, default=RewardConfig.verbosity_char_threshold)


def reward_config_from_args(args) -> RewardConfig:
    return RewardConfig(
        json_valid_weight=args.json_valid_weight,
        recoverable_json_weight=args.recoverable_json_weight,
        schema_weight=args.schema_weight,
        field_accuracy_weight=args.field_accuracy_weight,
        exact_json_bonus=args.exact_json_bonus,
        hallucination_penalty=args.hallucination_penalty,
        malformed_json_penalty=args.malformed_json_penalty,
        empty_response_penalty=args.empty_response_penalty,
        verbosity_penalty=args.verbosity_penalty,
        extra_field_penalty=args.extra_field_penalty,
        missing_field_penalty=args.missing_field_penalty,
        wrapped_json_penalty=args.wrapped_json_penalty,
        out_of_domain_penalty=args.out_of_domain_penalty,
        non_numeric_penalty=args.non_numeric_penalty,
        repeated_symbol_penalty=args.repeated_symbol_penalty,
        forbidden_character_penalty=args.forbidden_character_penalty,
        reward_clip_min=args.reward_clip_min,
        reward_clip_max=args.reward_clip_max,
        verbosity_char_threshold=args.verbosity_char_threshold,
    )
