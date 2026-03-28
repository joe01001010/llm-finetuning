from __future__ import annotations

from typing import Any

from ppo_config import RewardConfig
from weather_json_metrics import first_json_obj, norm, parse_obj, score_struct, token_f1


def count_reason_prefixes(reasons: list[str], prefix: str) -> int:
    return sum(1 for reason in reasons if reason.startswith(prefix))


def extra_text_length(text: str) -> int:
    stripped = text.strip()
    json_fragment = first_json_obj(stripped)
    if json_fragment is None:
        return len(stripped)
    prefix_index = stripped.find(json_fragment)
    suffix_index = prefix_index + len(json_fragment)
    surrounding = f"{stripped[:prefix_index]}{stripped[suffix_index:]}"
    return len(surrounding.strip())


def compute_weather_reward(
    prediction: str,
    reference: str,
    schema,
    config: RewardConfig,
) -> tuple[float, dict[str, Any]]:
    prediction_object, parse_reasons = parse_obj(prediction)
    reference_object, _ = parse_obj(reference)

    if schema is None:
        score = token_f1(prediction, reference)
        reward = max(config.reward_clip_min, min(config.reward_clip_max, score))
        return reward, {
            "reward": round(reward, 6),
            "accuracy": round(score, 6),
            "hallucinated": 0,
            "a1_score": round(score, 6),
            "exact_json_match": int(norm(prediction) == norm(reference)),
            "reasons": [],
            "json_valid": 0,
            "schema_score": 0.0,
        }

    breakdown = score_struct(prediction, reference, schema)
    reasons = set(breakdown["reasons"])
    required_keys = list(schema[0]) if schema is not None else list(reference_object.keys()) if reference_object else []

    missing_keys: list[str] = []
    extra_keys: list[str] = []
    schema_score = 0.0
    json_valid = int(prediction_object is not None)

    if prediction_object is not None and required_keys:
        missing_keys = [key for key in required_keys if key not in prediction_object]
        extra_keys = [key for key in prediction_object if key not in required_keys]
        matched_keys = len(required_keys) - len(missing_keys)
        schema_score = matched_keys / max(len(required_keys), 1)
    elif prediction_object is not None:
        schema_score = 1.0

    reward = 0.0
    if not prediction.strip():
        reward -= config.empty_response_penalty
    if json_valid:
        reward += config.json_valid_weight
    else:
        reward -= config.malformed_json_penalty

    reward += config.schema_weight * schema_score
    reward += config.field_accuracy_weight * float(breakdown["accuracy"])

    if breakdown["exact_json_match"]:
        reward += config.exact_json_bonus
    if breakdown["hallucinated"]:
        reward -= config.hallucination_penalty

    if required_keys:
        reward -= config.extra_field_penalty * (len(extra_keys) / len(required_keys))
        reward -= config.missing_field_penalty * (len(missing_keys) / len(required_keys))

    if "wrapped_json" in parse_reasons:
        reward -= config.wrapped_json_penalty
    reward -= config.out_of_domain_penalty * count_reason_prefixes(list(reasons), "out_of_domain:")
    reward -= config.non_numeric_penalty * count_reason_prefixes(list(reasons), "non_numeric:")

    verbosity_chars = extra_text_length(prediction)
    if verbosity_chars > config.verbosity_char_threshold:
        reward -= config.verbosity_penalty

    reward = float(max(config.reward_clip_min, min(config.reward_clip_max, reward)))
    reward_breakdown = {
        **breakdown,
        "reward": round(reward, 6),
        "json_valid": json_valid,
        "schema_score": round(schema_score, 6),
        "missing_keys": missing_keys,
        "extra_keys": extra_keys,
        "parse_reasons": sorted(parse_reasons),
        "verbosity_chars": verbosity_chars,
    }
    return reward, reward_breakdown
