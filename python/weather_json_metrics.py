from __future__ import annotations

import json
from typing import Any


def norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = norm(prediction).split()
    ref_tokens = norm(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    common = sum(min(count, ref_counts.get(token, 0)) for token, count in pred_counts.items())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def first_json_obj(text: str) -> str | None:
    depth = 0
    start = -1
    in_str = False
    escaped = False

    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : index + 1]
    return None


def parse_obj(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    stripped = text.strip()
    if not stripped:
        return None, ["empty_output"]

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        if isinstance(parsed, dict):
            return parsed, reasons
        return None, ["non_object_json"]
    except json.JSONDecodeError:
        pass

    fragment = first_json_obj(stripped)
    if fragment is None:
        return None, ["invalid_json"]

    try:
        parsed = json.loads(fragment)
    except json.JSONDecodeError:
        return None, ["invalid_json"]

    if not isinstance(parsed, dict):
        return None, ["non_object_json"]
    return parsed, ["wrapped_json"]


def infer_schema(examples: list[dict[str, Any]]) -> tuple[list[str], set[str], dict[str, tuple[float, float]], dict[str, set[str]]] | None:
    reference_objects = []
    for example in examples:
        reference = example.get("reference")
        if reference is None:
            continue
        parsed, _ = parse_obj(reference)
        if parsed is not None:
            reference_objects.append(parsed)

    if not reference_objects:
        return None

    keys = set(reference_objects[0].keys())
    for parsed in reference_objects[1:]:
        keys &= set(parsed.keys())
    if not keys:
        return None

    ordered_keys = [key for key in reference_objects[0].keys() if key in keys]
    numeric_keys: set[str] = set()
    numeric_ranges: dict[str, tuple[float, float]] = {}
    allowed_values: dict[str, set[str]] = {}

    for key in ordered_keys:
        numeric_values = []
        is_numeric = True
        for parsed in reference_objects:
            try:
                numeric_values.append(float(parsed[key]))
            except Exception:
                is_numeric = False
                break
        if is_numeric and numeric_values:
            numeric_keys.add(key)
            numeric_ranges[key] = (min(numeric_values), max(numeric_values))
        else:
            allowed_values[key] = {
                norm(str(parsed.get(key)))
                for parsed in reference_objects
                if parsed.get(key) is not None
            }

    return ordered_keys, numeric_keys, numeric_ranges, allowed_values


def score_struct(prediction: str, reference: str, schema) -> dict[str, Any]:
    keys, numeric_keys, numeric_ranges, allowed_values = schema
    reference_object, _ = parse_obj(reference)
    prediction_object, reasons = parse_obj(prediction)
    reason_set = set(reasons)

    if reference_object is None or prediction_object is None:
        return {
            "accuracy": 0.0,
            "hallucinated": 1,
            "a1_score": 0.0,
            "exact_json_match": 0,
            "reasons": sorted(reason_set),
        }

    missing_keys = [key for key in keys if key not in prediction_object]
    extra_keys = [key for key in prediction_object if key not in keys]
    if missing_keys:
        reason_set.add("missing_keys")
    if extra_keys:
        reason_set.add("extra_keys")

    scores = []
    exact = True
    for key in keys:
        if key not in prediction_object:
            scores.append(0.0)
            exact = False
            continue

        if key in numeric_keys:
            try:
                predicted_value = float(prediction_object[key])
                reference_value = float(reference_object[key])
                low, high = numeric_ranges.get(key, (reference_value, reference_value))
                score = max(0.0, 1.0 - abs(predicted_value - reference_value) / max(high - low, 1.0))
            except Exception:
                score = 0.0
                reason_set.add(f"non_numeric:{key}")
            scores.append(score)
            if score < 1.0:
                exact = False
        else:
            predicted_value = norm(str(prediction_object[key]))
            reference_value = norm(str(reference_object[key]))
            score = 1.0 if predicted_value == reference_value else 0.0
            scores.append(score)
            if score < 1.0:
                exact = False
            if allowed_values.get(key) and predicted_value not in allowed_values[key]:
                reason_set.add(f"out_of_domain:{key}")

    accuracy = sum(scores) / len(scores) if scores else 0.0
    hallucinated = int(bool(reason_set))
    a1_score = accuracy * (1 - hallucinated)
    return {
        "accuracy": round(accuracy, 6),
        "hallucinated": hallucinated,
        "a1_score": round(a1_score, 6),
        "exact_json_match": int(exact and not missing_keys and not extra_keys),
        "reasons": sorted(reason_set),
    }


def reward_from_prediction(prediction: str, reference: str, schema) -> tuple[float, dict[str, Any]]:
    if schema is None:
        score = token_f1(prediction, reference)
        return score, {
            "accuracy": round(score, 6),
            "hallucinated": 0,
            "a1_score": round(score, 6),
            "exact_json_match": int(norm(prediction) == norm(reference)),
            "reasons": [],
        }

    breakdown = score_struct(prediction, reference, schema)
    reward = (
        breakdown["accuracy"]
        - (0.25 if breakdown["hallucinated"] else 0.0)
        + (0.25 if breakdown["exact_json_match"] else 0.0)
    )
    reward = max(-1.0, min(1.25, float(reward)))
    return reward, breakdown
