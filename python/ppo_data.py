from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class WeatherPromptExample:
    sample_id: str
    prompt_text: str
    reference_text: str
    messages: list[dict[str, str]]
    metadata: dict[str, Any]


def _load_rows(dataset_path: Path) -> list[dict[str, Any]]:
    text = dataset_path.read_text(encoding="utf-8")
    if dataset_path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
            rows = payload["data"]
        elif isinstance(payload, dict):
            rows = [payload]
        else:
            rows = []
    return [row for row in rows if isinstance(row, dict)]


def _normalize_messages(row: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        normalized: list[dict[str, str]] = []
        reference_text = ""
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if not role or not content:
                continue
            normalized.append({"role": role, "content": content})
        if normalized and normalized[-1]["role"] == "assistant":
            reference_text = normalized[-1]["content"]
            normalized = normalized[:-1]
        return normalized, reference_text

    prompt_value = ""
    for key in ("prompt", "input", "instruction", "question", "text"):
        candidate = row.get(key)
        if candidate is not None and str(candidate).strip():
            prompt_value = str(candidate).strip()
            break

    reference_text = ""
    for key in ("reference", "target", "answer", "output", "response"):
        candidate = row.get(key)
        if candidate is not None and str(candidate).strip():
            reference_text = str(candidate).strip()
            break

    if not prompt_value:
        return [], ""

    return [{"role": "user", "content": prompt_value}], reference_text


def _fallback_chat_prompt(messages: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = message["role"].strip().lower()
        content = message["content"].strip()
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"{role.title()}: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _render_prompt(messages: list[dict[str, str]], tokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return _fallback_chat_prompt(messages)


def load_weather_prompt_examples(
    *,
    dataset_path: str | Path,
    tokenizer,
    max_samples: int = 0,
    shuffle: bool = False,
    seed: int = 42,
) -> list[WeatherPromptExample]:
    resolved_path = Path(dataset_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Weather dataset not found: {resolved_path}")

    rows = _load_rows(resolved_path)
    examples: list[WeatherPromptExample] = []

    for index, row in enumerate(rows):
        messages, reference_text = _normalize_messages(row)
        if not messages or not reference_text:
            continue
        prompt_text = _render_prompt(messages, tokenizer)
        sample_id = str(row.get("id", index))
        examples.append(
            WeatherPromptExample(
                sample_id=sample_id,
                prompt_text=prompt_text,
                reference_text=reference_text,
                messages=messages,
                metadata={
                    key: value
                    for key, value in row.items()
                    if key not in {"messages", "prompt", "input", "instruction", "question", "text", "reference", "target", "answer", "output", "response"}
                },
            )
        )

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(examples)

    if max_samples and max_samples > 0:
        examples = examples[:max_samples]

    if not examples:
        raise ValueError(f"No usable prompt/reference weather examples were found in {resolved_path}.")

    return examples
