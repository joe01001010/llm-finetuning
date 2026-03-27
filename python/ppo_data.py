from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WeatherPromptExample:
    sample_id: int
    prompt_text: str
    reference_text: str
    prompt_messages: list[dict[str, str]]
    metadata: dict[str, Any] = field(default_factory=dict)


def load_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record at line {line_number} in {path}: {exc}") from exc
            if isinstance(item, dict):
                records.append(item)
        return records

    payload = json.loads(text)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return [item for item in payload["data"] if isinstance(item, dict)]
        return [payload]
    return []


def clean_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        return []
    cleaned: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role is None or content is None:
            continue
        cleaned.append({"role": str(role), "content": str(content)})
    return cleaned


def apply_prompt_template(tokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    parts = [f"{message['role']}: {message['content']}" for message in messages]
    return "\n".join(parts) + "\nassistant:"


def record_to_example(record: dict[str, Any], sample_id: int, tokenizer) -> WeatherPromptExample | None:
    messages = clean_messages(record.get("messages"))
    if messages and messages[-1]["role"] == "assistant":
        prompt_messages = messages[:-1]
        reference_text = messages[-1]["content"]
        if not prompt_messages or not reference_text:
            return None
        metadata = {key: value for key, value in record.items() if key != "messages"}
        return WeatherPromptExample(
            sample_id=sample_id,
            prompt_text=apply_prompt_template(tokenizer, prompt_messages),
            reference_text=reference_text,
            prompt_messages=prompt_messages,
            metadata=metadata,
        )

    prompt = None
    for prompt_key in ("prompt", "input", "instruction", "question", "text"):
        if record.get(prompt_key) is not None and str(record[prompt_key]).strip():
            prompt = str(record[prompt_key])
            break
    if prompt is None:
        return None

    reference = None
    for reference_key in ("reference", "target", "answer", "output", "response"):
        if record.get(reference_key) is not None and str(record[reference_key]).strip():
            reference = str(record[reference_key])
            break
    if reference is None:
        return None

    return WeatherPromptExample(
        sample_id=sample_id,
        prompt_text=prompt,
        reference_text=reference,
        prompt_messages=[{"role": "user", "content": prompt}],
        metadata={key: value for key, value in record.items() if key not in {"prompt", "reference"}},
    )


def load_weather_prompt_examples(
    dataset_path: Path,
    tokenizer,
    max_samples: int = 0,
    shuffle: bool = False,
    seed: int = 42,
) -> list[WeatherPromptExample]:
    records = load_json_records(dataset_path)
    examples: list[WeatherPromptExample] = []
    for sample_id, record in enumerate(records):
        example = record_to_example(record, sample_id=sample_id, tokenizer=tokenizer)
        if example is not None:
            examples.append(example)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(examples)
    if max_samples > 0:
        examples = examples[:max_samples]
    if not examples:
        raise ValueError(f"No usable prompt/reference weather samples found in {dataset_path}.")
    return examples
