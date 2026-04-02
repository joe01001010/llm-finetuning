#!/usr/bin/env python
"""
Shared weather fine-tuning agent logic for SFT, PPO, and evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL_PATH = "/local-containers/Qwen2-7B-Instruct"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "seattle_weather_chat_train.jsonl"
DEFAULT_EVAL_DATA_PATH = PROJECT_ROOT / "data" / "seattle_weather_chat_eval.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "adapter-weights"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "weather_model_report.json"
VALID_LOAD_MODES = {"4bit", "16bit", "32bit"}
COMMON_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class WeatherPromptExample:
    sample_id: int
    prompt_text: str
    reference_text: str
    prompt_messages: list[dict[str, str]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SFTFeature:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


@dataclass
class RewardConfig:
    json_valid_weight: float = 0.2
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
    reward_clip_min: float = -1
    reward_clip_max: float = 1
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


@dataclass
class SFTTrainingConfig:
    mode: str = "qlora"
    model_path: str = DEFAULT_MODEL_PATH
    tokenizer_path: str = ""
    dataset_path: str | Path = DEFAULT_DATA_PATH
    output_dir: str | Path = ""
    num_train_epochs: float = 1.0
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    save_steps: int = 100
    logging_steps: int = 10
    save_total_limit: int = 2
    max_length: int = 768
    max_samples: int = 0
    seed: int = 42
    trust_remote_code: bool = False
    gradient_checkpointing: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass
class PPOTrainingConfig:
    mode: str = "qlora"
    model_path: str = DEFAULT_MODEL_PATH
    tokenizer_path: str = ""
    adapter_path: str | Path = ""
    dataset_path: str | Path = DEFAULT_DATA_PATH
    output_dir: str | Path = ""
    reference_load_mode: str = "auto"
    trust_remote_code: bool = False
    shuffle: bool = False
    num_train_epochs: int = PPOHyperParameters.num_train_epochs
    rollout_batch_size: int = PPOHyperParameters.rollout_batch_size
    mini_batch_size: int = PPOHyperParameters.mini_batch_size
    gradient_accumulation_steps: int = PPOHyperParameters.gradient_accumulation_steps
    ppo_epochs: int = PPOHyperParameters.ppo_epochs
    learning_rate: float = PPOHyperParameters.learning_rate
    weight_decay: float = PPOHyperParameters.weight_decay
    gamma: float = PPOHyperParameters.gamma
    gae_lambda: float = PPOHyperParameters.gae_lambda
    clip_range: float = PPOHyperParameters.clip_range
    clip_range_value: float = PPOHyperParameters.clip_range_value
    value_loss_coef: float = PPOHyperParameters.value_loss_coef
    entropy_coef: float = PPOHyperParameters.entropy_coef
    kl_coef: float = PPOHyperParameters.kl_coef
    max_grad_norm: float = PPOHyperParameters.max_grad_norm
    max_prompt_length: int = PPOHyperParameters.max_prompt_length
    max_new_tokens: int = PPOHyperParameters.max_new_tokens
    temperature: float = PPOHyperParameters.temperature
    top_p: float = PPOHyperParameters.top_p
    max_prompt_samples: int = PPOHyperParameters.max_prompt_samples
    save_steps: int = PPOHyperParameters.save_steps
    logging_steps: int = PPOHyperParameters.logging_steps
    seed: int = PPOHyperParameters.seed
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    value_head_dropout: float = 0.1
    reward_config: RewardConfig = field(default_factory=RewardConfig)


@dataclass
class EvaluationConfig:
    base_model_path: str = DEFAULT_MODEL_PATH
    tokenizer_path: str = ""
    dataset_path: str | Path = DEFAULT_EVAL_DATA_PATH
    output_report: str | Path = DEFAULT_REPORT_PATH
    output_predictions: str | Path = ""
    sft_lora_adapter: str | Path = ""
    sft_qlora_adapter: str | Path = ""
    ppo_lora_adapter: str | Path = ""
    ppo_qlora_adapter: str | Path = ""
    max_samples: int = 200
    shuffle: bool = False
    seed: int = 42
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    base_load_mode: str = "16bit"
    trust_remote_code: bool = False
    reward_config: RewardConfig = field(default_factory=RewardConfig)


@dataclass
class ModelVariantSpec:
    label: str
    adapter_path: Path | None


@dataclass
class RolloutBatch:
    prompt_width: int
    sample_ids: list[int]
    prompt_texts: list[str]
    reference_texts: list[str]
    response_texts: list[str]
    reward_breakdowns: list[dict[str, Any]]
    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    sequence_ids: torch.Tensor
    sequence_attention_mask: torch.Tensor
    response_ids: torch.Tensor
    response_mask: torch.Tensor
    old_logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    score_rewards: torch.Tensor
    kl_divergence: torch.Tensor
    response_lengths: torch.Tensor
    eos_flags: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.sequence_ids.size(0))

    @property
    def response_length(self) -> int:
        return int(self.response_ids.size(1))

    def iter_minibatches(self, mini_batch_size: int, shuffle: bool = True) -> Iterator[dict[str, torch.Tensor]]:
        indices = torch.arange(self.batch_size, device=self.sequence_ids.device)
        if shuffle:
            indices = indices[torch.randperm(self.batch_size, device=indices.device)]

        for start in range(0, self.batch_size, mini_batch_size):
            batch_indices = indices[start : start + mini_batch_size]
            yield {
                "sequence_ids": self.sequence_ids[batch_indices],
                "sequence_attention_mask": self.sequence_attention_mask[batch_indices],
                "response_mask": self.response_mask[batch_indices],
                "old_logprobs": self.old_logprobs[batch_indices],
                "ref_logprobs": self.ref_logprobs[batch_indices],
                "old_values": self.old_values[batch_indices],
                "advantages": self.advantages[batch_indices],
                "returns": self.returns[batch_indices],
            }


class SupervisedWeatherDataset(Dataset):
    """
    Chat-style SFT dataset where the loss is applied only to assistant tokens.
    """

    def __init__(self, examples: list[WeatherPromptExample], tokenizer, max_length: int) -> None:
        self.features: list[SFTFeature] = []
        eos_suffix = tokenizer.eos_token or ""

        for example in examples:
            prompt_tokens = tokenizer(
                example.prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            full_tokens = tokenizer(
                f"{example.prompt_text}{example.reference_text}{eos_suffix}",
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )

            input_ids = list(full_tokens["input_ids"])
            attention_mask = list(full_tokens["attention_mask"])
            prompt_length = len(prompt_tokens["input_ids"])

            if not input_ids or prompt_length >= len(input_ids):
                continue

            labels = input_ids.copy()
            labels[:prompt_length] = [-100] * prompt_length
            if all(label == -100 for label in labels):
                continue

            self.features.append(SFTFeature(input_ids=input_ids, attention_mask=attention_mask, labels=labels))

        if not self.features:
            raise ValueError("No usable SFT examples remained after tokenization.")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        feature = self.features[index]
        return {
            "input_ids": feature.input_ids,
            "attention_mask": feature.attention_mask,
            "labels": feature.labels,
        }


class SupervisedDataCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": pad_sequence(
                [torch.tensor(feature["input_ids"], dtype=torch.long) for feature in features],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ),
            "attention_mask": pad_sequence(
                [torch.tensor(feature["attention_mask"], dtype=torch.long) for feature in features],
                batch_first=True,
                padding_value=0,
            ),
            "labels": pad_sequence(
                [torch.tensor(feature["labels"], dtype=torch.long) for feature in features],
                batch_first=True,
                padding_value=-100,
            ),
        }


def infer_hidden_size(config) -> int:
    for attribute in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, attribute, None)
        if isinstance(value, int):
            return value
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        for attribute in ("hidden_size", "n_embd", "d_model"):
            value = getattr(text_config, attribute, None)
            if isinstance(value, int):
                return value
    raise ValueError("Unable to infer hidden size for value head initialization.")


class ScalarValueHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.summary = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.summary(self.dropout(hidden_states)).squeeze(-1)


class CausalLMWithValueHead(nn.Module):
    def __init__(self, pretrained_model, value_head_dropout: float = 0.1) -> None:
        super().__init__()
        self.pretrained_model = pretrained_model
        self.value_head = ScalarValueHead(
            hidden_size=infer_hidden_size(pretrained_model.config),
            dropout=value_head_dropout,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return {
            "logits": outputs.logits,
            "values": self.value_head(outputs.hidden_states[-1]),
        }

    def generate(self, *args, **kwargs):
        return self.pretrained_model.generate(*args, **kwargs)

    def save_value_head(self, path: Path) -> None:
        torch.save(self.value_head.state_dict(), path)

    def load_value_head(self, path: Path, strict: bool = True) -> None:
        state_dict = torch.load(path, map_location="cpu")
        self.value_head.load_state_dict(state_dict, strict=strict)


def resolve_path(path_like: str | Path, root: Path | None = None) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    if root is None:
        return path.resolve()
    return (root / path).resolve()


def looks_like_local_path(path_like: str | Path) -> bool:
    text = str(path_like)
    return (
        text.startswith(".")
        or text.startswith("/")
        or text.startswith("\\")
        or os.path.sep in text
        or (os.path.altsep is not None and os.path.altsep in text)
        or (len(text) >= 2 and text[1] == ":")
    )


def resolve_pretrained_source(
    source: str | Path,
    *,
    root: Path | None = None,
    kind: str = "model",
) -> tuple[str, bool]:
    source_text = str(source)
    if not looks_like_local_path(source_text):
        return source_text, False

    candidate = resolve_path(source_text, root)
    if candidate.exists():
        return str(candidate), True

    raise FileNotFoundError(
        f"Local {kind} path was not found: {candidate}\n"
        f"If you intended to use the locally pulled model, verify the path is mounted "
        f"inside the container and run `python python/pull_model.py` first."
    )


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token is not None:
        return
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return
    raise ValueError("Tokenizer has neither a pad token nor an eos token.")


def get_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def find_lora_target_modules(model) -> list[str]:
    matches: set[str] = set()
    for module_name, _module in model.named_modules():
        leaf_name = module_name.split(".")[-1]
        if leaf_name in COMMON_LORA_TARGETS:
            matches.add(leaf_name)
    if not matches:
        matches.update({"q_proj", "v_proj"})
    return sorted(matches)


def model_device(model) -> torch.device:
    for parameter in model.parameters():
        return parameter.device
    raise RuntimeError("Unable to infer model device because the model has no parameters.")


def tensor_dict_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def build_response_mask(response_ids: torch.Tensor, pad_token_id: int | None, eos_token_id: int | None) -> torch.Tensor:
    if response_ids.ndim != 2:
        raise ValueError("response_ids must have shape [batch, response_length].")

    if pad_token_id is None:
        mask = torch.ones_like(response_ids, dtype=torch.bool)
    else:
        mask = response_ids.ne(pad_token_id)

    if eos_token_id is None:
        return mask

    eos_seen = torch.zeros(response_ids.size(0), dtype=torch.bool, device=response_ids.device)
    trimmed_mask = torch.zeros_like(mask)
    for token_index in range(response_ids.size(1)):
        active = mask[:, token_index] & ~eos_seen
        trimmed_mask[:, token_index] = active
        eos_seen = eos_seen | (active & response_ids[:, token_index].eq(eos_token_id))
    return trimmed_mask


def decode_responses(tokenizer, response_ids: torch.Tensor, response_mask: torch.Tensor) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    texts: list[str] = []
    response_lengths = response_mask.long().sum(dim=1)
    eos_flags: list[bool] = []
    for row_ids, row_mask in zip(response_ids, response_mask):
        valid_ids = row_ids[row_mask]
        texts.append(tokenizer.decode(valid_ids, skip_special_tokens=True).strip())
        eos_flags.append(
            bool(
                tokenizer.eos_token_id is not None
                and valid_ids.numel() > 0
                and valid_ids[-1].item() == tokenizer.eos_token_id
            )
        )
    return texts, response_lengths, torch.tensor(eos_flags, dtype=torch.bool, device=response_ids.device)


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    token_logprobs = logits[:, :-1, :].float().log_softmax(dim=-1)
    gathered = token_logprobs.gather(dim=-1, index=labels[:, 1:].unsqueeze(-1)).squeeze(-1)
    return torch.nan_to_num(gathered, nan=0.0, posinf=0.0, neginf=0.0)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_probs = logits.float().log_softmax(dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    mask = mask.to(values.dtype)
    if dim is None:
        denom = mask.sum().clamp_min(1.0)
        return (values * mask).sum() / denom
    denom = mask.sum(dim=dim).clamp_min(1.0)
    return (values * mask).sum(dim=dim) / denom


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    mean = masked_mean(values, mask)
    variance = masked_mean((values - mean) ** 2, mask)
    whitened = (values - mean) / torch.sqrt(variance + epsilon)
    return whitened * mask


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(rewards.size(0), device=rewards.device, dtype=rewards.dtype)
    next_values = torch.zeros(rewards.size(0), device=rewards.device, dtype=rewards.dtype)

    for timestep in reversed(range(rewards.size(1))):
        timestep_mask = mask[:, timestep].to(rewards.dtype)
        delta = rewards[:, timestep] + gamma * next_values - values[:, timestep]
        last_advantage = delta + gamma * gae_lambda * last_advantage
        last_advantage = last_advantage * timestep_mask
        advantages[:, timestep] = last_advantage
        next_values = values[:, timestep] * timestep_mask

    returns = (advantages + values) * mask
    return advantages * mask, returns


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
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"


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
        value = record.get(prompt_key)
        if value is not None and str(value).strip():
            prompt = str(value)
            break
    if prompt is None:
        return None

    reference = None
    for reference_key in ("reference", "target", "answer", "output", "response"):
        value = record.get(reference_key)
        if value is not None and str(value).strip():
            reference = str(value)
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
        example = record_to_example(record, sample_id, tokenizer)
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
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
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


def infer_schema(
    examples: Sequence[dict[str, Any]],
) -> tuple[list[str], set[str], dict[str, tuple[float, float]], dict[str, set[str]]] | None:
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

    if extra_text_length(prediction) > config.verbosity_char_threshold:
        reward -= config.verbosity_penalty

    reward = float(max(config.reward_clip_min, min(config.reward_clip_max, reward)))
    return reward, {
        **breakdown,
        "reward": round(reward, 6),
        "json_valid": json_valid,
        "schema_score": round(schema_score, 6),
        "missing_keys": missing_keys,
        "extra_keys": extra_keys,
        "parse_reasons": sorted(parse_reasons),
    }


def add_reward_config_args(parser) -> None:
    parser.add_argument("--json-valid-weight", type=float, default=RewardConfig.json_valid_weight)
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
    parser.add_argument("--reward-clip-min", type=float, default=RewardConfig.reward_clip_min)
    parser.add_argument("--reward-clip-max", type=float, default=RewardConfig.reward_clip_max)
    parser.add_argument("--verbosity-char-threshold", type=int, default=RewardConfig.verbosity_char_threshold)


def reward_config_from_namespace(args) -> RewardConfig:
    return RewardConfig(
        json_valid_weight=args.json_valid_weight,
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
        reward_clip_min=args.reward_clip_min,
        reward_clip_max=args.reward_clip_max,
        verbosity_char_threshold=args.verbosity_char_threshold,
    )


def reward_config_from_args(args) -> RewardConfig:
    return reward_config_from_namespace(args)


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


class WeatherAgent:
    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or PROJECT_ROOT

    def load_tokenizer(
        self,
        tokenizer_path: str,
        fallback_model_path: str,
        trust_remote_code: bool,
        *,
        padding_side: str,
    ):
        load_path = tokenizer_path or fallback_model_path
        tokenizer_source, is_local = resolve_pretrained_source(load_path, kind="tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            use_fast=True,
            trust_remote_code=trust_remote_code,
            local_files_only=is_local,
        )
        ensure_pad_token(tokenizer)
        tokenizer.padding_side = padding_side
        return tokenizer

    def resolve_sft_output_dir(self, config: SFTTrainingConfig) -> Path:
        if config.output_dir:
            return resolve_path(config.output_dir, self.project_root)
        return DEFAULT_OUTPUT_ROOT / f"sft-{config.mode}"

    def resolve_ppo_output_dir(self, config: PPOTrainingConfig) -> Path:
        if config.output_dir:
            return resolve_path(config.output_dir, self.project_root)
        return DEFAULT_OUTPUT_ROOT / f"ppo-{config.mode}"

    def load_lm_backbone(
        self,
        model_path: str,
        load_mode: str,
        trust_remote_code: bool,
        *,
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

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": is_local,
        }
        if load_mode == "16bit" and compute_dtype != torch.float32:
            load_kwargs["torch_dtype"] = compute_dtype
        model = AutoModelForCausalLM.from_pretrained(model_source, **load_kwargs)
        if device_map is None and torch.cuda.is_available():
            model.to(torch.device("cuda"))
        resolved_mode = "16bit" if load_mode == "16bit" and compute_dtype != torch.float32 else load_mode
        return model, resolved_mode, compute_dtype

    def load_sft_backbone(self, config: SFTTrainingConfig):
        load_mode = "4bit" if config.mode == "qlora" else "16bit"
        model, resolved_mode, compute_dtype = self.load_lm_backbone(
            config.model_path,
            load_mode,
            config.trust_remote_code,
            device_map="auto" if load_mode == "4bit" else None,
        )
        if load_mode == "4bit":
            model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False
        return model, resolved_mode, compute_dtype

    def enable_model_input_grads(self, model) -> None:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
            return

        embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        if embeddings is None:
            return

        def hook(_module, _inputs, output):
            output.requires_grad_(True)

        embeddings.register_forward_hook(hook)

    def attach_trainable_adapter(
        self,
        model,
        *,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
    ):
        target_modules = find_lora_target_modules(model)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
        )
        return get_peft_model(model, lora_config)

    def prepare_sft_model(self, model, tokenizer, config: SFTTrainingConfig):
        model = self.attach_trainable_adapter(
            model,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )
        if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        self.enable_model_input_grads(model)
        if tokenizer.pad_token_id is not None and hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = tokenizer.pad_token_id
        return model

    def build_training_args(self, config: SFTTrainingConfig, output_dir: Path) -> TrainingArguments:
        use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
        use_fp16 = torch.cuda.is_available() and not use_bf16
        optim = "paged_adamw_8bit" if config.mode == "qlora" else "adamw_torch"
        return TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            save_total_limit=config.save_total_limit,
            bf16=use_bf16,
            fp16=use_fp16,
            optim=optim,
            report_to=[],
            remove_unused_columns=False,
            gradient_checkpointing=config.gradient_checkpointing,
            lr_scheduler_type="cosine",
            dataloader_pin_memory=torch.cuda.is_available(),
        )

    def train_sft(self, config: SFTTrainingConfig) -> dict[str, Any]:
        set_seed(config.seed)
        output_dir = self.resolve_sft_output_dir(config)
        output_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = self.load_tokenizer(
            config.tokenizer_path,
            config.model_path,
            config.trust_remote_code,
            padding_side="right",
        )
        examples = load_weather_prompt_examples(
            dataset_path=resolve_path(config.dataset_path, self.project_root),
            tokenizer=tokenizer,
            max_samples=config.max_samples,
            shuffle=False,
            seed=config.seed,
        )
        dataset = SupervisedWeatherDataset(examples, tokenizer, config.max_length)
        model, base_model_load_mode, compute_dtype = self.load_sft_backbone(config)
        model = self.prepare_sft_model(model, tokenizer, config)

        trainer = Trainer(
            model=model,
            args=self.build_training_args(config, output_dir),
            train_dataset=dataset,
            data_collator=SupervisedDataCollator(tokenizer),
        )
        train_result = trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        runtime_config = {
            "training_algorithm": "sft",
            "stage": "sft",
            "adapter_mode": config.mode,
            "base_model_path": config.model_path,
            "tokenizer_path": config.tokenizer_path or config.model_path,
            "dataset_path": str(resolve_path(config.dataset_path, self.project_root)),
            "dataset_size": len(dataset),
            "base_model_load_mode": base_model_load_mode,
            "compute_dtype": str(compute_dtype).replace("torch.", ""),
            "seed": config.seed,
            "sft": {
                "num_train_epochs": config.num_train_epochs,
                "batch_size": config.batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "learning_rate": config.learning_rate,
                "max_length": config.max_length,
                "lora_r": config.lora_r,
                "lora_alpha": config.lora_alpha,
                "lora_dropout": config.lora_dropout,
            },
        }
        summary = {
            "training_algorithm": "sft",
            "stage": "sft",
            "mode": config.mode,
            "output_dir": str(output_dir),
            "dataset_size": len(dataset),
            "train_loss": getattr(train_result, "training_loss", None),
            "global_step": getattr(train_result, "global_step", None),
            "metrics": getattr(train_result, "metrics", {}),
        }
        save_json(output_dir / "adapter_runtime_config.json", runtime_config)
        save_json(output_dir / "sft_training_summary.json", summary)
        return summary

    def determine_policy_load_mode(self, mode: str) -> str:
        return "4bit" if mode == "qlora" else "16bit"

    def determine_reference_load_mode(self, mode: str, requested_mode: str) -> str:
        if requested_mode in VALID_LOAD_MODES:
            return requested_mode
        return "4bit" if mode == "qlora" else ("4bit" if torch.cuda.is_available() else "16bit")

    def prepare_policy_model(self, config: PPOTrainingConfig, adapter_path: Path) -> tuple[CausalLMWithValueHead, str, torch.dtype]:
        load_mode = self.determine_policy_load_mode(config.mode)
        backbone, resolved_load_mode, compute_dtype = self.load_lm_backbone(
            config.model_path,
            load_mode,
            config.trust_remote_code,
            device_map="auto" if load_mode == "4bit" else None,
        )
        if load_mode == "4bit":
            backbone = prepare_model_for_kbit_training(backbone)

        backbone = PeftModel.from_pretrained(backbone, str(adapter_path), is_trainable=True)
        backbone.config.use_cache = False
        if hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable()
        self.enable_model_input_grads(backbone)

        policy_model = CausalLMWithValueHead(backbone, value_head_dropout=config.value_head_dropout)
        value_head_path = adapter_path / "ppo_value_head.pt"
        if value_head_path.exists():
            policy_model.load_value_head(value_head_path, strict=False)

        policy_model.value_head.to(model_device(policy_model.pretrained_model))
        return policy_model, resolved_load_mode, compute_dtype

    def prepare_reference_model(self, config: PPOTrainingConfig, adapter_path: Path):
        load_mode = self.determine_reference_load_mode(config.mode, config.reference_load_mode)
        reference_model, resolved_load_mode, _compute_dtype = self.load_lm_backbone(
            config.model_path,
            load_mode,
            config.trust_remote_code,
            device_map="auto" if load_mode == "4bit" else None,
        )
        reference_model = PeftModel.from_pretrained(reference_model, str(adapter_path), is_trainable=False)
        reference_model.eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
        return reference_model, resolved_load_mode

    def get_response_stats(
        self,
        policy_model: CausalLMWithValueHead,
        sequence_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_width: int,
        response_width: int,
    ):
        outputs = policy_model(input_ids=sequence_ids, attention_mask=attention_mask)
        token_logprobs = logprobs_from_logits(outputs["logits"], sequence_ids)
        token_values = torch.nan_to_num(outputs["values"][:, :-1].float(), nan=0.0, posinf=0.0, neginf=0.0)
        start_index = max(prompt_width - 1, 0)
        end_index = start_index + response_width
        return (
            torch.nan_to_num(token_logprobs[:, start_index:end_index], nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(token_values[:, start_index:end_index], nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(outputs["logits"][:, start_index:end_index, :].float(), nan=0.0, posinf=0.0, neginf=0.0),
        )

    def get_reference_logprobs(
        self,
        reference_model,
        sequence_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_width: int,
        response_width: int,
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = reference_model(input_ids=sequence_ids, attention_mask=attention_mask, return_dict=True)
        token_logprobs = logprobs_from_logits(outputs.logits, sequence_ids)
        start_index = max(prompt_width - 1, 0)
        end_index = start_index + response_width
        return torch.nan_to_num(token_logprobs[:, start_index:end_index], nan=0.0, posinf=0.0, neginf=0.0)

    def collect_rollout(
        self,
        batch_examples: list[WeatherPromptExample],
        tokenizer,
        policy_model: CausalLMWithValueHead,
        reference_model,
        schema,
        config: PPOTrainingConfig,
    ) -> RolloutBatch:
        device = model_device(policy_model.pretrained_model)
        prompt_texts = [example.prompt_text for example in batch_examples]
        reference_texts = [example.reference_text for example in batch_examples]
        prompt_batch = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_prompt_length,
        )
        prompt_batch = tensor_dict_to_device(prompt_batch, device)
        prompt_width = int(prompt_batch["input_ids"].size(1))

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": config.temperature > 0.0,
            "renormalize_logits": True,
            "remove_invalid_values": True,
        }
        if config.temperature > 0.0:
            generation_kwargs["temperature"] = config.temperature
            generation_kwargs["top_p"] = config.top_p

        policy_model.eval()
        with torch.no_grad():
            sequence_ids = policy_model.generate(**prompt_batch, **generation_kwargs)

        response_ids = sequence_ids[:, prompt_width:]
        response_mask = build_response_mask(response_ids, tokenizer.pad_token_id, tokenizer.eos_token_id)
        response_texts, response_lengths, eos_flags = decode_responses(tokenizer, response_ids, response_mask)
        sequence_attention_mask = torch.cat([prompt_batch["attention_mask"], response_mask.long()], dim=1)

        with torch.no_grad():
            old_logprobs, old_values, _response_logits = self.get_response_stats(
                policy_model,
                sequence_ids,
                sequence_attention_mask,
                prompt_width,
                response_ids.size(1),
            )
            reference_device = model_device(reference_model)
            ref_logprobs = self.get_reference_logprobs(
                reference_model,
                sequence_ids.to(reference_device),
                sequence_attention_mask.to(reference_device),
                prompt_width,
                response_ids.size(1),
            ).to(device)

        score_rewards: list[float] = []
        reward_breakdowns: list[dict[str, Any]] = []
        for prediction_text, reference_text in zip(response_texts, reference_texts):
            reward_value, breakdown = compute_weather_reward(
                prediction=prediction_text,
                reference=reference_text,
                schema=schema,
                config=config.reward_config,
            )
            score_rewards.append(reward_value)
            reward_breakdowns.append(breakdown)

        score_reward_tensor = torch.tensor(score_rewards, device=device, dtype=torch.float32)
        rewards = -config.kl_coef * torch.nan_to_num(old_logprobs - ref_logprobs, nan=0.0, posinf=0.0, neginf=0.0)
        rewards = rewards * response_mask
        last_token_indices = response_mask.long().sum(dim=1) - 1
        for row_index, last_index in enumerate(last_token_indices.tolist()):
            if last_index >= 0:
                rewards[row_index, last_index] += score_reward_tensor[row_index]

        advantages, returns = compute_gae(
            rewards=rewards,
            values=old_values * response_mask,
            mask=response_mask,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
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
        self,
        policy_model: CausalLMWithValueHead,
        optimizer: torch.optim.Optimizer,
        rollout: RolloutBatch,
        config: PPOTrainingConfig,
    ) -> dict[str, float]:
        policy_model.train()
        stats: defaultdict[str, list[float]] = defaultdict(list)
        step_count = 0
        trainable_parameters = [parameter for parameter in policy_model.parameters() if parameter.requires_grad]
        optimizer.zero_grad(set_to_none=True)

        for _ppo_epoch in range(config.ppo_epochs):
            for minibatch in rollout.iter_minibatches(config.mini_batch_size, shuffle=True):
                new_logprobs, new_values, response_logits = self.get_response_stats(
                    policy_model,
                    minibatch["sequence_ids"],
                    minibatch["sequence_attention_mask"],
                    rollout.prompt_width,
                    rollout.response_length,
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
                clipped = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range) * minibatch["advantages"]
                policy_loss = -masked_mean(torch.minimum(unclipped, clipped), response_mask)

                value_delta = new_values - minibatch["old_values"]
                clipped_values = minibatch["old_values"] + torch.clamp(
                    value_delta,
                    -config.clip_range_value,
                    config.clip_range_value,
                )
                value_loss_unclipped = (new_values - minibatch["returns"]) ** 2
                value_loss_clipped = (clipped_values - minibatch["returns"]) ** 2
                value_loss = 0.5 * masked_mean(torch.maximum(value_loss_unclipped, value_loss_clipped), response_mask)

                entropy_bonus = masked_mean(entropy_from_logits(response_logits), response_mask)
                loss = policy_loss + config.value_loss_coef * value_loss - config.entropy_coef * entropy_bonus

                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    stats["skipped_non_finite_loss"].append(1.0)
                    continue

                (loss / config.gradient_accumulation_steps).backward()
                step_count += 1

                if step_count % config.gradient_accumulation_steps == 0:
                    has_non_finite_grad = any(
                        parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                        for parameter in trainable_parameters
                    )
                    if has_non_finite_grad:
                        optimizer.zero_grad(set_to_none=True)
                        stats["skipped_non_finite_grad"].append(1.0)
                        continue
                    clip_grad_norm_(trainable_parameters, config.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    stats["skipped_non_finite_grad"].append(0.0)

                approx_kl = masked_mean(
                    torch.nan_to_num(minibatch["old_logprobs"] - new_logprobs, nan=0.0, posinf=0.0, neginf=0.0),
                    response_mask,
                )
                clip_fraction = masked_mean(
                    (torch.abs(ratio - 1.0) > config.clip_range).to(new_logprobs.dtype),
                    response_mask,
                )
                stats["loss"].append(float(loss.detach().item()))
                stats["policy_loss"].append(float(policy_loss.detach().item()))
                stats["value_loss"].append(float(value_loss.detach().item()))
                stats["entropy"].append(float(entropy_bonus.detach().item()))
                stats["approx_kl"].append(float(approx_kl.detach().item()))
                stats["clip_fraction"].append(float(clip_fraction.detach().item()))
                stats["skipped_non_finite_loss"].append(0.0)

        if step_count % config.gradient_accumulation_steps != 0:
            has_non_finite_grad = any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in trainable_parameters
            )
            if not has_non_finite_grad:
                clip_grad_norm_(trainable_parameters, config.max_grad_norm)
                optimizer.step()
            else:
                stats["skipped_non_finite_grad"].append(1.0)
            optimizer.zero_grad(set_to_none=True)

        return {key: round(sum(values) / len(values), 6) for key, values in stats.items() if values}

    def summarize_rollout(self, rollout: RolloutBatch) -> dict[str, float]:
        accuracy_values = [float(item["accuracy"]) for item in rollout.reward_breakdowns]
        hallucination_values = [float(item["hallucinated"]) for item in rollout.reward_breakdowns]
        a1_values = [float(item["a1_score"]) for item in rollout.reward_breakdowns]
        exact_json_values = [float(item["exact_json_match"]) for item in rollout.reward_breakdowns]
        return {
            "reward": round(float(rollout.score_rewards.mean().item()), 6),
            "accuracy": round(sum(accuracy_values) / len(accuracy_values), 6),
            "hallucination_rate": round(sum(hallucination_values) / len(hallucination_values), 6),
            "a1_score": round(sum(a1_values) / len(a1_values), 6),
            "exact_json_match": round(sum(exact_json_values) / len(exact_json_values), 6),
            "kl_divergence": round(float(rollout.kl_divergence.mean().item()), 6),
            "generated_tokens": round(float(rollout.response_lengths.float().mean().item()), 6),
            "eos_rate": round(float(rollout.eos_flags.float().mean().item()), 6),
        }

    def save_ppo_artifacts(
        self,
        output_dir: Path,
        policy_model: CausalLMWithValueHead,
        tokenizer,
        runtime_config: dict[str, Any],
        metrics: dict[str, Any],
        reward_config: RewardConfig,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        policy_model.pretrained_model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        policy_model.save_value_head(output_dir / "ppo_value_head.pt")
        save_json(output_dir / "adapter_runtime_config.json", runtime_config)
        save_json(output_dir / "ppo_runtime_config.json", runtime_config)
        save_json(output_dir / "ppo_reward_config.json", reward_config.to_dict())
        save_json(output_dir / "ppo_training_summary.json", metrics)

    def train_ppo(self, config: PPOTrainingConfig) -> dict[str, Any]:
        if not config.adapter_path:
            raise ValueError("PPO training requires --adapter-path pointing at a saved SFT adapter.")

        set_seed(config.seed)
        adapter_path = resolve_path(config.adapter_path, self.project_root)
        output_dir = self.resolve_ppo_output_dir(config)
        if output_dir.resolve() == adapter_path.resolve():
            raise ValueError("PPO output_dir must be different from adapter_path so the SFT adapter is preserved.")
        output_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = self.load_tokenizer(
            config.tokenizer_path,
            config.model_path,
            config.trust_remote_code,
            padding_side="left",
        )
        examples = load_weather_prompt_examples(
            dataset_path=resolve_path(config.dataset_path, self.project_root),
            tokenizer=tokenizer,
            max_samples=config.max_prompt_samples,
            shuffle=config.shuffle,
            seed=config.seed,
        )
        schema = infer_schema([{"reference": example.reference_text} for example in examples])

        policy_model, base_model_load_mode, compute_dtype = self.prepare_policy_model(config, adapter_path)
        reference_model, reference_load_mode = self.prepare_reference_model(config, adapter_path)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in policy_model.parameters() if parameter.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        runtime_config = {
            "training_algorithm": "ppo",
            "stage": "ppo",
            "mode": config.mode,
            "base_model_path": config.model_path,
            "tokenizer_path": config.tokenizer_path or config.model_path,
            "adapter_init_path": str(adapter_path),
            "initialization_source": str(adapter_path),
            "base_model_load_mode": base_model_load_mode,
            "reference_load_mode": reference_load_mode,
            "compute_dtype": str(compute_dtype).replace("torch.", ""),
            "dataset_path": str(resolve_path(config.dataset_path, self.project_root)),
            "seed": config.seed,
            "ppo": {
                "num_train_epochs": config.num_train_epochs,
                "rollout_batch_size": config.rollout_batch_size,
                "mini_batch_size": config.mini_batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "ppo_epochs": config.ppo_epochs,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "gamma": config.gamma,
                "gae_lambda": config.gae_lambda,
                "clip_range": config.clip_range,
                "clip_range_value": config.clip_range_value,
                "value_loss_coef": config.value_loss_coef,
                "entropy_coef": config.entropy_coef,
                "kl_coef": config.kl_coef,
                "max_grad_norm": config.max_grad_norm,
                "max_prompt_length": config.max_prompt_length,
                "max_new_tokens": config.max_new_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_prompt_samples": config.max_prompt_samples,
                "save_steps": config.save_steps,
                "logging_steps": config.logging_steps,
            },
            "reward": config.reward_config.to_dict(),
        }

        dataloader = DataLoader(
            examples,
            batch_size=config.rollout_batch_size,
            shuffle=config.shuffle,
            collate_fn=lambda batch: batch,
        )

        global_step = 0
        history: defaultdict[str, list[float]] = defaultdict(list)
        total_steps = len(dataloader) * max(config.num_train_epochs, 1)
        progress = tqdm(total=total_steps, desc="PPO weather training", dynamic_ncols=True)

        for epoch_index in range(config.num_train_epochs):
            for batch_examples in dataloader:
                global_step += 1
                rollout = self.collect_rollout(batch_examples, tokenizer, policy_model, reference_model, schema, config)
                rollout_stats = self.summarize_rollout(rollout)
                optimize_stats = self.optimize_policy(policy_model, optimizer, rollout, config)

                for key, value in {**rollout_stats, **optimize_stats}.items():
                    history[key].append(value)

                progress.update(1)
                progress.set_postfix(
                    reward=f"{rollout_stats['reward']:.3f}",
                    acc=f"{rollout_stats['accuracy']:.3f}",
                    a1=f"{rollout_stats['a1_score']:.3f}",
                    kl=f"{rollout_stats['kl_divergence']:.3f}",
                )

                if global_step % config.logging_steps == 0:
                    progress.write(
                        f"step={global_step} "
                        f"reward={rollout_stats['reward']:.4f} "
                        f"accuracy={rollout_stats['accuracy']:.4f} "
                        f"a1={rollout_stats['a1_score']:.4f} "
                        f"kl={rollout_stats['kl_divergence']:.4f}"
                    )

                if config.save_steps > 0 and global_step % config.save_steps == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    checkpoint_metrics = {
                        "training_algorithm": "ppo",
                        "stage": "ppo",
                        "global_step": global_step,
                        "epoch": epoch_index + 1,
                        "history_means": {
                            key: round(sum(values) / len(values), 6)
                            for key, values in history.items()
                            if values
                        },
                    }
                    self.save_ppo_artifacts(
                        checkpoint_dir,
                        policy_model,
                        tokenizer,
                        runtime_config,
                        checkpoint_metrics,
                        config.reward_config,
                    )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        progress.close()

        final_metrics = {
            "training_algorithm": "ppo",
            "stage": "ppo",
            "mode": config.mode,
            "output_dir": str(output_dir),
            "global_step": global_step,
            "num_examples": len(examples),
            "history_means": {
                key: round(sum(values) / len(values), 6)
                for key, values in history.items()
                if values
            },
        }
        self.save_ppo_artifacts(
            output_dir,
            policy_model,
            tokenizer,
            runtime_config,
            final_metrics,
            config.reward_config,
        )
        return final_metrics

    def load_runtime_config(self, adapter_path: Path) -> dict[str, Any]:
        for file_name in ("adapter_runtime_config.json", "ppo_runtime_config.json"):
            config_path = adapter_path / file_name
            if config_path.exists():
                payload = json.loads(config_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return payload
        return {}

    def infer_evaluation_load_mode(self, label: str, adapter_path: Path | None, base_load_mode: str) -> str:
        if label == "base":
            return base_load_mode
        if adapter_path is not None:
            runtime_mode = self.load_runtime_config(adapter_path).get("base_model_load_mode")
            if runtime_mode in VALID_LOAD_MODES:
                return runtime_mode
        return "4bit" if "qlora" in label else "16bit"

    def load_model_variant(self, base_model_path: str, adapter_path: Path | None, load_mode: str, trust_remote_code: bool):
        model, _resolved_mode, _compute_dtype = self.load_lm_backbone(
            base_model_path,
            load_mode,
            trust_remote_code,
            device_map="auto" if load_mode == "4bit" else None,
        )
        if adapter_path is not None:
            model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()
        return model

    def generate_prediction(self, model, tokenizer, prompt_text: str, config: EvaluationConfig) -> tuple[str, float, int, int]:
        inputs = tokenizer(prompt_text, return_tensors="pt")
        device = model_device(model)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": config.temperature > 0.0,
        }
        if config.temperature > 0.0:
            generation_kwargs["temperature"] = config.temperature
            generation_kwargs["top_p"] = config.top_p

        prompt_width = int(inputs["input_ids"].shape[1])
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        latency = time.perf_counter() - start_time
        generated_ids = outputs[0][prompt_width:]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        generated_tokens = int(generated_ids.numel())
        total_tokens = int(outputs[0].numel())
        return prediction, latency, generated_tokens, total_tokens

    def evaluate_model_variant(
        self,
        label: str,
        model,
        tokenizer,
        examples: list[WeatherPromptExample],
        schema,
        config: EvaluationConfig,
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        exact_match_values: list[int] = []
        token_f1_values: list[float] = []
        latency_values: list[float] = []
        reward_values: list[float] = []
        accuracy_values: list[float] = []
        hallucination_values: list[float] = []
        a1_values: list[float] = []
        exact_json_values: list[float] = []
        generated_token_values: list[int] = []
        total_token_values: list[int] = []
        tokens_per_second_values: list[float] = []
        reason_counts: dict[str, int] = {}

        for example in examples:
            prediction, latency, generated_tokens, total_tokens = self.generate_prediction(model, tokenizer, example.prompt_text, config)
            reward_value, breakdown = compute_weather_reward(
                prediction=prediction,
                reference=example.reference_text,
                schema=schema,
                config=config.reward_config,
            )

            exact_match = int(norm(prediction) == norm(example.reference_text))
            token_f1_value = token_f1(prediction, example.reference_text)
            tokens_per_second = generated_tokens / latency if latency > 0 else None

            exact_match_values.append(exact_match)
            token_f1_values.append(token_f1_value)
            latency_values.append(latency)
            reward_values.append(reward_value)
            accuracy_values.append(float(breakdown["accuracy"]))
            hallucination_values.append(float(breakdown["hallucinated"]))
            a1_values.append(float(breakdown["a1_score"]))
            exact_json_values.append(float(breakdown["exact_json_match"]))
            generated_token_values.append(generated_tokens)
            total_token_values.append(total_tokens)
            if tokens_per_second is not None:
                tokens_per_second_values.append(tokens_per_second)

            for reason in breakdown["reasons"]:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

            rows.append(
                {
                    "id": example.sample_id,
                    "prompt": example.prompt_text,
                    "reference": example.reference_text,
                    "prediction": prediction,
                    "latency_sec": round(latency, 6),
                    "generated_tokens": generated_tokens,
                    "total_tokens": total_tokens,
                    "tokens_per_second": round(tokens_per_second, 6) if tokens_per_second is not None else None,
                    "reward": round(reward_value, 6),
                    "exact_match": exact_match,
                    "token_f1": round(token_f1_value, 6),
                    "accuracy": round(float(breakdown["accuracy"]), 6),
                    "hallucinated": int(breakdown["hallucinated"]),
                    "a1_score": round(float(breakdown["a1_score"]), 6),
                    "exact_json_match": int(breakdown["exact_json_match"]),
                    "hallucination_reasons": breakdown["reasons"],
                }
            )

        summary = {
            "num_examples": len(rows),
            "exact_match": round(sum(exact_match_values) / len(exact_match_values), 4) if exact_match_values else None,
            "token_f1": round(sum(token_f1_values) / len(token_f1_values), 4) if token_f1_values else None,
            "accuracy": round(statistics.mean(accuracy_values), 4) if accuracy_values else None,
            "hallucination_rate": round(statistics.mean(hallucination_values), 4) if hallucination_values else None,
            "a1_score": round(statistics.mean(a1_values), 4) if a1_values else None,
            "exact_json_match": round(statistics.mean(exact_json_values), 4) if exact_json_values else None,
            "reward": round(statistics.mean(reward_values), 4) if reward_values else None,
            "latency_avg_sec": round(statistics.mean(latency_values), 4) if latency_values else None,
            "latency_median_sec": round(statistics.median(latency_values), 4) if latency_values else None,
            "latency_p95_sec": round(percentile(latency_values, 0.95), 4) if latency_values else None,
            "generated_tokens_avg": round(statistics.mean(generated_token_values), 4) if generated_token_values else None,
            "generated_tokens_median": round(statistics.median(generated_token_values), 4) if generated_token_values else None,
            "total_tokens_avg": round(statistics.mean(total_token_values), 4) if total_token_values else None,
            "tokens_per_second_avg": round(statistics.mean(tokens_per_second_values), 4) if tokens_per_second_values else None,
            "hallucination_reason_counts": dict(sorted(reason_counts.items())),
        }
        return {"label": label, "summary": summary, "predictions": rows}

    def build_model_specs(self, config: EvaluationConfig) -> list[ModelVariantSpec]:
        specs = [ModelVariantSpec("base", None)]
        explicit_paths = {
            "sft_lora": config.sft_lora_adapter,
            "sft_qlora": config.sft_qlora_adapter,
            "ppo_lora": config.ppo_lora_adapter,
            "ppo_qlora": config.ppo_qlora_adapter,
        }
        default_paths = {
            "sft_lora": DEFAULT_OUTPUT_ROOT / "sft-lora",
            "sft_qlora": DEFAULT_OUTPUT_ROOT / "sft-qlora",
            "ppo_lora": DEFAULT_OUTPUT_ROOT / "ppo-lora",
            "ppo_qlora": DEFAULT_OUTPUT_ROOT / "ppo-qlora",
        }
        for label, adapter_value in explicit_paths.items():
            adapter_path = resolve_path(adapter_value, self.project_root) if adapter_value else default_paths[label]
            if adapter_value and not adapter_path.exists():
                raise FileNotFoundError(f"Evaluation adapter path was not found for {label}: {adapter_path}")
            if adapter_path.exists():
                specs.append(ModelVariantSpec(label, adapter_path))
        return specs

    def merge_prediction_rows(self, results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        labels = list(results.keys())
        base_rows = results[labels[0]]["predictions"]
        merged_rows: list[dict[str, Any]] = []
        for row_index, base_row in enumerate(base_rows):
            merged = {
                "id": base_row["id"],
                "prompt": base_row["prompt"],
                "reference": base_row["reference"],
            }
            for label in labels:
                source = results[label]["predictions"][row_index]
                merged[f"{label}_prediction"] = source["prediction"]
                merged[f"{label}_latency_sec"] = source["latency_sec"]
                merged[f"{label}_generated_tokens"] = source["generated_tokens"]
                merged[f"{label}_total_tokens"] = source["total_tokens"]
                merged[f"{label}_reward"] = source["reward"]
                merged[f"{label}_accuracy"] = source["accuracy"]
                merged[f"{label}_hallucinated"] = source["hallucinated"]
                merged[f"{label}_a1_score"] = source["a1_score"]
                merged[f"{label}_exact_json_match"] = source["exact_json_match"]
                merged[f"{label}_exact_match"] = source["exact_match"]
                merged[f"{label}_token_f1"] = source["token_f1"]
                merged[f"{label}_hallucination_reasons"] = source["hallucination_reasons"]
            merged_rows.append(merged)
        return merged_rows

    def evaluate_variants(self, config: EvaluationConfig) -> dict[str, Any]:
        random.seed(config.seed)
        tokenizer = self.load_tokenizer(
            config.tokenizer_path,
            config.base_model_path,
            config.trust_remote_code,
            padding_side="left",
        )
        examples = load_weather_prompt_examples(
            dataset_path=resolve_path(config.dataset_path, self.project_root),
            tokenizer=tokenizer,
            max_samples=config.max_samples,
            shuffle=config.shuffle,
            seed=config.seed,
        )
        schema = infer_schema([{"reference": example.reference_text} for example in examples])

        results: dict[str, dict[str, Any]] = {}
        load_modes: dict[str, str] = {}
        for spec in self.build_model_specs(config):
            load_mode = self.infer_evaluation_load_mode(spec.label, spec.adapter_path, config.base_load_mode)
            load_modes[spec.label] = load_mode
            model = self.load_model_variant(config.base_model_path, spec.adapter_path, load_mode, config.trust_remote_code)
            results[spec.label] = self.evaluate_model_variant(spec.label, model, tokenizer, examples, schema, config)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        report = {
            "config": {
                "base_model": config.base_model_path,
                "tokenizer_path": config.tokenizer_path or config.base_model_path,
                "dataset": str(resolve_path(config.dataset_path, self.project_root)),
                "max_samples": config.max_samples,
                "max_new_tokens": config.max_new_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "seed": config.seed,
                "load_modes": load_modes,
            },
            "metric_definitions": {
                "accuracy": "Mean per-field score against the structured weather reference JSON.",
                "reward": "Average reward under the configured structured weather reward function.",
                "hallucination_rate": "Fraction of generations with formatting or schema violations.",
                "a1_score": "Mean(accuracy * (1 - hallucinated)).",
                "exact_json_match": "Fraction of samples whose JSON matches the reference exactly.",
                "generated_tokens_avg": "Average number of generated completion tokens.",
                "latency_avg_sec": "Average generation latency in seconds.",
            },
            "models": {label: payload["summary"] for label, payload in results.items()},
            "comparisons_vs_base": {},
        }

        base_summary = results["base"]["summary"]
        comparison_metrics = (
            "accuracy",
            "reward",
            "hallucination_rate",
            "a1_score",
            "exact_json_match",
            "exact_match",
            "token_f1",
            "latency_avg_sec",
            "generated_tokens_avg",
            "tokens_per_second_avg",
        )
        for label, payload in results.items():
            if label == "base":
                continue
            report["comparisons_vs_base"][label] = {
                metric_name: (
                    round(payload["summary"][metric_name] - base_summary[metric_name], 4)
                    if isinstance(payload["summary"].get(metric_name), (int, float))
                    and isinstance(base_summary.get(metric_name), (int, float))
                    else None
                )
                for metric_name in comparison_metrics
            }

        report_path = resolve_path(config.output_report, self.project_root)
        save_json(report_path, report)

        if config.output_predictions:
            prediction_rows = self.merge_prediction_rows(results)
            write_jsonl(resolve_path(config.output_predictions, self.project_root), prediction_rows)

        return report
