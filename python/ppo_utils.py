from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


COMMON_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


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
    """
    Resolve a model/tokenizer identifier into either a verified local path or a
    Hub repo ID. Local-looking paths are validated eagerly so callers get a
    helpful error instead of a Hugging Face repo-id validation traceback.
    """
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


def maybe_to_float(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
