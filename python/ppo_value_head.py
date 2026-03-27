from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


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
    """
    Lightweight wrapper that keeps the policy as a regular causal LM while
    adding a scalar value head for PPO updates.
    """

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
        hidden_states = outputs.hidden_states[-1]
        values = self.value_head(hidden_states)
        return {
            "logits": outputs.logits,
            "values": values,
        }

    def generate(self, *args, **kwargs):
        return self.pretrained_model.generate(*args, **kwargs)

    def save_value_head(self, path: Path) -> None:
        torch.save(self.value_head.state_dict(), path)

    def load_value_head(self, path: Path, strict: bool = True) -> None:
        state_dict = torch.load(path, map_location="cpu")
        self.value_head.load_state_dict(state_dict, strict=strict)
