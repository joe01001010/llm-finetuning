from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def _infer_hidden_size(config) -> int:
    for attribute_name in ("hidden_size", "n_embd", "d_model", "model_dim"):
        value = getattr(config, attribute_name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("Unable to infer hidden size from the causal LM config for PPO value head creation.")


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.summary = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.summary(self.dropout(hidden_states)).squeeze(-1)


class CausalLMWithValueHead(nn.Module):
    """
    Thin transformer-native PPO wrapper that keeps the causal LM as the policy
    and adds a scalar value head over the final hidden states.
    """

    def __init__(self, pretrained_model, value_head_dropout: float = 0.1) -> None:
        super().__init__()
        self.pretrained_model = pretrained_model
        hidden_size = _infer_hidden_size(pretrained_model.config)
        self.value_head = ValueHead(hidden_size, dropout=value_head_dropout)

    @property
    def config(self):
        return self.pretrained_model.config

    def forward(self, *args, **kwargs) -> dict[str, Any]:
        kwargs.setdefault("output_hidden_states", True)
        kwargs.setdefault("return_dict", True)
        outputs = self.pretrained_model(*args, **kwargs)
        if outputs.hidden_states is None:
            raise RuntimeError("Policy model did not return hidden states required for PPO value estimation.")
        hidden_states = outputs.hidden_states[-1]
        value_device = next(self.value_head.parameters()).device
        hidden_states = hidden_states.to(value_device)
        values = self.value_head(hidden_states)
        if values.device != outputs.logits.device:
            values = values.to(outputs.logits.device)
        return {"logits": outputs.logits, "values": values, "model_outputs": outputs}

    def generate(self, *args, **kwargs):
        return self.pretrained_model.generate(*args, **kwargs)

    def save_value_head(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.value_head.state_dict(), destination)

    def load_value_head(self, path: str | Path, strict: bool = True) -> None:
        state_dict = torch.load(Path(path), map_location="cpu")
        self.value_head.load_state_dict(state_dict, strict=strict)
