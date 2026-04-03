from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterator

import torch


@dataclass
class RolloutBatch:
    prompt_width: int
    sample_ids: list[str]
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

    def iter_minibatches(self, mini_batch_size: int, shuffle: bool = True) -> Iterator[dict[str, Any]]:
        if mini_batch_size <= 0:
            raise ValueError("mini_batch_size must be positive.")

        indices = list(range(self.batch_size))
        if shuffle:
            random.shuffle(indices)

        tensor_fields = (
            "prompt_input_ids",
            "prompt_attention_mask",
            "sequence_ids",
            "sequence_attention_mask",
            "response_ids",
            "response_mask",
            "old_logprobs",
            "ref_logprobs",
            "old_values",
            "rewards",
            "advantages",
            "returns",
            "score_rewards",
            "kl_divergence",
            "response_lengths",
            "eos_flags",
        )
        list_fields = (
            "sample_ids",
            "prompt_texts",
            "reference_texts",
            "response_texts",
            "reward_breakdowns",
        )

        for start_index in range(0, self.batch_size, mini_batch_size):
            batch_indices = indices[start_index : start_index + mini_batch_size]
            index_tensor = torch.tensor(batch_indices, device=self.sequence_ids.device, dtype=torch.long)
            minibatch: dict[str, Any] = {}
            for field_name in tensor_fields:
                minibatch[field_name] = getattr(self, field_name).index_select(0, index_tensor)
            for field_name in list_fields:
                field_value = getattr(self, field_name)
                minibatch[field_name] = [field_value[index] for index in batch_indices]
            yield minibatch
