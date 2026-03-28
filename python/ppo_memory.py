from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import torch


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
