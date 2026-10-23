from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor          # [N, L]
    attention_mask: torch.Tensor     # [N, L]
    completion_mask: torch.Tensor    # [N, L-1] float
    old_logprobs: torch.Tensor       # [N, L-1]
    ref_logprobs: torch.Tensor       # [N, L-1]
    rewards: torch.Tensor            # [N]
    advantages: torch.Tensor         # [N]

    task_names: Optional[list] = None
    completion_texts: Optional[list] = None

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            completion_mask=self.completion_mask.to(device, non_blocking=True),
            old_logprobs=self.old_logprobs.to(device, non_blocking=True),
            ref_logprobs=self.ref_logprobs.to(device, non_blocking=True),
            rewards=self.rewards.to(device, non_blocking=True),
            advantages=self.advantages.to(device, non_blocking=True),
            task_names=self.task_names,
            completion_texts=self.completion_texts,
        )


def iter_minibatches(
    batch: RolloutBatch,
    minibatch_size: int,
    shuffle: bool = True,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Iterator[RolloutBatch]:
    # del batch, minibatch_size, shuffle, generator, device
    # TODO(student): iterate over the rollout in minibatches, optionally shuffling the row indices,
    # and yield RolloutBatch objects containing the selected subset.

    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")

    n = batch.input_ids.size(0)
    if n == 0:
        return

    if shuffle:
        indices = torch.randperm(n, generator=generator).to(batch.input_ids.device)
    else:
        indices = torch.arange(n, device=batch.input_ids.device)

    for start in range(0, n, minibatch_size):
        idx = indices[start : start + minibatch_size]
        idx_list = idx.tolist()

        minibatch = RolloutBatch(
            input_ids=batch.input_ids[idx],
            attention_mask=batch.attention_mask[idx],
            completion_mask=batch.completion_mask[idx],
            old_logprobs=batch.old_logprobs[idx],
            ref_logprobs=batch.ref_logprobs[idx],
            rewards=batch.rewards[idx],
            advantages=batch.advantages[idx],
            task_names=None
            if batch.task_names is None
            else [batch.task_names[i] for i in idx_list],
            completion_texts=None
            if batch.completion_texts is None
            else [batch.completion_texts[i] for i in idx_list],
        )

        if device is not None:
            minibatch = minibatch.to(device)

        yield minibatch
    
    # raise NotImplementedError("Implement iter_minibatches in the student starter.")
