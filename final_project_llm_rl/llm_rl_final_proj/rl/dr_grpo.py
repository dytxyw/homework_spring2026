from __future__ import annotations

from typing import Dict

import torch

from llm_rl_final_proj.models.load import PolicyModel
from llm_rl_final_proj.rl.base import RLAlgorithm
from llm_rl_final_proj.rollout.rollout_buffer import RolloutBatch


class DrGRPO(RLAlgorithm):
    """DrGRPO removes the GRPO sequence-length normalization and std scaling."""

    name = "dr_grpo"

    def update(
        self,
        model: PolicyModel,
        optimizer: torch.optim.Optimizer,
        rollout: RolloutBatch,
        grad_accum_steps: int = 1,
    ) -> Dict[str, float]:
        # del model, optimizer, rollout, grad_accum_steps
        # TODO(student): implement DrGRPO.
        # Start from your GRPO implementation, then make the two intended changes:
        #   1. use the DrGRPO advantage convention (configured in online/train_rm_grpo.py),
        #   2. remove the per-sequence length normalization inside the surrogate.

        model.train()
        device = next(model.parameters()).device
        cfg = self.cfg
        grad_accum_steps = max(1, grad_accum_steps)

        optimizer.zero_grad(set_to_none=True)

        accum_count = 0
        num_mb = 0
        total_steps = 0
        sum_loss = sum_pg = sum_kl = sum_clipfrac = sum_ratio = sum_gn = 0.0

        for _epoch in range(cfg.ppo_epochs):
            seed = self._next_update_seed()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)

            for mb in iter_minibatches(
                rollout,
                cfg.minibatch_size,
                shuffle=True,
                generator=generator,
                device=device,
            ):
                new_logprobs = _compute_new_logprobs(model, mb)
                old_logprobs = mb.old_logprobs.float()
                ref_logprobs = mb.ref_logprobs.float()
                mask = mb.completion_mask.float()
                adv = mb.advantages.float().clamp(-cfg.adv_clip, cfg.adv_clip)

                ratio = torch.exp(new_logprobs - old_logprobs)

                low = 1.0 - cfg.clip_eps
                high = 1.0 + (cfg.clip_eps_high if cfg.clip_eps_high > 0.0 else cfg.clip_eps)
                clipped_ratio = torch.clamp(ratio, low, high)

                token_loss = -torch.min(
                    ratio * adv.unsqueeze(-1),
                    clipped_ratio * adv.unsqueeze(-1),
                )

                # DrGRPO: 不除以每个 response 的长度，除以固定常数 B * L_max
                max_len = mask.size(1)
                policy_loss = (token_loss * mask).sum() / (mask.size(0) * max_len)

                delta = ref_logprobs - new_logprobs
                kl_per_token = torch.expm1(delta) - delta
                kl = (kl_per_token * mask).sum() / mask.sum().clamp_min(1.0)

                loss = policy_loss + cfg.kl_coef * kl
                (loss / grad_accum_steps).backward()
                accum_count += 1

                with torch.no_grad():
                    clipped = ((ratio < low) | (ratio > high)).float()
                    clipfrac = (clipped * mask).sum() / mask.sum().clamp_min(1.0)
                    ratio_mean = (ratio * mask).sum() / mask.sum().clamp_min(1.0)
                    sum_loss += loss.item()
                    sum_pg += policy_loss.item()
                    sum_kl += kl.item()
                    sum_clipfrac += clipfrac.item()
                    sum_ratio += ratio_mean.item()
                    num_mb += 1

                if accum_count % grad_accum_steps == 0:
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    sum_gn += float(gn.item())
                    total_steps += 1

        if accum_count % grad_accum_steps != 0:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            sum_gn += float(gn.item())
            total_steps += 1

        n = max(1, num_mb)
        return {
            "loss": sum_loss / n,
            "policy_loss": sum_pg / n,
            "kl": sum_kl / n,
            "clipfrac": sum_clipfrac / n,
            "ratio_mean": sum_ratio / n,
            "grad_norm": sum_gn / max(1, total_steps),
        }

        # raise NotImplementedError("Implement DrGRPO.update in the student starter.")
