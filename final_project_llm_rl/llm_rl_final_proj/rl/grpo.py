from __future__ import annotations

from typing import Dict

import torch

from llm_rl_final_proj.models.load import PolicyModel
from llm_rl_final_proj.rl.base import RLAlgorithm
from llm_rl_final_proj.rollout.rollout_buffer import RolloutBatch


class GRPO(RLAlgorithm):
    """GRPO update with a PPO-style clipped surrogate over completion tokens."""

    name = "grpo"

    def update(
        self,
        model: PolicyModel,
        optimizer: torch.optim.Optimizer,
        rollout: RolloutBatch,
        grad_accum_steps: int = 1,
    ) -> Dict[str, float]:
        # del model, optimizer, rollout, grad_accum_steps
        # TODO(student): implement one GRPO training iteration.
        # The intended structure is:
        #   1. loop over PPO epochs,
        #   2. iterate over rollout minibatches,
        #   3. recompute token log-probabilities under the current policy,
        #   4. form PPO ratios against mb.old_logprobs,
        #   5. apply token-level clipping with the sequence-level GRPO averaging used in this codebase,
        #   6. add KL regularization against mb.ref_logprobs,
        #   7. handle gradient accumulation / clipping / optimizer steps,
        #   8. return the logged metrics expected by the training script.

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
                new_logprobs = _compute_new_logprobs(model, mb)      # [B, L-1]
                old_logprobs = mb.old_logprobs.float()
                ref_logprobs = mb.ref_logprobs.float()
                mask = mb.completion_mask.float()
                adv = mb.advantages.float().clamp(-cfg.adv_clip, cfg.adv_clip)  # [B]

                log_ratio = new_logprobs - old_logprobs
                ratio = torch.exp(log_ratio)

                low = 1.0 - cfg.clip_eps
                high = 1.0 + (cfg.clip_eps_high if cfg.clip_eps_high > 0.0 else cfg.clip_eps)
                clipped_ratio = torch.clamp(ratio, low, high)

                surr1 = ratio * adv.unsqueeze(-1)
                surr2 = clipped_ratio * adv.unsqueeze(-1)
                token_loss = -torch.min(surr1, surr2)               # [B, L-1]

                # GRPO: 每个 response 内对 completion token 平均，再对 batch 平均
                seq_loss = (token_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
                policy_loss = seq_loss.mean()

                # sampled-token KL proxy: exp(Δ) - Δ - 1, Δ = log π_ref - log π_θ
                delta = ref_logprobs - new_logprobs
                kl_per_token = torch.expm1(delta) - delta           # = exp(Δ)-1-Δ
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

        # 处理最后不足一个 grad_accum 的累积
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

        # raise NotImplementedError("Implement GRPO.update in the student starter.")
