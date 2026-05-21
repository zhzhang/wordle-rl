"""Minimal GRPO training step.

Given a batch of rollouts (tokens, mask, sampling logprobs, scores), this
module runs one optimizer step on a HuggingFace policy model using a
PPO/GRPO-style clipped surrogate objective with importance-sampling ratios
between the rollout policy (vLLM) and the current trainable policy.

Memory-saving choices for a 4B model on a single GPU:
  * Trainable parameters are restricted to a small LoRA adapter so the
    optimizer state stays tiny.
  * Each rollout is forwarded as its own micro-batch (one sequence per
    forward pass) and the policy-gradient losses are accumulated, giving
    a flexible "effective batch size = number of rollouts".
  * Gradient checkpointing trades compute for activation memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


@dataclass
class GRPOConfig:
    lr: float = 1e-5
    clip_eps: float = 0.2
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    # When `True`, the optimizer trains a LoRA adapter only. Full fine-tuning
    # is much heavier on a 40 GB A100 with a ~4B model so we default to LoRA.
    use_lora: bool = True
    # `dtype` for model weights during training. bf16 is preferred on A100.
    dtype: torch.dtype = torch.bfloat16
    grad_checkpointing: bool = True
    max_grad_norm: float = 1.0


def _normalize_advantages(scores: list[int]) -> list[float]:
    n = len(scores)
    if n == 0:
        return []
    mean = sum(scores) / n
    if n == 1:
        return [0.0]
    var = sum((s - mean) ** 2 for s in scores) / n
    std = var**0.5
    if std == 0:
        return [0.0] * n
    return [(s - mean) / std for s in scores]


def load_policy(model_id: str, config: GRPOConfig, *, device: int | str | None = None):
    """Load the trainable policy model (optionally with a LoRA adapter)."""
    if device is None:
        device_map = "cuda"
    elif isinstance(device, int):
        device_map = {"": f"cuda:{device}"}
    else:
        device_map = {"": device}

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=config.dtype,
        device_map=device_map,
    )
    if config.grad_checkpointing:
        # `use_cache` must be disabled with gradient checkpointing or HF warns
        # and the forward output is wrong for training.
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if config.use_lora:
        lora_cfg = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        for p in model.parameters():
            p.requires_grad_(True)

    model.train()
    return model


def _gather_token_logprobs(
    model,
    tokens: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward `tokens` through `model` and return per-token logprobs of the
    realized tokens together with the alignment mask and per-token entropy.

    Logits at position `t` predict the token at position `t + 1`, so we shift
    inputs/targets/masks accordingly.

    Returns:
        logprobs: shape [T-1], logprob of the realized next token at each
            position whose mask is 1.
        mask: shape [T-1], the shifted target mask (1 = train on this token).
        entropy: shape [T-1], entropy of the policy distribution at each
            position (in nats).
    """
    input_ids = tokens.unsqueeze(0)  # [1, T]
    outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits[0]  # [T, V]

    shift_logits = logits[:-1, :]  # [T-1, V]
    shift_targets = tokens[1:]  # [T-1]
    shift_mask = target_mask[1:]  # [T-1]

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_logprobs = log_probs.gather(-1, shift_targets.unsqueeze(-1)).squeeze(-1)
    # H(p) = -sum_v p(v) * log p(v); computed from log_probs in a numerically
    # stable way as -sum_v exp(log p) * log p.
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return token_logprobs, shift_mask, entropy


def grpo_step(
    model,
    optimizer,
    rollouts,
    advantages: list[float],
    config: GRPOConfig,
) -> dict[str, float]:
    """Run one GRPO optimizer step over a batch of rollouts.

    Each rollout is processed as a micro-batch of size 1; per-token policy
    losses are summed and a single `loss.backward()` is invoked per rollout.
    Gradients accumulate across rollouts, then a single optimizer step is
    taken (effective batch size == number of rollouts).
    """
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)

    total_tokens = 0
    total_loss_acc = 0.0
    total_ratio_acc = 0.0
    total_clip_frac_acc = 0.0
    total_kl_acc = 0.0
    total_entropy_acc = 0.0

    # Pre-compute denominator (total trainable tokens across the group) so
    # each token contributes the same amount to the final loss regardless of
    # which rollout it came from.
    per_rollout_train_tokens: list[int] = []
    for rollout in rollouts:
        per_rollout_train_tokens.append(sum(rollout.mask[1:]))
    grand_total = max(1, sum(per_rollout_train_tokens))

    for rollout, advantage in zip(rollouts, advantages):
        if sum(rollout.mask[1:]) == 0:
            continue

        tokens = torch.tensor(rollout.tokens, dtype=torch.long, device=device)
        mask = torch.tensor(rollout.mask, dtype=torch.float32, device=device)
        old_logprobs_full = torch.tensor(
            rollout.sample_logprobs, dtype=torch.float32, device=device
        )

        new_logprobs, shift_mask, entropy = _gather_token_logprobs(model, tokens, mask)
        # `old_logprobs_full[i]` is the logprob with which the sampler chose
        # `tokens[i]` (the token *at* position i). The new_logprobs returned
        # above are aligned to predict `tokens[i+1]` at index `i`. To match
        # them we shift `old_logprobs_full` by one as well, so both tensors
        # are indexed by the *target* token's position.
        old_logprobs = old_logprobs_full[1:]

        adv = torch.tensor(advantage, dtype=torch.float32, device=device)

        # Importance ratio between current policy and rollout policy.
        log_ratio = new_logprobs - old_logprobs
        ratio = log_ratio.exp()
        clipped_ratio = ratio.clamp(1.0 - config.clip_eps, 1.0 + config.clip_eps)

        per_token_obj = torch.min(ratio * adv, clipped_ratio * adv)
        per_token_loss = -per_token_obj  # we minimise

        # Mask out non-agent tokens AND positions whose sampling logprob is
        # the 0.0 placeholder (these belong to the prompt continuation just
        # added before this turn, not to model-generated tokens).
        loss_mask = shift_mask
        masked_loss = (per_token_loss * loss_mask).sum() / grand_total
        masked_loss.backward()

        with torch.no_grad():
            tok_count = loss_mask.sum().clamp_min(1.0)
            total_loss_acc += float((per_token_loss * loss_mask).sum().item())
            total_ratio_acc += float((ratio * loss_mask).sum().item())
            # Fraction of tokens whose ratio fell outside the clip range.
            clipped_flag = ((ratio < 1.0 - config.clip_eps) |
                            (ratio > 1.0 + config.clip_eps)).float()
            total_clip_frac_acc += float((clipped_flag * loss_mask).sum().item())
            # Approximate KL = E[ -log_ratio ] for diagnostics only.
            total_kl_acc += float((-log_ratio * loss_mask).sum().item())
            total_entropy_acc += float((entropy * loss_mask).sum().item())
            total_tokens += int(tok_count.item())

    # `clip_grad_norm_` returns the total norm of trainable params *before*
    # clipping, which is exactly the diagnostic we want to log. When clipping
    # is disabled we pass `inf` so no clipping happens but we still get the
    # norm back.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    clip_value = config.max_grad_norm if config.max_grad_norm is not None else float("inf")
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, clip_value)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    n = max(1, total_tokens)
    return {
        "loss": total_loss_acc / n,
        "mean_ratio": total_ratio_acc / n,
        "clip_frac": total_clip_frac_acc / n,
        "approx_kl": total_kl_acc / n,
        "entropy": total_entropy_acc / n,
        "grad_norm": float(grad_norm),
        "trained_tokens": float(total_tokens),
    }
