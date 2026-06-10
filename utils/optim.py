from __future__ import annotations

import math

import torch
import torch.nn as nn


def configure_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """AdamW with two param groups:
    - 2D matrices (attention/MLP weights): weight decay applied
    - 1D params (biases, norms, embeddings): no weight decay
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Embeddings are 2D but should not be decayed
        if p.dim() >= 2 and "embedding" not in name:
            decay.append(p)
        else:
            no_decay.append(p)

    groups = [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=True)


def cosine_lr(
    step: int,
    *,
    warmup_steps: int,
    max_steps: int,
    lr: float,
    min_lr: float,
) -> float:
    """Linear warmup → cosine decay → constant min_lr."""
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


def configure_optimizer_mini(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    n_embd: int = 896,
    n_head: int = 14,
    n_query_groups: int = 2,
):
    """Adam-mini: ~50% less optimizer memory than AdamW by sharing LRs within
    parameter blocks. Requires the adam-mini package."""
    from adam_mini import Adam_mini
    return Adam_mini(
        named_parameters=model.named_parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        dim=n_embd,
        n_heads=n_head,
        n_kv_heads=n_query_groups,
    )


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
