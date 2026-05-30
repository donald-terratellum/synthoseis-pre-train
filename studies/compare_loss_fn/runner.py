"""Runner entrypoint for compare-loss-function studies.

This module provides a single helper that moves model/loss to the selected
study device and trains for one epoch. By default it auto-selects MPS on Apple
Silicon and falls back to CPU.
"""

from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

try:
    from .train_loop import select_study_device, train_epoch
except ImportError:  # pragma: no cover - supports direct script execution.
    from train_loop import select_study_device, train_epoch


def run_compare_loss_epoch(
    model: Module,
    dataloader: Iterable[Dict[str, Tensor]],
    optimizer: Optimizer,
    loss_fn: Module,
    device: torch.device | str | None = None,
    grad_accum_steps: int = 1,
) -> tuple[float, torch.device]:
    """Train one compare-loss epoch using automatic device selection.

    Args:
        model: Model to train.
        dataloader: Iterable yielding ``input``, ``target``, ``mask`` tensors.
        optimizer: Optimizer used for updates.
        loss_fn: Composite loss callable returning ``(loss, components)``.
        device: Explicit device override. If ``None``, picks MPS then CPU.
        grad_accum_steps: Gradient accumulation factor.

    Returns:
        Tuple ``(mean_loss, selected_device)``.
    """
    selected_device = select_study_device(prefer_mps=True) if device is None else torch.device(device)
    model = model.to(selected_device)
    loss_fn = loss_fn.to(selected_device)

    mean_loss = train_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=selected_device,
        loss_fn=loss_fn,
        grad_accum_steps=grad_accum_steps,
    )
    return mean_loss, selected_device
