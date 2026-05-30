"""Training utilities for loss-function comparison experiments."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Iterable

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer


def select_study_device(prefer_mps: bool = True) -> torch.device:
    """Select a device for local study runs.

    On Apple Silicon, this prefers Metal (`mps`) when available and otherwise
    falls back to CPU. This intentionally avoids CUDA so study scripts run
    predictably on an M4 Mac mini.

    Args:
        prefer_mps: Whether to prefer MPS when available.

    Returns:
        Selected torch device.
    """
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _device_type(device: torch.device) -> str:
    """Return normalized device type string for AMP decisions."""
    if isinstance(device, torch.device):
        return device.type
    return str(device)


def _resolve_device(device: torch.device | str | None) -> torch.device:
    """Resolve caller-provided device or choose study default automatically."""
    if device is None:
        return select_study_device(prefer_mps=True)
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def train_epoch(
    model: Module,
    dataloader: Iterable[Dict[str, Tensor]],
    optimizer: Optimizer,
    device: torch.device | str | None,
    loss_fn: Module,
    grad_accum_steps: int = 1,
) -> float:
    """Train one epoch with optional gradient accumulation.

    This function is Apple Silicon friendly: it uses AMP + GradScaler only on
    CUDA, while MPS/CPU execute in full precision.

    Args:
        model: PyTorch model to train.
        dataloader: Iterable producing dict batches with keys ``input``,
            ``target``, and ``mask``.
        optimizer: Optimizer instance.
        device: Target device. If ``None``, automatically selects MPS when
            available, otherwise CPU.
        loss_fn: Loss module returning ``(loss, components)``.
        grad_accum_steps: Number of mini-batches to accumulate before stepping.

    Returns:
        Mean epoch loss over all loader batches.
    """
    device = _resolve_device(device)
    model.train()
    device_type = _device_type(device)
    use_cuda_amp = device_type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(dataloader):
        num_batches += 1
        x = batch["input"].to(device)
        y = batch["target"].to(device)
        m = batch["mask"].to(device)

        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_cuda_amp
            else nullcontext()
        )
        with amp_context:
            pred = model(x)
            loss, _components = loss_fn(pred, y, m)
            loss = loss / grad_accum_steps

        if use_cuda_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            if use_cuda_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * grad_accum_steps

    return total_loss / max(num_batches, 1)
