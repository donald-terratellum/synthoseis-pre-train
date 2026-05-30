"""Composite masked losses for 3D seismic reconstruction studies."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from .losses_ssim3d import MSSSIM3DMasked
except ImportError:  # pragma: no cover - supports direct script execution.
    from losses_ssim3d import MSSSIM3DMasked


class SeismicCompositeLoss(nn.Module):
    """Weighted sum of masked MS-SSIM, MSE, TV, and spectral losses.

    Total loss is:
    ``alpha*(1-ms_ssim) + beta*mse + gamma*tv + delta*fft``.
    """

    def __init__(
        self,
        alpha_ssim: float = 0.6,
        beta_mse: float = 0.2,
        gamma_tv: float = 0.1,
        delta_fft: float = 0.1,
    ) -> None:
        """Initialize loss weights and internal MS-SSIM module."""
        super().__init__()
        self.ms_ssim = MSSSIM3DMasked()
        self.alpha = alpha_ssim
        self.beta = beta_mse
        self.gamma = gamma_tv
        self.delta = delta_fft

    def mse_masked(self, x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        """Compute masked mean squared error."""
        diff2 = (x - y) ** 2 * mask
        return diff2.sum() / mask.sum().clamp(min=1)

    def _cubic_smooth3d(self, x: Tensor) -> Tensor:
        """Apply a separable cubic-style smoothing filter in 3D.

        Uses the 1D kernel ``[1, 4, 1] / 6`` on each axis as a compact
        cubic-like smoother, applied depthwise per channel.
        """
        channels = int(x.shape[1])
        k = torch.tensor([1.0, 4.0, 1.0], device=x.device, dtype=x.dtype) / 6.0
        kernel = (k[:, None, None] * k[None, :, None] * k[None, None, :]).view(1, 1, 3, 3, 3)
        kernel = kernel.expand(channels, 1, 3, 3, 3)
        return F.conv3d(x, kernel, padding=1, groups=channels)

    def mae_smooth_masked(self, x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        """Compute MAE after cubic smoothing both prediction and target."""
        x_s = self._cubic_smooth3d(x)
        y_s = self._cubic_smooth3d(y)
        diff = (x_s - y_s).abs() * mask
        return diff.sum() / mask.sum().clamp(min=1)

    def tv_loss(self, x: Tensor, mask: Tensor) -> Tensor:
        """Compute masked anisotropic total variation over 3D neighbors."""
        dz = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
        dy = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
        dx = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])

        mz = mask[:, :, 1:, :, :] * mask[:, :, :-1, :, :]
        my = mask[:, :, :, 1:, :] * mask[:, :, :, :-1, :]
        mx = mask[:, :, :, :, 1:] * mask[:, :, :, :, :-1]

        tv = (dz * mz).sum() + (dy * my).sum() + (dx * mx).sum()
        denom = (mz.sum() + my.sum() + mx.sum()).clamp(min=1)
        return tv / denom

    def fft_loss(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute mean squared magnitude mismatch in 3D Fourier space."""
        x_fft = torch.fft.fftn(x, dim=(-3, -2, -1))
        y_fft = torch.fft.fftn(y, dim=(-3, -2, -1))
        diff2 = (torch.abs(x_fft) - torch.abs(y_fft)) ** 2
        return diff2.mean()

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute total composite loss and detached component diagnostics.

        Args:
            pred: Predicted tensor with shape ``(B, C, D, H, W)``.
            target: Target tensor with shape ``(B, C, D, H, W)``.
            mask: Valid-data mask with shape ``(B, 1, D, H, W)``.

        Returns:
            Tuple of ``(total_loss, components_dict)``.
        """
        ms_ssim_val = self.ms_ssim(pred, target, mask)
        mse_val = self.mse_masked(pred, target, mask)
        tv_val = self.tv_loss(pred, mask)
        fft_val = self.fft_loss(pred, target)

        loss = (
            self.alpha * (1.0 - ms_ssim_val)
            + self.beta * mse_val
            + self.gamma * tv_val
            + self.delta * fft_val
        )

        return loss, {
            "ms_ssim": ms_ssim_val.detach(),
            "mse": mse_val.detach(),
            "tv": tv_val.detach(),
            "fft": fft_val.detach(),
        }
