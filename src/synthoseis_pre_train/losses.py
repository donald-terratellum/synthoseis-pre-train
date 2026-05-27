"""Loss functions for seismic pre-training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMHybridLoss3D(nn.Module):
    """Hybrid 3D loss: w1 * (1 - SSIM) + w2 * MSE + w3 * L1.

    Design notes:
    - SSIM is implemented for 5D tensors shaped [N, C, D, H, W].
    - The default SSIM window is cubic 7x7x7 to match this codebase's 3D volumes.
    - SSIM constants follow common practice from the original SSIM paper and
      popular implementations (K1=0.01, K2=0.03) with data range L=1.

    Input range normalization assumption:
    - This codebase commonly uses approximately zero-centered amplitudes in an
      approximate range of [-10, 10].
    - Before SSIM, values are mapped to [0, 1] via x01 = clamp(x / 20 + 0.5, 0, 1).
      This centers zero at 0.5 and matches L=1 constants.

    Revisit this normalization if:
    - observed train/target amplitude range deviates materially from [-10, 10],
    - clipping to [0, 1] becomes frequent,
    - data preprocessing changes (e.g., robust scaling, per-volume normalization),
    - SSIM term collapses or dominates due to range mismatch.
    """

    def __init__(
        self,
        window_size: int = 7,
        w1: float = 1.0,
        w2: float = 0.0,
        w3: float = 0.0,
        k1: float = 0.01,
        k2: float = 0.03,
        eps: float = 1e-8,
    ):
        super().__init__()
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer >= 3")
        if w1 < 0 or w2 < 0 or w3 < 0:
            raise ValueError("w1, w2, and w3 must be >= 0")
        if k1 <= 0 or k2 <= 0:
            raise ValueError("k1 and k2 must be > 0")

        self.window_size = int(window_size)
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.w3 = float(w3)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.eps = float(eps)

        kernel = self._gaussian_kernel_3d(self.window_size, sigma=self.window_size / 6.0)
        self.register_buffer("kernel", kernel, persistent=False)

    @staticmethod
    def _gaussian_kernel_3d(window_size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
        g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
        g = g / g.sum().clamp_min(1e-12)
        k3 = g[:, None, None] * g[None, :, None] * g[None, None, :]
        k3 = k3 / k3.sum().clamp_min(1e-12)
        return k3.view(1, 1, window_size, window_size, window_size)

    @staticmethod
    def _to_unit_interval(x: torch.Tensor) -> torch.Tensor:
        # Expected raw amplitude range is approximately [-10, 10].
        # Map to [0, 1] and center 0 at 0.5 for SSIM with L=1 constants.
        return torch.clamp(x / 20.0 + 0.5, 0.0, 1.0)

    def _ssim_3d(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError("pred and target must have identical shape")
        if pred.ndim != 5:
            raise ValueError("expected tensors shaped [N, C, D, H, W]")

        _, channels, depth, height, width = pred.shape
        ws = self.window_size
        if min(depth, height, width) < ws:
            raise ValueError(
                f"ssim window_size ({ws}) must be <= each spatial dimension "
                f"(got D,H,W={depth},{height},{width})"
            )

        x = self._to_unit_interval(pred.float())
        y = self._to_unit_interval(target.float())

        kernel = self.kernel.to(device=x.device, dtype=x.dtype).repeat(channels, 1, 1, 1, 1)
        padding = ws // 2

        mu_x = F.conv3d(x, kernel, padding=padding, groups=channels)
        mu_y = F.conv3d(y, kernel, padding=padding, groups=channels)

        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv3d(x * x, kernel, padding=padding, groups=channels) - mu_x_sq
        sigma_y_sq = F.conv3d(y * y, kernel, padding=padding, groups=channels) - mu_y_sq
        sigma_xy = F.conv3d(x * y, kernel, padding=padding, groups=channels) - mu_xy

        c1 = (self.k1 * 1.0) ** 2
        c2 = (self.k2 * 1.0) ** 2

        numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        ssim_map = numerator / (denominator + self.eps)

        return ssim_map.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ssim_score = self._ssim_3d(pred, target)
        ssim_term = 1.0 - ssim_score
        mse_term = F.mse_loss(pred, target)
        l1_term = F.l1_loss(pred, target)
        return self.w1 * ssim_term + self.w2 * mse_term + self.w3 * l1_term
