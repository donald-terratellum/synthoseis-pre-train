
"""Loss functions for seismic pre-training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SMAELoss(nn.Module):
    """Smooth MAE (SMAE) loss for regression.

    SMAE(e) = e * tanh(e / 2), where e = pred - target
    Reference: https://arxiv.org/pdf/2303.09935

    modification:
    - The original paper defines SMAE as e * tanh(e), but this implementation uses e * tanh(e / 2) to reduce gradient saturation for large errors.
    - This change maintains the same qualitative behavior while providing stronger gradients for large errors, which may
    SMAE(e) = e * tanh(e), where e = pred - target
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError("pred and target must have identical shape")
        e = pred - target
        loss = e * torch.tanh(e) * 2.5  # Scale factor to roughly match MSE magnitude
        return loss.mean()


class MAESmoothLoss3D(nn.Module):
    """MAE between smoothed prediction and target volumes.

    The same separable 3D kernel is applied to both tensors, then standard
    unmasked L1/MAE is computed.
    """

    def __init__(self, kernel_weights: list[float] | tuple[float, ...] = (1.0, 2.0, 1.0)):
        super().__init__()
        if len(kernel_weights) < 3 or len(kernel_weights) % 2 == 0:
            raise ValueError("kernel_weights must contain an odd number of values >= 3")
        if any(w < 0 for w in kernel_weights):
            raise ValueError("kernel_weights values must be >= 0")
        if float(sum(kernel_weights)) <= 0.0:
            raise ValueError("kernel_weights sum must be > 0")

        w = torch.tensor(kernel_weights, dtype=torch.float32)
        w = w / w.sum().clamp_min(1e-12)
        self.kernel_size = int(w.numel())
        self.padding = self.kernel_size // 2

        kernel = (w[:, None, None] * w[None, :, None] * w[None, None, :]).contiguous()
        self.register_buffer("kernel", kernel.view(1, 1, self.kernel_size, self.kernel_size, self.kernel_size), persistent=False)

    def _smooth3d(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("expected tensors shaped [N, C, D, H, W]")
        channels = int(x.shape[1])
        kernel_buf = self.get_buffer("kernel")
        kernel = kernel_buf.to(device=x.device, dtype=x.dtype).repeat(channels, 1, 1, 1, 1)
        return F.conv3d(x, kernel, padding=self.padding, groups=channels)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError("pred and target must have identical shape")
        pred_s = self._smooth3d(pred)
        target_s = self._smooth3d(target)
        return F.l1_loss(pred_s, target_s)


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
        k1: float = 0.1,
        k2: float = 0.3,
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

        kernel_buf = self.get_buffer("kernel")
        kernel = kernel_buf.to(device=x.device, dtype=x.dtype).repeat(channels, 1, 1, 1, 1)
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


class SlidingWindowStatsLoss3D(nn.Module):
    """3D local-statistics loss using sliding-window moments and extrema.

    The total loss is a weighted sum of six terms:
    - local mean MAE
    - local std-ratio penalty
    - local minima MAE
    - local maxima MAE
    - voxelwise MAE
    - voxelwise MSE

    Mask behavior:
    - ``mask_mode='none'`` (default): ignore masks and use all voxels.
    - ``mask_mode='valid'``: if ``valid_mask`` is provided in ``forward``, use it.
    """

    def __init__(
        self,
        window_size: tuple[int, int, int] = (9, 9, 9),
        mean_weight: float = 1.0,
        std_weight: float = 1.0,
        min_weight: float = 1.0,
        max_weight: float = 1.0,
        mae_weight: float = 1.0,
        mse_weight: float = 1.0,
        eps: float = 1e-6,
        std_ratio_clip: float = 10.0,
        mask_mode: str = "none",
    ) -> None:
        super().__init__()
        if len(window_size) != 3:
            raise ValueError("window_size must be a 3-tuple like (9, 9, 9)")
        if any(int(k) <= 0 for k in window_size):
            raise ValueError("window_size entries must be positive")
        if (
            mean_weight < 0
            or std_weight < 0
            or min_weight < 0
            or max_weight < 0
            or mae_weight < 0
            or mse_weight < 0
        ):
            raise ValueError("all component weights must be non-negative")
        if eps <= 0:
            raise ValueError("eps must be > 0")
        if std_ratio_clip <= 1.0:
            raise ValueError("std_ratio_clip must be > 1")
        mask_mode_norm = str(mask_mode).strip().lower()
        if mask_mode_norm not in {"none", "valid"}:
            raise ValueError("mask_mode must be one of: none, valid")

        self.window_size = tuple(int(k) for k in window_size)
        self.mean_weight = float(mean_weight)
        self.std_weight = float(std_weight)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.mae_weight = float(mae_weight)
        self.mse_weight = float(mse_weight)
        self.eps = float(eps)
        self.std_ratio_clip = float(std_ratio_clip)
        self.mask_mode = mask_mode_norm

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        kz, ky, kx = self.window_size
        pz_l = (kz - 1) // 2
        pz_r = kz - 1 - pz_l
        py_l = (ky - 1) // 2
        py_r = ky - 1 - py_l
        px_l = (kx - 1) // 2
        px_r = kx - 1 - px_l
        return F.pad(x, (px_l, px_r, py_l, py_r, pz_l, pz_r), mode="replicate")

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(self._pad(x), kernel_size=self.window_size, stride=1)

    def _masked_local_moments(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wsum = self._pool(valid_mask).clamp_min(self.eps)
        mean = self._pool(x * valid_mask) / wsum
        ex2 = self._pool((x * x) * valid_mask) / wsum
        var = (ex2 - mean * mean).clamp_min(0.0)
        std = torch.sqrt(var + self.eps)
        return mean, std

    def _masked_local_extrema(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_min = x.amin()
        x_max = x.amax()
        span = (x_max - x_min).abs() + 1.0
        very_pos = x_max + 10.0 * span
        very_neg = x_min - 10.0 * span

        valid_bool = valid_mask > 0.5
        x_for_max = torch.where(valid_bool, x, very_neg)
        x_for_min = torch.where(valid_bool, x, very_pos)

        local_max = F.max_pool3d(self._pad(x_for_max), kernel_size=self.window_size, stride=1)
        local_min = -F.max_pool3d(self._pad(-x_for_min), kernel_size=self.window_size, stride=1)

        support = F.max_pool3d(self._pad(valid_mask), kernel_size=self.window_size, stride=1)
        support = (support > 0).to(dtype=x.dtype)
        local_max = torch.where(support > 0, local_max, torch.zeros_like(local_max))
        local_min = torch.where(support > 0, local_min, torch.zeros_like(local_min))
        return local_min, local_max, support

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError("pred and target must have identical shape")
        if pred.ndim != 5:
            raise ValueError("expected tensors shaped [B, C, D, H, W]")

        pred32 = pred.float()
        target32 = target.float()

        if self.mask_mode == "valid" and valid_mask is not None:
            if valid_mask.shape != pred.shape:
                raise ValueError("valid_mask must match pred shape")
            mask = valid_mask.to(dtype=torch.float32)
            use_mask = True
        else:
            mask = torch.ones_like(pred32, dtype=torch.float32)
            use_mask = False

        mean_pred, std_pred = self._masked_local_moments(pred32, mask)
        mean_tgt, std_tgt = self._masked_local_moments(target32, mask)
        min_pred, max_pred, extrema_support = self._masked_local_extrema(pred32, mask)
        min_tgt, max_tgt, _ = self._masked_local_extrema(target32, mask)

        mean_mae = torch.abs(mean_pred - mean_tgt)

        std_ratio = (std_tgt + self.eps) / (std_pred + self.eps)
        std_ratio = torch.clamp(std_ratio, 1.0 / self.std_ratio_clip, self.std_ratio_clip)
        std_ratio_penalty = torch.abs(std_ratio - 1.0)

        min_mae = torch.abs(min_pred - min_tgt)
        max_mae = torch.abs(max_pred - max_tgt)

        mae_voxel = torch.abs(pred32 - target32)
        mse_voxel = (pred32 - target32) ** 2

        if not use_mask:
            l_mean = mean_mae.mean(dtype=torch.float32)
            l_std = std_ratio_penalty.mean(dtype=torch.float32)
            l_min = min_mae.mean(dtype=torch.float32)
            l_max = max_mae.mean(dtype=torch.float32)
            l_mae = mae_voxel.mean(dtype=torch.float32)
            l_mse = mse_voxel.mean(dtype=torch.float32)
        else:
            denom = mask.sum(dtype=torch.float32).clamp_min(1.0)
            l_mean = (mean_mae * mask).sum(dtype=torch.float32) / denom
            l_std = (std_ratio_penalty * mask).sum(dtype=torch.float32) / denom
            extrema_denom = extrema_support.sum(dtype=torch.float32).clamp_min(1.0)
            l_min = (min_mae * extrema_support).sum(dtype=torch.float32) / extrema_denom
            l_max = (max_mae * extrema_support).sum(dtype=torch.float32) / extrema_denom
            l_mae = (mae_voxel * mask).sum(dtype=torch.float32) / denom
            l_mse = (mse_voxel * mask).sum(dtype=torch.float32) / denom

        return (
            self.mean_weight * l_mean
            + self.std_weight * l_std
            + self.min_weight * l_min
            + self.max_weight * l_max
            + self.mae_weight * l_mae
            + self.mse_weight * l_mse
        )
