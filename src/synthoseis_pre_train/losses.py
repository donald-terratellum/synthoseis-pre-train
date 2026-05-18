"""Loss functions for seismic pre-training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rescale_zero_centered_to_unit(x: torch.Tensor, data_range: float) -> torch.Tensor:
    """Map zero-centered amplitudes to [0, 1] using configured data range.

    Assumes nominal input support is approximately [-data_range/2, data_range/2].
    Values outside this range are clamped for SSIM stability.
    """
    y = x / float(data_range) + 0.5
    return torch.clamp(y, 0.0, 1.0)


class SSIMMSELoss3D(nn.Module):
    """Mixed zero-mean SSIM and MSE loss for 3D seismic volumes.

    Purpose:
        Combine voxelwise error and structural similarity into a single loss
        suitable for 3D seismic reconstruction.

    Method:
        This implementation uses a local, separable Gaussian window in 3D and
        assumes the signal is centered around zero mean. It therefore uses only
        second-order local moments (x^2, y^2, x*y) and does not estimate local
        means.

    Args:
        data_range: Dynamic range used to set the SSIM stabilization constant.
            For standardized seismic amplitudes, values near 30.0 correspond to
            an approximate range of [-15, 15].
        window_size: Edge length of the 3D Gaussian window used for local SSIM
            statistics.
        sigma: Standard deviation of the Gaussian window, in voxel units.
        alpha: Blend factor in [0, 1] for combining MSE and the SSIM-derived
            component. alpha=0 gives pure MSE, alpha=1 gives pure SSIM.
        min_valid_ratio: Minimum local fraction of valid voxels required for a
            neighborhood to contribute to the SSIM aggregation.
        eps: Small positive constant used to avoid division by zero.

    Returns:
        A loss module whose forward pass returns a scalar tensor >= 0 where
        lower is better.

        total_loss = (1 - alpha) * mse + alpha * ssim_component

        with ssim_component = 0.5 * (1 - ssim_score), so perfect SSIM gives 0.
    """

    def __init__(
        self,
        data_range: float = 30.0,
        window_size: int = 16,
        sigma: float = 16.0 / 6.0,
        alpha: float = 1.0 / 6.0,
        min_valid_ratio: float = 0.5,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.data_range = float(data_range)
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.alpha = float(alpha)
        self.min_valid_ratio = float(min_valid_ratio)
        self.eps = float(eps)
        self.k2 = 0.03

        if self.window_size < 3:
            raise ValueError("window_size must be >= 3")
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")
        if self.data_range <= 0:
            raise ValueError("data_range must be > 0")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        if not (0.0 <= self.min_valid_ratio <= 1.0):
            raise ValueError("min_valid_ratio must be in [0, 1]")

        kernel = self._gaussian_1d(self.window_size, self.sigma)
        self.register_buffer("kernel_1d", kernel, persistent=False)

    @staticmethod
    def _gaussian_1d(size: int, sigma: float) -> torch.Tensor:
        """Build a normalized 1D Gaussian kernel.

        Purpose:
            Create the separable smoothing kernel used by the 3D SSIM window.

        Method:
            Samples a discrete Gaussian centered at the midpoint of the window,
            then normalizes the kernel so its coefficients sum to 1.

        Args:
            size: Number of samples in the 1D kernel.
            sigma: Standard deviation of the Gaussian in sample units.

        Returns:
            A float32 tensor of shape [size] whose sum is 1.
        """
        # Even window sizes are supported; center sits between two samples.
        center = (size - 1) / 2.0
        coords = torch.arange(size, dtype=torch.float32) - center
        kernel = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
        kernel = kernel / kernel.sum().clamp_min(1e-12)
        return kernel

    def _pad_same(self, x: torch.Tensor, dim: str) -> torch.Tensor:
        """Apply replicate padding for same-sized 3D convolution along one axis.

        Purpose:
            Preserve input spatial dimensions when applying one axis of the
            separable Gaussian blur.

        Method:
            Computes asymmetric left/right padding for the configured window
            size and applies replicate padding only along the requested axis.

        Args:
            x: Tensor shaped [B, C, D, H, W].
            dim: Axis selector: "w", "h", or "d".

        Returns:
            The padded tensor, ready for one 3D convolution pass.
        """
        k = self.window_size
        left = (k - 1) // 2
        right = k - 1 - left
        if dim == "w":
            pad = (left, right, 0, 0, 0, 0)
        elif dim == "h":
            pad = (0, 0, left, right, 0, 0)
        else:
            pad = (0, 0, 0, 0, left, right)
        return F.pad(x, pad, mode="replicate")

    def _blur3d(self, x: torch.Tensor) -> torch.Tensor:
        """Blur a 5D tensor with a separable 3D Gaussian window.

        Purpose:
            Compute local neighborhood statistics for SSIM without materializing
            a dense 3D kernel, which would be more memory intensive.

        Method:
            Expands the stored 1D Gaussian kernel into three depthwise conv3d
            kernels and applies them sequentially along width, height, and
            depth. This is equivalent to a separable 3D Gaussian blur.

        Args:
            x: Tensor shaped [B, C, D, H, W].

        Returns:
            Blurred tensor with the same shape as the input.
        """
        c = x.shape[1]
        k = self.kernel_1d.to(device=x.device, dtype=x.dtype)

        kw = k.view(1, 1, 1, 1, -1).repeat(c, 1, 1, 1, 1)
        kh = k.view(1, 1, 1, -1, 1).repeat(c, 1, 1, 1, 1)
        kd = k.view(1, 1, -1, 1, 1).repeat(c, 1, 1, 1, 1)

        y = F.conv3d(self._pad_same(x, "w"), kw, groups=c)
        y = F.conv3d(self._pad_same(y, "h"), kh, groups=c)
        y = F.conv3d(self._pad_same(y, "d"), kd, groups=c)
        return y

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute mixed 3D SSIM-MSE loss for seismic reconstruction.

        Purpose:
            Measure reconstruction quality using both voxelwise error and local
            structural similarity, while keeping the final loss in a familiar
            "lower is better" form.

        Method:
            1. Cast prediction and target to float32 for stable SSIM math.
            2. Compute masked MSE over valid voxels only.
            3. Estimate zero-mean local second-order moments using separable
               3D Gaussian smoothing of x^2, y^2, and x*y.
            4. Compute the contrast/structure-style SSIM term with a seismic-
               specific zero-mean assumption.
            5. Restrict SSIM aggregation to neighborhoods with sufficient valid
               mask support.
            6. Convert SSIM score to a non-negative loss component and combine
               it with MSE.

        Args:
            pred: Predicted tensor shaped [B, C, D, H, W].
            target: Ground-truth tensor with the same shape as ``pred``.
            valid_mask: Optional tensor with the same shape, where non-zero
                values indicate voxels that should contribute to the loss.

        Returns:
            A scalar tensor containing the mixed SSIM-MSE loss.

        Raises:
            ValueError: If tensor shapes do not match or tensors are not 5D.
        """
        if pred.shape != target.shape:
            raise ValueError("pred and target must have identical shape")
        if pred.ndim != 5:
            raise ValueError("expected tensors shaped [B, C, D, H, W]")

        # Keep SSIM math in float32 regardless of autocast precision.
        pred32 = pred.float()
        target32 = target.float()

        # SSIM assumes bounded dynamic range. Rescale seismic amplitudes from
        # approximately [-data_range/2, data_range/2] to [0, 1].
        pred32 = _rescale_zero_centered_to_unit(pred32, self.data_range)
        target32 = _rescale_zero_centered_to_unit(target32, self.data_range)

        if valid_mask is None:
            mask = torch.ones_like(pred32, dtype=torch.float32)
        else:
            if valid_mask.shape != pred.shape:
                raise ValueError("valid_mask must match pred shape")
            mask = valid_mask.to(dtype=torch.float32)

        # MSE over valid voxels only.
        denom = mask.sum(dtype=torch.float32).clamp_min(1.0)
        mse = (((pred32 - target32) ** 2) * mask).sum(dtype=torch.float32) / denom
        if self.alpha <= 0.0:
            return mse

        # Local normalized second-order moments (zero-mean assumption).
        norm = self._blur3d(mask).clamp_min(self.eps)
        ex2 = self._blur3d(mask * pred32 * pred32) / norm
        ey2 = self._blur3d(mask * target32 * target32) / norm
        exy = self._blur3d(mask * pred32 * target32) / norm

        # Effective SSIM range is 1.0 after rescaling to [0, 1].
        c2 = (self.k2 * 1.0) ** 2
        cs = (2.0 * exy + c2) / (ex2 + ey2 + c2 + self.eps)

        # Restrict aggregation to neighborhoods with enough valid support.
        support = (norm >= self.min_valid_ratio).to(dtype=torch.float32)
        support_sum = support.sum(dtype=torch.float32)
        if support_sum <= 0:
            ssim_score = torch.clamp(cs.mean(dtype=torch.float32), min=-1.0, max=1.0)
        else:
            ssim_score = torch.clamp(
                (cs * support).sum(dtype=torch.float32) / support_sum,
                min=-1.0,
                max=1.0,
            )

        # Map score in [-1,1] to a non-negative loss-like component.
        ssim_component = 0.5 * (1.0 - ssim_score)
        if self.alpha >= 1.0:
            return ssim_component
        return (1.0 - self.alpha) * mse + self.alpha * ssim_component


class MONAIStyleSSIMMSELoss3D(nn.Module):
    """MONAI-style 3D SSIM + MSE mixture.

    This implementation follows MONAI's standard SSIM formulation with local
    means and covariance terms (non-zero-mean aware), then blends it with MSE:

            total = (1 - alpha) * mse + alpha * dssim

        where dssim = 0.5 * (1 - ssim), so perfect structural similarity maps
        to 0 and larger values are worse.

    Defaults track MONAI SSIM defaults for stability constants and window size.
    """

    def __init__(
        self,
        data_range: float = 30.0,
        win_size: int = 7,
        k1: float = 0.01,
        k2: float = 0.03,
        alpha: float = 1.0 / 6.0,
        min_valid_ratio: float = 0.5,
        eps: float = 1e-8,
    ):
        super().__init__()
        if win_size < 3:
            raise ValueError("win_size must be >= 3")
        if data_range <= 0:
            raise ValueError("data_range must be > 0")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        if not (0.0 <= min_valid_ratio <= 1.0):
            raise ValueError("min_valid_ratio must be in [0, 1]")

        self.data_range = float(data_range)
        self.win_size = int(win_size)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.alpha = float(alpha)
        self.min_valid_ratio = float(min_valid_ratio)
        self.eps = float(eps)

        kernel = torch.ones((1, 1, self.win_size, self.win_size, self.win_size), dtype=torch.float32)
        kernel = kernel / float(self.win_size ** 3)
        self.register_buffer("w", kernel, persistent=False)
        self.cov_norm = float(self.win_size ** 2) / float(self.win_size ** 2 - 1)

    def _ssim_component(
        self,
        pred32: torch.Tensor,
        target32: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if pred32.shape != target32.shape:
            raise ValueError("pred and target must have identical shape")
        if pred32.ndim != 5:
            raise ValueError("expected tensors shaped [B, C, D, H, W]")

        if valid_mask.shape != pred32.shape:
            raise ValueError("valid_mask must match pred shape")

        c = pred32.shape[1]
        w = self.w.to(device=pred32.device, dtype=pred32.dtype).repeat(c, 1, 1, 1, 1)
        conv = F.conv3d

        x = pred32 * valid_mask
        y = target32 * valid_mask

        ux = conv(x, w, groups=c)
        uy = conv(y, w, groups=c)
        uxx = conv(x * x, w, groups=c)
        uyy = conv(y * y, w, groups=c)
        uxy = conv(x * y, w, groups=c)

        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)

        # Effective SSIM range is 1.0 after rescaling to [0, 1].
        c1 = (self.k1 * 1.0) ** 2
        c2 = (self.k2 * 1.0) ** 2

        numerator = (2.0 * ux * uy + c1) * (2.0 * vxy + c2)
        denom = (ux * ux + uy * uy + c1) * (vx + vy + c2) + self.eps
        ssim_map = numerator / denom

        support = conv(valid_mask, w, groups=c)
        support_mask = (support >= self.min_valid_ratio).to(dtype=ssim_map.dtype)
        support_sum = support_mask.sum(dtype=torch.float32)

        if support_sum <= 0:
            ssim_score = torch.clamp(ssim_map.mean(dtype=torch.float32), min=-1.0, max=1.0)
        else:
            ssim_score = torch.clamp(
                (ssim_map * support_mask).sum(dtype=torch.float32) / support_sum,
                min=-1.0,
                max=1.0,
            )

        return 0.5 * (1.0 - ssim_score)

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

        # Match custom SSIM path: rescale zero-centered seismic amplitudes to
        # [0, 1] before both MSE and SSIM computation.
        pred32 = _rescale_zero_centered_to_unit(pred32, self.data_range)
        target32 = _rescale_zero_centered_to_unit(target32, self.data_range)

        if valid_mask is None:
            mask = torch.ones_like(pred32, dtype=torch.float32)
        else:
            if valid_mask.shape != pred.shape:
                raise ValueError("valid_mask must match pred shape")
            mask = valid_mask.to(dtype=torch.float32)

        denom = mask.sum(dtype=torch.float32).clamp_min(1.0)
        mse = (((pred32 - target32) ** 2) * mask).sum(dtype=torch.float32) / denom
        if self.alpha <= 0.0:
            return mse

        ssim_component = self._ssim_component(pred32, target32, mask)
        if self.alpha >= 1.0:
            return ssim_component
        return (1.0 - self.alpha) * mse + self.alpha * ssim_component


class SlidingWindowStatsLoss3D(nn.Module):
    """3D local-statistics loss using sliding-window mean and std terms.

    The loss is additive:

        total = (
            mean_weight * L_mean
            + std_weight * L_std
            + min_weight * L_minima
            + max_weight * L_maxima
            + mae_weight * L_mae
            + mse_weight * L_mse
        )

    where:
    - L_mean is MAE between local sliding-window means of pred and target.
    - L_std measures how far local std ratio (target/pred) deviates from 1.
    - L_minima is MAE between local sliding-window minima of pred and target.
    - L_maxima is MAE between local sliding-window maxima of pred and target.
    - L_mae is voxelwise MAE between pred and target.
    - L_mse is voxelwise MSE between pred and target.

    Local statistics are computed with stride-1 avg pooling over a kernel
    (default 9x9x9). When a valid mask is provided, local moments are computed
    as masked moments (weighted by valid voxels) unless ``apply_to_all_voxels``
    is enabled.
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
        apply_to_all_voxels: bool = False,
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

        self.window_size = tuple(int(k) for k in window_size)
        self.mean_weight = float(mean_weight)
        self.std_weight = float(std_weight)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.mae_weight = float(mae_weight)
        self.mse_weight = float(mse_weight)
        self.eps = float(eps)
        self.std_ratio_clip = float(std_ratio_clip)
        self.apply_to_all_voxels = bool(apply_to_all_voxels)

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
        # Build finite sentinels from batch range so masked-out voxels never
        # dominate local extrema while staying numerically stable on MPS/CUDA.
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

        if valid_mask is None or self.apply_to_all_voxels:
            mask = torch.ones_like(pred32, dtype=torch.float32)
        else:
            if valid_mask.shape != pred.shape:
                raise ValueError("valid_mask must match pred shape")
            mask = valid_mask.to(dtype=torch.float32)

        mean_pred, std_pred = self._masked_local_moments(pred32, mask)
        mean_tgt, std_tgt = self._masked_local_moments(target32, mask)
        min_pred, max_pred, extrema_support = self._masked_local_extrema(pred32, mask)
        min_tgt, max_tgt, _ = self._masked_local_extrema(target32, mask)

        # Sliding-window averaged MAE term.
        mean_mae = torch.abs(mean_pred - mean_tgt)

        # Local std-ratio term, stabilized and clipped to avoid blowups.
        std_ratio = (std_tgt + self.eps) / (std_pred + self.eps)
        std_ratio = torch.clamp(std_ratio, 1.0 / self.std_ratio_clip, self.std_ratio_clip)
        std_ratio_penalty = torch.abs(std_ratio - 1.0)

        # Local extrema alignment terms.
        min_mae = torch.abs(min_pred - min_tgt)
        max_mae = torch.abs(max_pred - max_tgt)

        # Global voxelwise reconstruction terms.
        mae_voxel = torch.abs(pred32 - target32)
        mse_voxel = (pred32 - target32) ** 2

        # Keep reduction aligned with valid voxels unless all-voxel mode is on.
        if self.apply_to_all_voxels:
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


class CompositeClusterAwareLoss(nn.Module):
    """Composite loss that upweights traces with high masking density.

    This wrapper composes an existing 3D loss (e.g. :class:`SSIMMSELoss3D`) and
    creates a two-term objective:

        loss = base_weight * L_base + cluster_weight * L_cluster

    where:
    - ``L_base`` is the loss computed over traces with LOW masking density
      (high proportion of valid voxels)
    - ``L_cluster`` is the loss computed over traces with HIGH masking density
      (high proportion of masked voxels)

    This design targets the reconstruction of heavily-masked regions while
    maintaining baseline reconstruction quality across sparsely-masked regions.

    Notes:
    - Expects model/target tensors shaped ``[B, C, D, H, W]``.
    - Per-trace masking density is computed as the median fraction of masked
      voxels across all traces in the batch. Traces are then split at this
      median for base vs. cluster weighting.
        - Weight assignment is at the VOXEL level, not trace-level.
        - Works with any masking strategy: geometric padding, sparse voxel masking, etc.
    """

    def __init__(
        self,
        base_criterion: nn.Module,
        kernel_size: int = 5,
        eps: float = 1e-6,
        base_weight: float = 1.0 / 3.0,
        cluster_weight: float = 2.0 / 3.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_criterion, nn.Module):
            raise ValueError("base_criterion must be an nn.Module")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if base_weight < 0 or cluster_weight < 0:
            raise ValueError("weights must be non-negative")
        self.base = base_criterion
        self.kernel_size = int(kernel_size)
        self.eps = float(eps)
        self.base_weight = float(base_weight)
        self.cluster_weight = float(cluster_weight)

    def _weight_masks(
        self,
        reference: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the base and cluster masks based on per-trace masking density.
        
        Instead of looking for completely masked traces, this method uses masking
        DENSITY to identify regions:
        - base_mask: Traces with LOW masking density (high proportion of valid voxels)
        - cluster_mask: Traces with HIGH masking density (high proportion of masked voxels)
        
        This approach works with sparse voxel-level masking where no traces are
        completely masked.
        
        Returns:
            base_mask: Tensor of shape [B, C, D, H, W] = 1.0 for traces with low mask density
            cluster_mask: Tensor of shape [B, C, D, H, W] = 1.0 for traces with high mask density
        """
        if reference.ndim != 5:
            raise ValueError("expected tensors shaped [B, C, D, H, W]")

        B, C, D, H, W = reference.shape
        
        if valid_mask is None:
            # No mask provided: all voxels valid, uniform base weighting
            base_mask = torch.ones((B, C, D, H, W), dtype=reference.dtype, device=reference.device)
            cluster_mask = torch.zeros((B, C, D, H, W), dtype=reference.dtype, device=reference.device)
        else:
            if valid_mask.shape != reference.shape:
                raise ValueError("valid_mask must match pred shape")
            
            # Compute per-trace masking density: fraction of MASKED (invalid) voxels
            valid_mask_float = valid_mask.float()  # 1.0 = valid, 0.0 = masked
            per_trace_valid_count = valid_mask_float.sum(dim=2)  # [B, C, H, W]: sum over D
            per_trace_valid_fraction = per_trace_valid_count / D  # [B, C, H, W]: fraction in [0, 1]
            
            # Masking density = 1 - valid_fraction
            # High density = many masked voxels, Low density = mostly valid
            per_trace_mask_density = 1.0 - per_trace_valid_fraction  # [B, C, H, W]
            
            # Use median masking density as threshold to separate base from cluster regions
            mask_density_flat = per_trace_mask_density.view(-1)
            threshold = torch.median(mask_density_flat)
            
            # base_mask: traces with low masking density (below median)
            base_mask_2d = (per_trace_mask_density[:, 0, :, :] <= threshold).float()  # [B, H, W]
            base_mask_2d = base_mask_2d.view(B, 1, 1, H, W).expand(B, C, D, H, W)
            
            # cluster_mask: traces with high masking density (above median)
            cluster_mask_2d = (per_trace_mask_density[:, 0, :, :] > threshold).float()  # [B, H, W]
            cluster_mask_2d = cluster_mask_2d.view(B, 1, 1, H, W).expand(B, C, D, H, W)
            
            base_mask = base_mask_2d.to(dtype=reference.dtype)
            cluster_mask = cluster_mask_2d.to(dtype=reference.dtype)
        
        return base_mask, cluster_mask

    def diagnostic_weight_maps(
        self,
        reference: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-voxel weight maps for visualization and diagnostics.
        
        These maps show the actual weight emphasis applied to each voxel by the
        composite loss:
        - base_weight_map: Applied to traces with LOW masking density
        - cluster_weight_map: Applied to traces with HIGH masking density
        
        The split between base and cluster is determined by the median masking
        density across all traces in the batch.
        
        Args:
            reference: Reference tensor of shape [B, C, D, H, W].
            valid_mask: Optional mask of shape [B, C, D, H, W] indicating valid voxels.
        
        Returns:
            Tuple of (base_weight_map, cluster_weight_map), both [B, C, D, H, W]
        """
        base_mask, cluster_mask = self._weight_masks(reference, valid_mask)
        base_weight_map = self.base_weight * base_mask
        cluster_weight_map = self.cluster_weight * cluster_mask
        return base_weight_map, cluster_weight_map

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the composite cluster-aware loss.

        Args:
            pred: Predicted tensor shaped [B, C, D, H, W].
            target: Ground-truth tensor with same shape as ``pred``.
            valid_mask: Optional tensor with same shape marking valid voxels
                (non-zero means valid). When omitted, all voxels are treated
                as valid for the base loss.

        Returns:
            Scalar composite loss tensor = base_weight * L_base + cluster_weight * L_cluster
        """
        if pred.shape != target.shape:
            raise ValueError("pred and target must have identical shape")
        if pred.ndim != 5:
            raise ValueError("expected tensors shaped [B, C, D, H, W]")

        base_mask, cluster_mask = self._weight_masks(pred, valid_mask)
        
        # Base loss: computed over base mask (all valid, non-padding regions)
        L_base = self.base(pred, target, valid_mask=base_mask)
        
        # Cluster loss: computed over cluster mask (completely masked traces + neighbors)
        L_cluster = self.base(pred, target, valid_mask=cluster_mask)
        
        return self.base_weight * L_base + self.cluster_weight * L_cluster
