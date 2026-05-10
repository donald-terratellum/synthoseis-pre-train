"""Loss functions for seismic pre-training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        c2 = (self.k2 * self.data_range) ** 2
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


class CompositeClusterAwareLoss(nn.Module):
    """Composite loss that upweights traces near masked clusters.

    This wrapper composes an existing 3D loss (e.g. :class:`SSIMMSELoss3D`) and
    creates a two-term objective:

        loss = base_weight * L_base + cluster_weight * L_cluster

    where ``L_base`` is the original loss computed with the provided
    ``valid_mask`` (or all voxels if none provided), and ``L_cluster`` is the
    same loss but computed only over traces whose 2D neighborhood (after a
    5x5 average filter) is > ``eps``.

    Notes:
    - Expects model/target tensors shaped ``[B, C, D, H, W]``.
    - Implements the 2D neighborhood smoothing using ``F.avg_pool2d`` so the
      computation runs on the same device (GPU) as the tensors.
    - The per-trace selection logic treats a trace as "masked" when all
      depth voxels are invalid in the supplied ``valid_mask`` (or zero if
      ``valid_mask`` is None).
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
            Scalar composite loss tensor.
        """
        # Validate shapes like the inner loss does.
        if pred.shape != target.shape:
            raise ValueError("pred and target must have identical shape")
        if pred.ndim != 5:
            raise ValueError("expected tensors shaped [B, C, D, H, W]")

        B, C, D, H, W = pred.shape

        # Base loss uses the supplied valid_mask (converted to float)
        if valid_mask is None:
            base_mask = None
        else:
            if valid_mask.shape != pred.shape:
                raise ValueError("valid_mask must match pred shape")
            base_mask = valid_mask

        L_base = self.base(pred, target, valid_mask=base_mask)

        # Derive a 2D trace-level mask where a trace is considered "masked"
        # when ALL depth voxels are invalid (zero) in the provided valid_mask.
        if valid_mask is None:
            # No masked traces when no valid_mask is provided.
            trace_all_masked = torch.zeros((B, H, W), dtype=pred.dtype, device=pred.device)
        else:
            # Collapse channel dimension first: [B, C, D, H, W] -> [B, D, H, W]
            per_depth = valid_mask.to(dtype=pred.dtype).sum(dim=1)
            # A voxel is valid if per_depth > eps; then a trace is fully masked
            # when no depth voxel is valid.
            depth_valid = per_depth > self.eps
            trace_all_masked = (~depth_valid).all(dim=1).to(dtype=pred.dtype)

        # Smooth the 2D trace map with an average-pool (kernel_size x kernel_size)
        # to obtain neighborhood density in a GPU-friendly manner.
        # Shape conv expects [B, C, H, W].
        trace_map = trace_all_masked.unsqueeze(1)  # [B,1,H,W]
        smoothed = F.avg_pool2d(
            trace_map,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            count_include_pad=False,
        )
        # smoothed: [B,1,H,W] -> drop channel
        smoothed = smoothed.squeeze(1)

        # Build cluster selection: True where smoothed density > eps
        cluster_sel = (smoothed > self.eps)

        # Construct a valid_mask selecting all voxels in selected traces.
        # cluster_sel: [B, H, W] -> expand to [B, C, D, H, W]
        cluster_mask = cluster_sel.view(B, 1, 1, H, W).expand(B, C, D, H, W).to(dtype=pred.dtype)

        L_cluster = self.base(pred, target, valid_mask=cluster_mask)

        return self.base_weight * L_base + self.cluster_weight * L_cluster
