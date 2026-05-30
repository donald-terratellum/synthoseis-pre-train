"""Masked 3D SSIM and MS-SSIM losses for seismic volumes.

This module implements local statistics under a binary mask and computes a
masked 3D SSIM score. A simple multi-scale variant is provided by averaging
SSIM across progressively downsampled resolutions.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def gaussian_kernel_3d(
    kernel_size: Tuple[int, int, int],
    sigma: Tuple[float, float, float],
    device: torch.device,
) -> Tensor:
    """Create a normalized separable-like 3D Gaussian kernel.

    Args:
        kernel_size: Kernel size as ``(D, H, W)``.
        sigma: Standard deviations as ``(sigma_d, sigma_h, sigma_w)``.
        device: Device where the kernel should be allocated.

    Returns:
        A tensor of shape ``(1, 1, D, H, W)`` whose values sum to 1.
    """
    kz, ky, kx = kernel_size
    sz, sy, sx = sigma

    z = torch.arange(kz, device=device) - (kz - 1) / 2
    y = torch.arange(ky, device=device) - (ky - 1) / 2
    x = torch.arange(kx, device=device) - (kx - 1) / 2

    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    g = torch.exp(-(zz**2 / (2 * sz**2) + yy**2 / (2 * sy**2) + xx**2 / (2 * sx**2)))
    g = g / g.sum()
    return g.view(1, 1, kz, ky, kx)


def conv3d_same(x: Tensor, weight: Tensor) -> Tensor:
    """Depthwise 3D convolution with same-size output.

    Args:
        x: Input tensor of shape ``(B, C, D, H, W)``.
        weight: Convolution kernel of shape ``(1, 1, kD, kH, kW)``.

    Returns:
        Tensor with the same shape as ``x``.
    """
    channels = int(x.shape[1])
    w = weight.expand(channels, 1, *weight.shape[2:])
    padding = tuple(k // 2 for k in weight.shape[2:])
    return F.conv3d(x, w, padding=padding, groups=channels)


def masked_local_stats(
    x: Tensor,
    y: Tensor,
    mask: Tensor,
    kernel: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute masked local moments used by SSIM.

    Args:
        x: Prediction tensor of shape ``(B, C, D, H, W)``.
        y: Target tensor of shape ``(B, C, D, H, W)``.
        mask: Binary or soft mask of shape ``(B, 1, D, H, W)``.
        kernel: Gaussian kernel with shape ``(1, 1, kD, kH, kW)``.

    Returns:
        Tuple containing ``(mu_x, mu_y, sigma_x2, sigma_y2, sigma_xy, m_norm)``.
    """
    eps = 1e-8
    mask = mask.clamp(0, 1)

    m_norm = conv3d_same(mask, kernel)
    m_norm = torch.clamp(m_norm, min=eps)

    x_m = x * mask
    y_m = y * mask

    mu_x = conv3d_same(x_m, kernel) / m_norm
    mu_y = conv3d_same(y_m, kernel) / m_norm

    x2_m = (x**2) * mask
    y2_m = (y**2) * mask
    xy_m = (x * y) * mask

    sigma_x2 = conv3d_same(x2_m, kernel) / m_norm - mu_x**2
    sigma_y2 = conv3d_same(y2_m, kernel) / m_norm - mu_y**2
    sigma_xy = conv3d_same(xy_m, kernel) / m_norm - mu_x * mu_y

    sigma_x2 = torch.clamp(sigma_x2, min=0.0)
    sigma_y2 = torch.clamp(sigma_y2, min=0.0)

    return mu_x, mu_y, sigma_x2, sigma_y2, sigma_xy, m_norm


def ssim_3d_masked(
    x: Tensor,
    y: Tensor,
    mask: Tensor,
    kernel_size: Tuple[int, int, int] = (11, 11, 3),
    sigma: Tuple[float, float, float] = (3.0, 3.0, 1.0),
    c1: float = 0.15**2,
    c2: float = 0.25**2,
) -> Tensor:
    """Compute masked 3D SSIM between two volumes.

    Args:
        x: Prediction tensor of shape ``(B, C, D, H, W)``.
        y: Target tensor of shape ``(B, C, D, H, W)``.
        mask: Valid-data mask of shape ``(B, 1, D, H, W)``.
        kernel_size: Gaussian window size as ``(D, H, W)``.
        sigma: Gaussian sigma values as ``(sigma_d, sigma_h, sigma_w)``.
        c1: SSIM luminance stabilization constant.
        c2: SSIM contrast/structure stabilization constant.

    Returns:
        Scalar tensor containing the mean masked SSIM value.
    """
    device = x.device
    kernel = gaussian_kernel_3d(kernel_size, sigma, device)

    mu_x, mu_y, sigma_x2, sigma_y2, sigma_xy, m_norm = masked_local_stats(x, y, mask, kernel)

    c1_t = torch.tensor(c1, device=device)
    c2_t = torch.tensor(c2, device=device)

    luminance = (2 * mu_x * mu_y + c1_t) / (mu_x**2 + mu_y**2 + c1_t)
    contrast_structure = (2 * sigma_xy + c2_t) / (sigma_x2 + sigma_y2 + c2_t)
    ssim_map = luminance * contrast_structure

    valid = m_norm > 0
    ssim_mean = (ssim_map * valid).sum() / valid.sum().clamp(min=1)
    return ssim_mean


class MSSSIM3DMasked(nn.Module):
    """Multi-scale masked SSIM module for 3D seismic reconstruction.

    The score is computed at each scale and averaged across ``levels``.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int, int] = (11, 11, 3),
        sigma: Tuple[float, float, float] = (3.0, 3.0, 1.0),
        levels: int = 3,
    ) -> None:
        """Initialize MS-SSIM settings.

        Args:
            kernel_size: Gaussian window size for each scale.
            sigma: Gaussian sigma values for each scale.
            levels: Number of scales used for pooling/aggregation.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.levels = levels

    def forward(self, x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        """Compute masked multi-scale SSIM.

        Args:
            x: Prediction tensor ``(B, C, D, H, W)``.
            y: Target tensor ``(B, C, D, H, W)``.
            mask: Mask tensor ``(B, 1, D, H, W)``.

        Returns:
            Scalar MS-SSIM score where higher is better.
        """
        ssim_vals: list[Tensor] = []
        x_l, y_l, m_l = x, y, mask
        for level in range(self.levels):
            ssim_vals.append(
                ssim_3d_masked(
                    x_l,
                    y_l,
                    m_l,
                    kernel_size=self.kernel_size,
                    sigma=self.sigma,
                )
            )
            if level < self.levels - 1:
                x_l = F.avg_pool3d(x_l, kernel_size=2, stride=2)
                y_l = F.avg_pool3d(y_l, kernel_size=2, stride=2)
                m_l = F.avg_pool3d(m_l, kernel_size=2, stride=2)
                m_l = (m_l > 0.5).float()

        ms_ssim = torch.stack(ssim_vals).mean()
        return ms_ssim
