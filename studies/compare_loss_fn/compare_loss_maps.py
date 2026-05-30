
from __future__ import annotations

def _adjust_times_1(label: Tensor, rng: torch.Generator) -> Tuple[Tensor, str]:
    """Scale label by 1.0 (identity)."""
    _ = rng
    pred = label * 1.0
    return pred, "times_1 (label * 1.0)"

"""Compare voxelwise loss maps under controlled pseudo-prediction perturbations.

This study script:
1. Loads a single 128^3 sample via the project dataloader.
2. Creates pseudo-predictions by perturbing the label volume.
3. Computes non-reduced 3D loss maps for several loss methods.
4. Plots center cross-sections side-by-side:
    - left: label
    - center: pseudo-prediction
    - right: voxelwise loss map

The center slice used for plotting is ``[0, 0, :, 63, :]`` for tensors shaped
``(B, C, D, H, W)``.
"""

import argparse
import os
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


from synthoseis_pre_train.dataloader import create_dataloader
from synthoseis_pre_train.losses import SMAELoss
def loss_map_smae(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Voxelwise SMAE map: e * tanh(e / 2) * mask."""
    e = pred - target
    loss = 2.5 * e * torch.tanh(e / 2)
    return loss * mask

try:
    from .losses_ssim3d import gaussian_kernel_3d, masked_local_stats
    from .train_loop import select_study_device
except ImportError:  # pragma: no cover - supports direct script execution.
    from losses_ssim3d import gaussian_kernel_3d, masked_local_stats
    from train_loop import select_study_device


AdjustmentFn = Callable[[Tensor, torch.Generator], Tuple[Tensor, str]]


def cubic_smooth3d(x: Tensor) -> Tensor:
    """Apply a separable cubic-style smoothing filter in 3D.

    Uses the 1D kernel ``[1, 4, 1] / 6`` on each axis and applies the 3D
    kernel depthwise per channel.
    """
    channels = int(x.shape[1])
    k = torch.tensor([1.0, 4.0, 1.0], device=x.device, dtype=x.dtype) / 6.0
    kernel = (k[:, None, None] * k[None, :, None] * k[None, None, :]).view(1, 1, 3, 3, 3)
    kernel = kernel.expand(channels, 1, 3, 3, 3)
    return F.conv3d(x, kernel, padding=1, groups=channels)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the study runner."""
    parser = argparse.ArgumentParser(description="Compare voxelwise loss maps for pseudo-predictions")
    parser.add_argument("--data-path", required=True, help="Path to a zarr dataset")
    parser.add_argument(
        "--array-keys",
        nargs="+",
        default=None,
        help="Optional list of zarr array keys to sample from",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Dataloader batch size")
    parser.add_argument(
        "--sample-shape",
        type=int,
        nargs=3,
        default=(128, 128, 128),
        metavar=("D", "H", "W"),
        help="Sample shape for dataloader extraction",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--noise-low", type=float, default=0.04, help="Low radial band fraction for noise")
    parser.add_argument("--noise-high", type=float, default=0.22, help="High radial band fraction for noise")
    parser.add_argument("--noise-std", type=float, default=0.25, help="Stddev of generated bandlimited noise")
    parser.add_argument("--output-dir", type=str, default="diagnostics/loss_map_compare", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU instead of MPS")
    parser.add_argument(
        "--ignore-masking",
        action="store_true",
        help="Ignore dataloader masking by using full target and all-ones mask",
    )
    parser.add_argument(
        "--geologic-score-min",
        type=float,
        default=0.5,
        help="Minimum geologic score used when ranking candidate crop centers",
    )
    return parser.parse_args()


def _ensure_5d(batch_tensor: Tensor) -> Tensor:
    """Ensure dataloader sample has shape ``(B, C, D, H, W)``."""
    if batch_tensor.ndim == 5:
        return batch_tensor
    if batch_tensor.ndim == 4:
        return batch_tensor.unsqueeze(1)
    raise ValueError(f"Expected 4D or 5D tensor from dataloader, got shape {tuple(batch_tensor.shape)}")


def _get_single_sample(args: argparse.Namespace, device: torch.device) -> Tuple[Tensor, Tensor, Tensor]:
    """Read one batch from dataloader and return ``(input, target, mask)`` on device."""
    # Use a run-unique val-center JSON so augment=False still draws a fresh
    # geologic-score center instead of reusing a previous fixed center.
    center_json_name = f"geologic_score_val_center.compare_loss_maps.{args.seed}.{os.getpid()}.json"
    loader = create_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        sample_shape=tuple(args.sample_shape),
        num_workers=0,
        pin_memory=False,
        array_keys=args.array_keys,
        augment=False,
        normalize=True,
        trace_mask_ratio=0.0 if args.ignore_masking else 0.07,
        geologic_score_sampling=True,
        geologic_score_min=args.geologic_score_min,
        geologic_val_center_json_name=center_json_name,
    )

    dataset = getattr(loader, "dataset", None)
    center_xyz = getattr(dataset, "_fixed_val_center_xyz", None)
    if center_xyz is not None:
        print(f"Selected geologic-score center_xyz: {center_xyz}")

    batch = next(iter(loader))
    input_data, target, mask = batch

    x = _ensure_5d(input_data.float().to(device))
    y = _ensure_5d(target.float().to(device))
    m = _ensure_5d(mask.to(device)).float()

    if args.ignore_masking:
        # Bypass both clustered trace masking and extrema-derived masking effects
        # for this study by using the full target volume and an all-ones mask.
        x = y.clone()
        m = torch.ones_like(y, dtype=y.dtype, device=y.device)

    if tuple(y.shape[-3:]) != (128, 128, 128):
        raise ValueError(f"Expected sample spatial shape (128,128,128), got {tuple(y.shape[-3:])}")

    return x[:1], y[:1], m[:1]


def _radial_band_mask(shape: Tuple[int, int, int], low: float, high: float, device: torch.device) -> Tensor:
    """Build a 3D radial frequency band mask with normalized radius in [0, 1]."""
    d, h, w = shape
    fz = torch.fft.fftfreq(d, device=device).view(d, 1, 1)
    fy = torch.fft.fftfreq(h, device=device).view(1, h, 1)
    fx = torch.fft.fftfreq(w, device=device).view(1, 1, w)
    radius = torch.sqrt(fz * fz + fy * fy + fx * fx)
    r_max = float(radius.max().item())
    radius_n = radius / max(r_max, 1e-8)
    return ((radius_n >= low) & (radius_n <= high)).float()


def _adjust_bandlimited_noise(
    label: Tensor,
    rng: torch.Generator,
    noise_low: float,
    noise_high: float,
    noise_std: float,
) -> Tuple[Tensor, str]:
    """Add random bandlimited noise in Fourier domain to the label volume."""
    noise = torch.randn_like(label, generator=rng) * noise_std
    n_fft = torch.fft.fftn(noise, dim=(-3, -2, -1))
    spatial_shape = (
        int(label.shape[-3]),
        int(label.shape[-2]),
        int(label.shape[-1]),
    )
    band = _radial_band_mask(spatial_shape, noise_low, noise_high, label.device)
    band = band.view(1, 1, *band.shape)
    n_band = torch.fft.ifftn(n_fft * band, dim=(-3, -2, -1)).real
    pred = label + n_band
    desc = f"bandlimited_noise(low={noise_low:.2f}, high={noise_high:.2f}, std={noise_std:.2f})"
    return pred, desc

def _adjust_times_0(label: Tensor, rng: torch.Generator) -> Tuple[Tensor, str]:
    """Scale label by 0."""
    _ = rng
    pred = label * 0.0
    return pred, "times_0 (label * 0)"

def _adjust_times_m1(label: Tensor, rng: torch.Generator) -> Tuple[Tensor, str]:
    """Scale label by -1."""
    _ = rng
    pred = label * -1.0
    return pred, "times_m1 (label * -1)"

def _adjust_divide_2(label: Tensor, rng: torch.Generator) -> Tuple[Tensor, str]:
    """Divide label by 2."""
    _ = rng
    pred = label / 2.0
    return pred, "divide_2 (label / 2)"

def _adjust_divide_m2(label: Tensor, rng: torch.Generator) -> Tuple[Tensor, str]:
    """Divide label by -2."""
    _ = rng
    pred = label / -2.0
    return pred, "divide_m2 (label / -2)"


def _adjust_smoothing(label: Tensor, rng: torch.Generator) -> Tuple[Tensor, str]:
    """Apply 3x3x3 smoothing kernel with center=1.0 and elsewhere=0.5, normalized."""
    _ = rng
    kernel = torch.full((1, 1, 3, 3, 3), 0.5, device=label.device, dtype=label.dtype)
    kernel[0, 0, 1, 1, 1] = 1.0
    kernel = kernel / kernel.sum()
    pred = F.conv3d(label, kernel, padding=1)
    return pred, "smooth_3x3x3(center=1.0, other=0.5, normalized=True)"


def _adjust_random_shift(label: Tensor, rng: torch.Generator) -> Tuple[Tensor, str]:
    """Randomly shift label by integer voxels in each axis in [-1, 1]."""
    shifts = [int(torch.randint(-1, 2, (1,), generator=rng, device=label.device).item()) for _ in range(3)]
    pred = torch.roll(label, shifts=tuple(shifts), dims=(-3, -2, -1))
    return pred, f"roll(dz={shifts[0]}, dx={shifts[1]}, dy={shifts[2]})"


def _adjust_swap_xy(label: Tensor, rng: torch.Generator) -> Tuple[Tensor, str]:
    """Swap X and Y axes (last two spatial axes)."""
    _ = rng
    pred = label.transpose(-1, -2).contiguous()
    return pred, "swap_xy(last_two_axes)"


def _adjust_random_scale(label: Tensor, rng: torch.Generator) -> Tuple[Tensor, str]:
    """Randomly scale label amplitude by a factor in [0.66, 1.5]."""
    scale = float(torch.empty(1, device=label.device).uniform_(0.66, 1.5, generator=rng).item())
    pred = label * scale
    return pred, f"scale({scale:.3f})"


def loss_map_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Voxelwise MSE map."""
    return (pred - target).pow(2) * mask


def loss_map_mae(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Voxelwise MAE map."""
    return 2.0 * (pred - target).abs() * mask


def loss_map_mae_smooth(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Voxelwise MAEsmooth map using cubic smoothing on both volumes."""
    pred_s = cubic_smooth3d(pred)
    target_s = cubic_smooth3d(target)
    return 2.0 * (pred_s - target_s).abs() * mask


def loss_map_ssim(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Voxelwise masked SSIM loss map computed as ``1 - ssim_map``."""
    kernel = gaussian_kernel_3d((11, 11, 3), (3.0, 3.0, 1.0), pred.device).to(dtype=pred.dtype)
    mu_x, mu_y, sigma_x2, sigma_y2, sigma_xy, m_norm = masked_local_stats(pred, target, mask, kernel)
    c1_ = 0.30
    c2_ = 0.90
    c1 = torch.tensor(c1_**2, dtype=pred.dtype, device=pred.device)
    c2 = torch.tensor(c2_**2, dtype=pred.dtype, device=pred.device)
    luminance = (2 * mu_x * mu_y + c1) / (mu_x.pow(2) + mu_y.pow(2) + c1)
    contrast_structure = (2 * sigma_xy + c2) / (sigma_x2 + sigma_y2 + c2)
    ssim_map = luminance * contrast_structure
    valid = (m_norm > 0).to(pred.dtype)
    return (1.0 - ssim_map) * valid


def loss_map_tv(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Voxelwise TV-like regularity map on prediction volume under mask.

    This map mirrors the spirit of ``tv_loss`` by penalizing local spatial jumps
    and assigning directional differences back to voxels.
    """
    _ = target
    tv_map = torch.zeros_like(pred)

    dz = (pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]).abs()
    dy = (pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]).abs()
    dx = (pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]).abs()

    mz = mask[:, :, 1:, :, :] * mask[:, :, :-1, :, :]
    my = mask[:, :, :, 1:, :] * mask[:, :, :, :-1, :]
    mx = mask[:, :, :, :, 1:] * mask[:, :, :, :, :-1]

    dz_m = dz * mz
    dy_m = dy * my
    dx_m = dx * mx

    tv_map[:, :, 1:, :, :] += 0.5 * dz_m
    tv_map[:, :, :-1, :, :] += 0.5 * dz_m
    tv_map[:, :, :, 1:, :] += 0.5 * dy_m
    tv_map[:, :, :, :-1, :] += 0.5 * dy_m
    tv_map[:, :, :, :, 1:] += 0.5 * dx_m
    tv_map[:, :, :, :, :-1] += 0.5 * dx_m

    return tv_map


def loss_map_fft(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Spatial proxy map for FFT magnitude mismatch.

    The spectral squared-magnitude residual is transformed back with inverse FFT,
    and absolute value is used as a non-negative spatial contribution proxy.
    """
    _ = mask
    p_fft = torch.fft.fftn(pred, dim=(-3, -2, -1))
    t_fft = torch.fft.fftn(target, dim=(-3, -2, -1))
    freq_diff2 = (torch.abs(p_fft) - torch.abs(t_fft)).pow(2)
    spatial_proxy = torch.fft.ifftn(freq_diff2, dim=(-3, -2, -1)).real.abs()
    return spatial_proxy

def loss_map_cosine_similarity(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Cosine similarity loss map (1 - cosine similarity) along Z axis (depth)."""
    # Input shape: (B, C, D, H, W)
    # Cosine similarity along D (axis=2), for each (B, C, H, W)
    # Output shape: (B, C, H, W)
    cos = torch.nn.CosineSimilarity(dim=2, eps=1e-8)
    # Flatten mask to binary for valid locations
    valid = (mask.abs().sum(dim=2) > 0)
    # Compute cosine similarity only where mask is valid
    sim = cos(pred, target)
    loss_map = (1.0 - sim) * valid.float()
    # Broadcast back to (B, C, D, H, W) for plotting: repeat along D
    loss_map = loss_map.unsqueeze(2).expand_as(pred)
    return loss_map


def center_slice(vol: Tensor) -> np.ndarray:
    """Extract center cross-section ``[0, 0, :, 63, :]`` as NumPy array."""
    return vol[0, 0, :, 63, :].detach().cpu().numpy()


def _safe_name(text: str) -> str:
    """Sanitize text for use as a filename stem."""
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("_", "-"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def plot_triplet(
    label: Tensor,
    pred: Tensor,
    loss_map: Tensor,
    method_name: str,
    adjustment_desc: str,
    output_path: Path,
    show: bool,
) -> None:
    """Plot label/pseudo-pred/difference/loss center slices in a 2x2 layout."""
    label_2d = center_slice(label)
    pred_2d = center_slice(pred)
    diff_2d = label_2d - pred_2d
    loss_2d = center_slice(loss_map)

    signal_abs_max = float(np.max(np.abs(np.stack([label_2d, pred_2d, diff_2d], axis=0))))
    signal_v = max(signal_abs_max, 1e-8)
    loss_vmax = float(np.percentile(loss_map.detach().cpu().numpy(), 98.5))
    loss_vmax = max(loss_vmax, 1e-8)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(f"{method_name} | {adjustment_desc}")
    ax_label, ax_pred = axes[0, 0], axes[0, 1]
    ax_diff, ax_loss = axes[1, 0], axes[1, 1]

    im_label = ax_label.imshow(
        label_2d,
        cmap="seismic",
        origin="lower",
        aspect="auto",
        vmin=-signal_v,
        vmax=signal_v,
    )
    ax_label.set_title("Label [0,0,:,63,:]")
    ax_label.set_xlabel("Y")
    ax_label.set_ylabel("Z")

    im_pred = ax_pred.imshow(
        pred_2d,
        cmap="seismic",
        origin="lower",
        aspect="auto",
        vmin=-signal_v,
        vmax=signal_v,
    )
    ax_pred.set_title("Pseudo-prediction [0,0,:,63,:]")
    ax_pred.set_xlabel("Y")
    ax_pred.set_ylabel("Z")

    im_diff = ax_diff.imshow(
        diff_2d,
        cmap="seismic",
        origin="lower",
        aspect="auto",
        vmin=-signal_v,
        vmax=signal_v,
    )
    ax_diff.set_title("Difference (label - pseudo) [0,0,:,63,:]")
    ax_diff.set_xlabel("Y")
    ax_diff.set_ylabel("Z")

    im_loss = ax_loss.imshow(
        loss_2d,
        cmap="gist_ncar",
        origin="lower",
        aspect="auto",
        vmin=0.0,
        vmax=loss_vmax,
    )
    ax_loss.set_title("Loss map [0,0,:,63,:]")
    ax_loss.set_xlabel("Y")
    ax_loss.set_ylabel("Z")

    cbar_signal = fig.colorbar(im_pred, ax=[ax_label, ax_pred, ax_diff], fraction=0.03, pad=0.02)
    cbar_signal.set_label("Amplitude / Difference")
    cbar_loss = fig.colorbar(im_loss, ax=ax_loss, fraction=0.046, pad=0.04)
    cbar_loss.set_label("Loss")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    """Run the compare-loss-map study for one dataloader sample."""
    args = parse_args()
    device = torch.device("cpu") if args.force_cpu else select_study_device(prefer_mps=True)
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = torch.Generator(device=device.type)
    rng.manual_seed(args.seed)

    _input_data, label, mask = _get_single_sample(args, device=device)

    adjustments: List[Tuple[str, AdjustmentFn]] = [
        (
            "bandlimited_noise",
            lambda lbl, g: _adjust_bandlimited_noise(
                lbl,
                g,
                noise_low=args.noise_low,
                noise_high=args.noise_high,
                noise_std=args.noise_std,
            ),
        ),
        ("smooth_3x3x3", _adjust_smoothing),
        ("random_shift", _adjust_random_shift),
        ("swap_xy", _adjust_swap_xy),
        ("random_scale", _adjust_random_scale),
        ("times_0", _adjust_times_0),
        ("times_1", _adjust_times_1),
        ("times_m1", _adjust_times_m1),
        ("divide_2", _adjust_divide_2),
        ("divide_m2", _adjust_divide_m2),
    ]

    loss_methods: List[Tuple[str, Callable[[Tensor, Tensor, Tensor], Tensor]]] = [
        ("MSE", loss_map_mse),
        ("MAE", loss_map_mae),
        ("MAEsmooth", loss_map_mae_smooth),
        ("SMAE", loss_map_smae),
        ("SSIM", loss_map_ssim),
        ("tv_loss", loss_map_tv),
        ("fft_lostt", loss_map_fft),
        ("CosineSimilarity", loss_map_cosine_similarity),
    ]

    out_dir = Path(args.output_dir)
    generated = 0

    for adj_name, adj_fn in adjustments:
        pred, adj_desc = adj_fn(label, rng)
        for method_name, method_fn in loss_methods:
            map_3d = method_fn(pred, label, mask)
            out_name = f"{adj_name}__{_safe_name(method_name)}.png"
            out_path = out_dir / out_name
            plot_triplet(
                label=label,
                pred=pred,
                loss_map=map_3d,
                method_name=method_name,
                adjustment_desc=adj_desc,
                output_path=out_path,
                show=args.show,
            )

            scalar = float(map_3d.mean().detach().cpu().item())
            print(f"Saved {out_path} | mean({method_name})={scalar:.6f}")
            generated += 1

    print(f"Generated {generated} plots at: {out_dir}")


if __name__ == "__main__":
    main()
