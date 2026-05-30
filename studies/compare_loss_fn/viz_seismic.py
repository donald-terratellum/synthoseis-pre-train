import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


def to_numpy(x: Tensor | np.ndarray) -> np.ndarray:
    """Convert a tensor/array to a NumPy array on CPU.

    Args:
        x: Input tensor or NumPy array.

    Returns:
        NumPy representation of ``x``.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x


def show_seismic_slices(
    volume: Tensor | np.ndarray,
    title: str,
    n_slices: int = 4,
    axis: str = "inline",
) -> None:
    """Display representative 2D slices from a 3D seismic volume.

    Args:
        volume: Tensor/array of shape ``(C, D, H, W)`` or ``(D, H, W)``.
        title: Figure title.
        n_slices: Number of slices to display.
        axis: Slice direction, one of ``inline``, ``crossline``, ``depth``.
    """
    vol = to_numpy(volume)
    if vol.ndim == 4:
        vol = vol[0]

    depth, height, width = vol.shape

    if axis == "inline":
        indices = np.linspace(0, depth - 1, n_slices, dtype=int)
        slices = [vol[i, :, :] for i in indices]
    elif axis == "crossline":
        indices = np.linspace(0, height - 1, n_slices, dtype=int)
        slices = [vol[:, i, :] for i in indices]
    else:
        indices = np.linspace(0, width - 1, n_slices, dtype=int)
        slices = [vol[:, :, i] for i in indices]

    fig, axs = plt.subplots(1, n_slices, figsize=(4 * n_slices, 4))
    fig.suptitle(title)
    for ax, slc, idx in zip(axs, slices, indices):
        _im = ax.imshow(slc.T, cmap="seismic", aspect="auto", origin="lower")
        ax.set_title(f"{axis}={idx}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def compare_reconstructions(
    input_vol: Tensor | np.ndarray,
    target_vol: Tensor | np.ndarray,
    mse_recon: Tensor | np.ndarray,
    ssim_recon: Tensor | np.ndarray,
) -> None:
    """Visual comparison of input/target and two reconstruction outputs.

    Args:
        input_vol: Masked input volume, shape ``(1, C, D, H, W)`` or ``(C, D, H, W)``.
        target_vol: Full target volume, shape ``(1, C, D, H, W)`` or ``(C, D, H, W)``.
        mse_recon: Reconstruction from MSE-trained model.
        ssim_recon: Reconstruction from MS-SSIM composite-trained model.
    """
    inp = input_vol[0] if input_vol.ndim == 5 else input_vol
    tgt = target_vol[0] if target_vol.ndim == 5 else target_vol
    mse_r = mse_recon[0] if mse_recon.ndim == 5 else mse_recon
    ssim_r = ssim_recon[0] if ssim_recon.ndim == 5 else ssim_recon

    show_seismic_slices(inp, "Input (masked)", axis="inline")
    show_seismic_slices(tgt, "Target (full)", axis="inline")
    show_seismic_slices(mse_r, "MSE-trained reconstruction", axis="inline")
    show_seismic_slices(ssim_r, "MS-SSIM composite reconstruction", axis="inline")
