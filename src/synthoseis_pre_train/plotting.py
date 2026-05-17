"""
Diagnostic plots for seismic pre-training.

Produces cross-section figures for TensorBoard logging.
Tensor convention: model input/output is (B, 1, Z, X, Y).

Cross-section definitions:
  center-X slice → fix X=center → (Z, Y) plane  (vertical section along Y)
  center-Y slice → fix Y=center → (Z, X) plane  (vertical section along X)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive; safe in background training loops
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def _to_numpy(vol) -> np.ndarray:
    """Convert a tensor or ndarray to shape (Z, X, Y) float32."""
    if hasattr(vol, "detach"):
        vol = vol.detach().cpu().float().numpy()
    vol = np.asarray(vol, dtype=np.float32)
    if vol.ndim == 4:   # (1, Z, X, Y) → (Z, X, Y)
        vol = vol[0]
    return vol


def _symrange(*arrays) -> tuple:
    """Return (-v, v) where v = max absolute value across all arrays."""
    vmax = max(float(np.abs(a).max()) for a in arrays)
    return (-vmax or -1.0, vmax or 1.0)


def _overlay_rgba(ax, base_slice, cluster_slice, overlay_vmax: float) -> None:
    """Overlay weighted loss maps on top of a grayscale seismic slice."""
    if overlay_vmax <= 0:
        return

    cluster_norm = None
    if cluster_slice is not None:
        cluster_norm = np.clip(cluster_slice / overlay_vmax, 0.0, 1.0)

    if base_slice is not None:
        # Keep blue at a single, fixed opacity and suppress it anywhere yellow
        # exists so mixed blue+yellow blending does not create misleading tones.
        alpha = np.zeros_like(base_slice, dtype=np.float32)
        base_mask = base_slice > 0
        if cluster_norm is not None:
            base_mask &= (cluster_norm <= 0.0)
        alpha[base_mask] = 0.22
        if np.any(alpha > 0):
            rgba = np.zeros(base_slice.shape + (4,), dtype=np.float32)
            rgba[..., 2] = 1.0
            rgba[..., 3] = alpha
            ax.imshow(rgba, aspect="auto", origin="upper")

    if cluster_slice is not None:
        core_mask = cluster_norm >= 0.75
        neighbor_mask = (cluster_norm > 0.0) & (~core_mask)

        # Keep neighbors faint while making core traces much less transparent.
        # Transparency relationship target: core ~= 1/3 of neighbor transparency.
        alpha = np.zeros_like(cluster_norm, dtype=np.float32)
        alpha[neighbor_mask] = 0.11
        alpha[core_mask] = 0.70

        if np.any(alpha > 0):
            rgba = np.zeros(cluster_slice.shape + (4,), dtype=np.float32)
            rgba[..., 0] = 1.0
            rgba[..., 1] = 1.0
            rgba[..., 3] = alpha
            ax.imshow(rgba, aspect="auto", origin="upper")


def make_4panel_figure(
    input_vol,
    output_vol,
    label_vol,
    suptitle: str,
    base_weight_vol=None,
    cluster_weight_vol=None,
    fixed_amplitude_range: tuple[float, float] | None = None,
) -> Figure:
    """
    6-panel training diagnostic figure (2 rows × 3 columns).

    Layout:
      [x  center-X]  [ŷ  center-X]  [y  center-X]   ← ZY plane (top row)
      [x  center-Y]  [ŷ  center-Y]  [y  center-Y]   ← ZX plane (bottom row)

    Args:
        input_vol:  (1, Z, X, Y) or (Z, X, Y) tensor/ndarray — masked model input (x)
        output_vol: (1, Z, X, Y) or (Z, X, Y) tensor/ndarray — model reconstruction (ŷ)
        label_vol:  (1, Z, X, Y) or (Z, X, Y) tensor/ndarray — ground-truth target (y)
        suptitle:   Figure title string (dataset name, epoch, loss)
        base_weight_vol: Optional weighted base-loss mask with same volume shape.
        cluster_weight_vol: Optional weighted cluster-loss mask with same volume shape.
        fixed_amplitude_range: Optional fixed (vmin, vmax) for grayscale amplitudes.
    """
    inp = _to_numpy(input_vol)
    out = _to_numpy(output_vol)
    lbl = _to_numpy(label_vol)
    base = None if base_weight_vol is None else _to_numpy(base_weight_vol)
    cluster = None if cluster_weight_vol is None else _to_numpy(cluster_weight_vol)

    cx = inp.shape[1] // 2  # center X index
    cy = inp.shape[2] // 2  # center Y index

    # ZY plane (fix X at center)
    inp_cx = inp[:, cx, :]
    out_cx = out[:, cx, :]
    lbl_cx = lbl[:, cx, :]
    base_cx = None if base is None else base[:, cx, :]
    cluster_cx = None if cluster is None else cluster[:, cx, :]
    # ZX plane (fix Y at center)
    inp_cy = inp[:, :, cy]
    out_cy = out[:, :, cy]
    lbl_cy = lbl[:, :, cy]
    base_cy = None if base is None else base[:, :, cy]
    cluster_cy = None if cluster is None else cluster[:, :, cy]

    if fixed_amplitude_range is not None:
        vmin, vmax = fixed_amplitude_range
    else:
        vmin, vmax = _symrange(inp_cx, out_cx, lbl_cx, inp_cy, out_cy, lbl_cy)
    overlay_vmax = 0.0
    if base is not None:
        overlay_vmax = max(overlay_vmax, float(np.max(base)))
    if cluster is not None:
        overlay_vmax = max(overlay_vmax, float(np.max(cluster)))

    imkw = dict(aspect="auto", cmap="gray", vmin=vmin, vmax=vmax, origin="upper")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    axes[0, 0].imshow(inp_cx, **imkw)
    _overlay_rgba(axes[0, 0], base_cx, cluster_cx, overlay_vmax)
    axes[0, 0].set_title("x (input) — center-X  (ZY)")
    axes[0, 0].set_xlabel("Y")
    axes[0, 0].set_ylabel("Z (time/depth)")

    axes[0, 1].imshow(out_cx, **imkw)
    axes[0, 1].set_title("ŷ (output) — center-X  (ZY)")
    axes[0, 1].set_xlabel("Y")
    axes[0, 1].set_ylabel("Z (time/depth)")

    axes[0, 2].imshow(lbl_cx, **imkw)
    axes[0, 2].set_title("y (label) — center-X  (ZY)")
    axes[0, 2].set_xlabel("Y")
    axes[0, 2].set_ylabel("Z (time/depth)")

    axes[1, 0].imshow(inp_cy, **imkw)
    _overlay_rgba(axes[1, 0], base_cy, cluster_cy, overlay_vmax)
    axes[1, 0].set_title("x (input) — center-Y  (ZX)")
    axes[1, 0].set_xlabel("X")
    axes[1, 0].set_ylabel("Z (time/depth)")

    axes[1, 1].imshow(out_cy, **imkw)
    axes[1, 1].set_title("ŷ (output) — center-Y  (ZX)")
    axes[1, 1].set_xlabel("X")
    axes[1, 1].set_ylabel("Z (time/depth)")

    axes[1, 2].imshow(lbl_cy, **imkw)
    axes[1, 2].set_title("y (label) — center-Y  (ZX)")
    axes[1, 2].set_xlabel("X")
    axes[1, 2].set_ylabel("Z (time/depth)")

    if overlay_vmax > 0:
        for ax in (axes[0, 0], axes[1, 0]):
            ax.text(
                0.02,
                0.02,
                "blue=base, yellow=cluster",
                transform=ax.transAxes,
                fontsize=8,
                color="white",
                bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=2.0),
            )

    sm = plt.cm.ScalarMappable(cmap="gray", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.45, label="Amplitude")
    fig.suptitle(suptitle, fontsize=10, y=1.01)
    fig.tight_layout()
    return fig


def make_crosssection_figure(
    vol,
    title: str,
    axis: str = "x",
    fixed_amplitude_range: tuple[float, float] | None = None,
) -> Figure:
    """
    Single cross-section figure for TensorBoard (one panel).

    Args:
        vol:   (1, Z, X, Y) or (Z, X, Y) tensor/ndarray
        title: Figure title
        axis:  'x' → center-X slice (ZY plane)
               'y' → center-Y slice (ZX plane)
        fixed_amplitude_range: Optional fixed (vmin, vmax) for grayscale amplitudes.
    """
    v = _to_numpy(vol)

    if axis == "x":
        cx = v.shape[1] // 2
        slc = v[:, cx, :]
        xlabel, plane_label = "Y", "center-X  (ZY plane)"
    else:
        cy = v.shape[2] // 2
        slc = v[:, :, cy]
        xlabel, plane_label = "X", "center-Y  (ZX plane)"

    if fixed_amplitude_range is not None:
        vmin, vmax = fixed_amplitude_range
    else:
        vabs = float(np.abs(slc).max()) or 1.0
        vmin, vmax = -vabs, vabs

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(slc, aspect="auto", cmap="gray",
                   vmin=vmin, vmax=vmax, origin="upper")
    fig.colorbar(im, ax=ax, label="Amplitude")
    ax.set_title(plane_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Z (time/depth)")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return fig
