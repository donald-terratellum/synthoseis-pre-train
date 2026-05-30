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
from matplotlib import colors
from scipy import optimize
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
    vmax = 3.30  # TODO: remove? fix it since all arrays being plotted are supposed to conform to standard normal
    return (-vmax or -1.0, vmax or 1.0)


def _extrema_spacing_stats_z(vol: np.ndarray) -> tuple[float, float, float, float, float, float] | None:
    """Return spacing stats between consecutive peak/trough indices along Z.

    Peak/trough detection matches dataloader masking logic: a peak satisfies
    ``z_i > z_{i-1}`` and ``z_i > z_{i+1}``; a trough satisfies
    ``z_i < z_{i-1}`` and ``z_i < z_{i+1}``.
    """
    if vol.ndim != 3:
        return None
    z_dim, x_dim, y_dim = vol.shape
    if z_dim <= 2:
        return None

    is_peak = (vol[1:-1, :, :] > vol[:-2, :, :]) & (vol[1:-1, :, :] > vol[2:, :, :])
    is_trough = (vol[1:-1, :, :] < vol[:-2, :, :]) & (vol[1:-1, :, :] < vol[2:, :, :])
    peaks_troughs = is_peak | is_trough

    distances: list[np.ndarray] = []
    for xi in range(x_dim):
        for yi in range(y_dim):
            idx = np.flatnonzero(peaks_troughs[:, xi, yi]) + 1  # shift back to original z-index
            if idx.size >= 2:
                distances.append(np.diff(idx))

    if not distances:
        return None

    d = np.concatenate(distances).astype(np.float32, copy=False)
    return (
        float(d.min()),
        float(np.percentile(d, 25.0)),
        float(d.mean()),
        float(np.median(d)),
        float(np.percentile(d, 75.0)),
        float(d.max()),
    )


def make_4panel_figure(input_vol, output_vol, label_vol, suptitle: str) -> Figure:
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
    """
    inp = _to_numpy(input_vol)
    out = _to_numpy(output_vol)
    lbl = _to_numpy(label_vol)

    # Least-squares linear fit: lbl ≈ ls_scale * out + ls_offset  (on full volume)
    out_flat = out.ravel()
    lbl_flat = lbl.ravel()
    A = np.stack([out_flat, np.ones_like(out_flat)], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(A, lbl_flat, rcond=None)
    ls_scale, ls_offset = float(coeffs[0]), float(coeffs[1])

    # least-squares linear fit using curve_fit
    def func(x, a, b):
        y = a*x + b
        return y
    alpha = optimize.curve_fit(func, xdata=out_flat, ydata=lbl_flat)[0]
    print(f"         . linear lsq from curve_fit: {alpha}")

    # Compute and print stats about peak/trough separation distances in Z index.
    # Uses the same local-extrema definition as the dataloader masking path.
    spacing_stats = _extrema_spacing_stats_z(lbl)
    if spacing_stats is None:
        print("         . peak/trough Z-spacing (idx): unavailable (insufficient extrema)")
    else:
        dmin, dp25, dmean, dmedian, dp75, dmax = spacing_stats
        print(
            "         . peak/trough Z-spacing (idx): "
            f"min={dmin:.1f}, P25={dp25:.1f}, mean={dmean:.2f}, median={dmedian:.1f}, "
            f"P75={dp75:.1f}, max={dmax:.1f}"
        )

    cx = inp.shape[1] // 2  # center X index
    cy = inp.shape[2] // 2  # center Y index

    # ZY plane (fix X at center)
    inp_cx = inp[:, cx, :]
    out_cx = out[:, cx, :]
    lbl_cx = lbl[:, cx, :]
    # ZX plane (fix Y at center)
    inp_cy = inp[:, :, cy]
    out_cy = out[:, :, cy]
    lbl_cy = lbl[:, :, cy]

    # Apply fit to output cross-sections for middle-column display
    out_cx_fit = ls_scale * out_cx + ls_offset
    out_cy_fit = ls_scale * out_cy + ls_offset
    print(f"         . ls_scale, ls_offset={ls_scale:.4f}, {ls_offset:.4f}  ")
    print(
        f"         . out_cx std={out_cx.std():.4f}  "
        f"out_cx_fit std={out_cx_fit.std():.4f}  "
        f"lbl_cx std={lbl_cx.std():.4f}"
    )
    print(
        f"         . out_cy std={out_cy.std():.4f}  "
        f"out_cy_fit std={out_cy_fit.std():.4f}  "
        f"lbl_cy std={lbl_cy.std():.4f}"
    )
    out_cx_fit = out_cx * lbl_cx.std() / out_cx.std()
    out_cy_fit = out_cy * lbl_cy.std() / out_cy.std()
    out_cx_fit = out_cx.copy()
    out_cy_fit = out_cy.copy()
    # vmin, vmax = _symrange(inp_cx, out_cx_fit, lbl_cx, inp_cy, out_cy_fit, lbl_cy)
    vmin, vmax = -3.3, 3.3  # TODO: remove? fix it since all arrays being plotted are supposed to conform to standard normal 
    imkw = dict(aspect="auto", cmap="gray", vmin=vmin, vmax=vmax, origin="upper")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    axes[0, 0].imshow(inp_cx, **imkw)
    axes[0, 0].set_title("x (input) — center-X  (ZY)")
    axes[0, 0].set_xlabel("Y")
    axes[0, 0].set_ylabel("Z (time/depth)")

    axes[0, 1].imshow(out_cx_fit, **imkw)
    axes[0, 1].set_title("ŷ (output, LS-scaled) — center-X  (ZY)")
    axes[0, 1].set_xlabel("Y")
    axes[0, 1].set_ylabel("Z (time/depth)")
    axes[0, 1].text(
        0.03, 0.04,
        f"scale={ls_scale:5.2f}  offset={ls_offset:5.2f}",
        transform=axes[0, 1].transAxes,
        fontsize=8, color="white",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
    )

    axes[0, 2].imshow(lbl_cx, **imkw)
    axes[0, 2].set_title("y (label) — center-X  (ZY)")
    axes[0, 2].set_xlabel("Y")
    axes[0, 2].set_ylabel("Z (time/depth)")

    axes[1, 0].imshow(inp_cy, **imkw)
    axes[1, 0].set_title("x (input) — center-Y  (ZX)")
    axes[1, 0].set_xlabel("X")
    axes[1, 0].set_ylabel("Z (time/depth)")

    axes[1, 1].imshow(out_cy_fit, **imkw)
    axes[1, 1].set_title("ŷ (output, LS-scaled) — center-Y  (ZX)")
    axes[1, 1].set_xlabel("X")
    axes[1, 1].set_ylabel("Z (time/depth)")
    axes[1, 1].text(
        0.03, 0.04,
        f"scale={ls_scale:5.2f}  offset={ls_offset:5.2f}",
        transform=axes[1, 1].transAxes,
        fontsize=8, color="white",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
    )

    axes[1, 2].imshow(lbl_cy, **imkw)
    axes[1, 2].set_title("y (label) — center-Y  (ZX)")
    axes[1, 2].set_xlabel("X")
    axes[1, 2].set_ylabel("Z (time/depth)")

    sm = plt.cm.ScalarMappable(cmap="gray", norm=colors.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.45, label="Amplitude")
    fig.suptitle(suptitle, fontsize=10, y=1.01)
    fig.tight_layout()
    return fig


def make_crosssection_figure(vol, title: str, axis: str = "x") -> Figure:
    """
    Single cross-section figure for TensorBoard (one panel).

    Args:
        vol:   (1, Z, X, Y) or (Z, X, Y) tensor/ndarray
        title: Figure title
        axis:  'x' → center-X slice (ZY plane)
               'y' → center-Y slice (ZX plane)
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

    vabs = float(np.abs(slc).max()) or 1.0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(slc, aspect="auto", cmap="gray",
                   vmin=-vabs, vmax=vabs, origin="upper")
    fig.colorbar(im, ax=ax, label="Amplitude")
    ax.set_title(plane_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Z (time/depth)")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return fig
