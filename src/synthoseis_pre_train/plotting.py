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

    vmin, vmax = _symrange(inp_cx, out_cx, lbl_cx, inp_cy, out_cy, lbl_cy)
    imkw = dict(aspect="auto", cmap="gray", vmin=vmin, vmax=vmax, origin="upper")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    axes[0, 0].imshow(inp_cx, **imkw)
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

    sm = plt.cm.ScalarMappable(cmap="gray", norm=plt.Normalize(vmin=vmin, vmax=vmax))
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
