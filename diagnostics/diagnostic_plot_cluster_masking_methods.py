"""Create a diagnostic PNG comparing 3 cluster-center masking methods.

This script generates one figure with three square 128x128 subplots:
- Left: uniform random centers with 3x3 cluster masking
- Center: Mitchell best-candidate centers with 3x3 cluster masking
- Right: Bridson Poisson-disc centers with 3x3 cluster masking

The image is rendered with nearest-neighbor interpolation so individual trace
pixels remain crisp (no smoothing).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter

from synthoseis_pre_train.masking import (
    _estimate_center_count_for_target_mask_ratio,
    _select_cluster_centers,
    create_mask_3d,
)


def _build_panel_mask(
    method: str,
    side: int,
    target_masked_fraction: float,
    cluster_prob: float,
    cluster_shape: int,
    seed: int,
) -> tuple[np.ndarray, int, float]:
    """Build one 2D trace mask panel using create_mask_3d.

    Args:
        method: Cluster-center sampling method for create_mask_3d.
        side: Width/height of the square trace grid.
        target_masked_fraction: Target final masked fraction.
        cluster_prob: Per-trace probability inside each 3x3 cluster.
        cluster_shape: Cluster edge size (odd integer).
        seed: Random seed for deterministic output.

    Returns:
        Tuple of:
        - 2D array shaped [side, side] where 1=masked and 0=unmasked.
        - Cluster-center count used for this method.
        - Masked fraction in [0, 1].
    """
    # Use z=2 so create_mask_3d skips peak/trough stage (z > 2 check),
    # isolating the cluster-center masking behavior for this diagnostic.
    seismic_stub = np.zeros((2, side, side), dtype=np.float32)

    # Estimate center count exactly as create_mask_3d does.
    n_centers = _estimate_center_count_for_target_mask_ratio(
        width=side,
        height=side,
        trace_mask_ratio=target_masked_fraction,
        cluster_prob=cluster_prob,
        cluster_shape=cluster_shape,
    )

    # Rebuild center list for reporting with deterministic seed and method.
    centers = _select_cluster_centers(
        width=side,
        height=side,
        n_centers=n_centers,
        method=method,
        rng=np.random.default_rng(seed),
    )

    # Build full 3D bool mask where True=preserve, False=masked.
    mask3d = create_mask_3d(
        seismic_data=seismic_stub,
        trace_mask_ratio=target_masked_fraction,
        cluster_prob=cluster_prob,
        cluster_shape=cluster_shape,
        random_seed=seed,
        center_selection_method=method,
    )

    # Convert first z-slice to a display image where masked traces are 1.
    masked = (~mask3d[0]).astype(np.uint8)
    masked_fraction = float(masked.mean())
    return masked, len(centers), masked_fraction


def make_diagnostic_plot(
    output_png: Path,
    side: int = 128,
    trace_mask_ratio: float = 0.15,
    cluster_prob: float = 0.8,
    cluster_shape: int = 3,
    base_seed: int = 123,
) -> None:
    """Generate and save the 3-panel cluster masking comparison figure."""
    # Prepare three panels with deterministic, method-specific seeds.
    uniform_panel, uniform_clusters, uniform_masked = _build_panel_mask(
        method="uniform_random",
        side=side,
        target_masked_fraction=trace_mask_ratio,
        cluster_prob=cluster_prob,
        cluster_shape=cluster_shape,
        seed=base_seed + 0,
    )
    mitchell_panel, mitchell_clusters, mitchell_masked = _build_panel_mask(
        method="mitchell_best_candidate",
        side=side,
        target_masked_fraction=trace_mask_ratio,
        cluster_prob=cluster_prob,
        cluster_shape=cluster_shape,
        seed=base_seed + 1,
    )
    poisson_panel, poisson_clusters, poisson_masked = _build_panel_mask(
        method="poisson_disc",
        side=side,
        target_masked_fraction=trace_mask_ratio,
        cluster_prob=cluster_prob,
        cluster_shape=cluster_shape,
        seed=base_seed + 2,
    )

    # Build one row of three square subplots (taller to accommodate colorbars).
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15, 5),
        constrained_layout=True,
    )

    # Configure panel data/titles.
    panels = [
        (uniform_panel, "Uniform Random + 3x3", uniform_clusters, uniform_masked),
        (mitchell_panel, "Mitchell Best-Candidate + 3x3", mitchell_clusters, mitchell_masked),
        (poisson_panel, "Bridson Poisson-Disc + 3x3", poisson_clusters, poisson_masked),
    ]

    # Draw each panel: filtered density behind, binary mask on top.
    for ax, (img, title, n_clusters, masked_fraction) in zip(axes, panels):
        # Smoothed density layer (background) — uniform box filter over the binary mask.
        filtered = uniform_filter(img.astype(float), size=(5, 5))
        im_filtered = ax.imshow(
            filtered,
            cmap="YlOrRd",
            interpolation="nearest",
            origin="upper",
            vmin=0.0,
            vmax=1.0,
        )
        fig.colorbar(
            im_filtered,
            ax=ax,
            fraction=0.046,
            pad=0.02,
            label="uniform_filter(3×3) density",
        )
        # Binary mask overlay (semi-transparent so density shows through).
        ax.imshow(
            img,
            cmap="gray_r",
            interpolation="nearest",
            origin="upper",
            vmin=0,
            vmax=1,
            alpha=0.45,
        )
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.02,
            0.98,
            f"clusters={n_clusters} (k={cluster_shape})\\nmasked={masked_fraction*100.0:.1f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.85, "pad": 2},
        )

    # Add a compact figure title and write PNG output.
    fig.suptitle(
        f"Cluster Masking Diagnostics (side={side}, trace_mask_ratio={trace_mask_ratio}, cluster_prob={cluster_prob})"
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the diagnostic plot script."""
    parser = argparse.ArgumentParser(
        description="Create a 3-panel PNG illustrating cluster masking center methods."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("masking_cluster_methods_128.png"),
        help="Output PNG path (default: masking_cluster_methods_128.png)",
    )
    parser.add_argument(
        "--side",
        type=int,
        default=128,
        help="Square trace-grid side length per panel (default: 128)",
    )
    parser.add_argument(
        "--trace-mask-ratio",
        type=float,
        default=0.15,
        help="Target final masked trace ratio passed to create_mask_3d (default: 0.15)",
    )
    parser.add_argument(
        "--cluster-prob",
        type=float,
        default=0.8,
        help="Per-trace masking probability inside each cluster neighborhood (default: 0.8)",
    )
    parser.add_argument(
        "--cluster-shape",
        type=int,
        default=3,
        help="Odd cluster edge size (e.g. 3, 5, 7) (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base random seed (default: 123)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for generating the masking diagnostic PNG."""
    args = _parse_args()

    # Validate user inputs before building the figure.
    if args.side <= 0:
        raise ValueError("--side must be > 0")
    if not (0.0 <= args.trace_mask_ratio <= 1.0):
        raise ValueError("--trace-mask-ratio must be in [0, 1]")
    if not (0.0 <= args.cluster_prob <= 1.0):
        raise ValueError("--cluster-prob must be in [0, 1]")
    if args.cluster_shape <= 0 or args.cluster_shape % 2 == 0:
        raise ValueError("--cluster-shape must be a positive odd integer")

    # Generate and save the requested diagnostic image.
    make_diagnostic_plot(
        output_png=args.output,
        side=args.side,
        trace_mask_ratio=args.trace_mask_ratio,
        cluster_prob=args.cluster_prob,
        cluster_shape=args.cluster_shape,
        base_seed=args.seed,
    )
    print(f"Wrote diagnostic plot: {args.output}")


if __name__ == "__main__":
    main()
