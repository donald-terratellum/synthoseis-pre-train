"""
Geometric correctness tests for stretch_squeeze_3d and augment_pair_3d.

Root-cause context
------------------
A previous bug scrambled which scale factor was applied to which axis:
  - stretch_squeeze_3d was called with (sx, sy, sz) but expected (sz, sx, sy)
  - zoom_factors inside the function mapped the wrong scales to the wrong axes

Consequence: when the z-axis was squeezed (sz < 1), the zoom actually squeezed
the x-axis, and the edge mask checked the x-scale for z-boundaries.  This left
real squeeze artifacts at the z-boundaries unmasked, producing the horizontal
band of high-amplitude spurious voxels visible in TensorBoard at z ≈ 100-120.

These tests reproduce that failure mode and verify the fix.
"""

import numpy as np
import pytest
from scipy.ndimage import zoom as scipy_zoom

from synthoseis_pre_train.augmentation import (
    stretch_squeeze_3d,
    augment_pair_3d,
    crop_or_pad_to_shape,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def make_uniform_layers(
    shape=(128, 128, 128),
    n_layers: int = 8,
    dip_x_deg: float = 0.0,
    seed: int = 0,
):
    """
    Synthetic seismic volume: constant-amplitude layers dipping in x.

    Each layer has a scalar amplitude drawn from N(0,1).  Layer boundaries
    are equally spaced in z at x=0 and shift linearly with x according to
    ``dip_x_deg`` (degrees from horizontal).

    Returns
    -------
    vol : float32 ndarray, shape (z, x, y)
    layer_ids : int32 ndarray, shape (z, x, y)  — value = layer index 0..n-1
    amplitudes : float32 1-D array, length n_layers
    """
    rng = np.random.default_rng(seed)
    nz, nx, ny = shape
    dip = np.tan(np.radians(dip_x_deg))
    z_bounds = np.linspace(0, nz, n_layers + 1)
    amplitudes = rng.standard_normal(n_layers).astype(np.float32)

    vol = np.zeros(shape, dtype=np.float32)
    layer_ids = np.full(shape, -1, dtype=np.int32)

    for xi in range(nx):
        shift = dip * xi
        for li in range(n_layers):
            z0 = int(np.clip(z_bounds[li] + shift, 0, nz))
            z1 = int(np.clip(z_bounds[li + 1] + shift, 0, nz))
            if z0 < z1:
                vol[z0:z1, xi, :] = amplitudes[li]
                layer_ids[z0:z1, xi, :] = li

    return vol, layer_ids, amplitudes


# ---------------------------------------------------------------------------
# 1. Axis-correctness unit tests for stretch_squeeze_3d
#    These directly reproduce the bug: squeezing one axis must shrink only
#    that axis's valid-data extent, leaving the others at full size.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("squeeze_axis,scale_factors", [
    (0, (0.5, 1.0, 1.0)),   # squeeze z only  — (sz, sx, sy)
    (1, (1.0, 0.5, 1.0)),   # squeeze x only
    (2, (1.0, 1.0, 0.5)),   # squeeze y only
])
def test_squeeze_applied_to_correct_axis(squeeze_axis, scale_factors):
    """
    A squeeze factor < 1 on a single axis must reduce that axis's extent only.

    Failure mode with the old bug: squeezing z would shrink x instead, so
    this test would fail on the squeeze_axis=0 case.
    """
    vol = np.ones((128, 128, 128), dtype=np.float32)
    result, mask = stretch_squeeze_3d(vol, scale_factors=scale_factors, mask_edges=True)

    valid = result.copy()
    valid[~mask] = 0.0

    extents = [
        int((valid.any(axis=(1, 2))).sum()),   # z-extent
        int((valid.any(axis=(0, 2))).sum()),   # x-extent
        int((valid.any(axis=(0, 1))).sum()),   # y-extent
    ]

    squeeze_factor = scale_factors[squeeze_axis]
    expected_lo = int(128 * squeeze_factor * 0.65)
    expected_hi = int(128 * squeeze_factor * 1.35)

    assert expected_lo <= extents[squeeze_axis] <= expected_hi, (
        f"axis {squeeze_axis}: squeezed extent {extents[squeeze_axis]} "
        f"not in [{expected_lo}, {expected_hi}] for factor={squeeze_factor:.2f}.  "
        f"All extents: z={extents[0]}, x={extents[1]}, y={extents[2]}"
    )

    for other in range(3):
        if other == squeeze_axis:
            continue
        assert extents[other] >= 100, (
            f"Unexpected shrinkage on axis {other} (extent={extents[other]}) "
            f"when only axis {squeeze_axis} was supposed to be squeezed."
        )


# ---------------------------------------------------------------------------
# 2. Edge-mask must cover the zero-padded region after squeezing
# ---------------------------------------------------------------------------

def test_edge_mask_covers_zero_padded_region_z():
    """
    After squeezing z by 0.5, every voxel that crop_or_pad set to zero must
    be marked False in the returned edge_mask (i.e. excluded from the loss).
    """
    vol = np.ones((128, 128, 128), dtype=np.float32)
    result, mask = stretch_squeeze_3d(vol, scale_factors=(0.5, 1.0, 1.0), mask_edges=True)

    # Every zero voxel must be outside the mask
    zero_inside_mask = (result == 0.0) & mask
    assert not zero_inside_mask.any(), (
        f"{zero_inside_mask.sum()} zero voxels inside the edge mask — "
        "squeeze boundary not properly covered."
    )


# ---------------------------------------------------------------------------
# 3. No spurious extrema in y label — integration test
#    Dipping layers give predictable amplitudes; after augmentation the
#    normalised y should have no voxels with |value| >> 5 σ.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_no_spurious_extrema_in_y_label(seed):
    """
    The y label from augment_pair_3d must not contain extreme outlier values.

    With the old zoom-axis bug, unmasked z-boundary artifacts appeared as
    bright/dark voxels with |amplitude| >> anything in the rest of the volume.
    After normalisation to target_std=1, genuine seismic should stay within
    about ±5 σ; we use ±8 as a generous threshold.
    """
    # augment_pair_3d now expects a full cube in (x, y, z) order and may
    # extract a larger subvolume when scales < 1.  Use a larger synthetic cube
    # than the 128^3 target so scale-aware extraction is always feasible.
    vol_zxy, _, _ = make_uniform_layers(shape=(192, 192, 192), seed=seed, dip_x_deg=15.0)
    cube_xyz = np.transpose(vol_zxy, (1, 2, 0)).copy()

    _, y_arr, mask, _ = augment_pair_3d(
        cube_xyz,
        target_shape=(128, 128, 128),
        normalize=True,
        target_std=1.0,
    )

    valid_y = y_arr[mask]
    if valid_y.size == 0:
        pytest.skip("No valid voxels in mask — degenerate sample, skip.")

    max_abs = float(np.abs(valid_y).max())
    assert max_abs < 8.0, (
        f"Spurious extremum in y label: |max|={max_abs:.2f} after normalisation "
        f"(seed={seed}).  Likely an unmasked z-boundary artefact."
    )


# ---------------------------------------------------------------------------
# 4. Dipping-layer topology preservation — analytical ground-truth comparison
#
#    Ground truth is built by:
#      (a) Applying the same zoom (nearest-neighbour) to the integer layer-ID
#          array so layer membership is remapped without blending.
#      (b) Refilling each layer-ID with the layer's original amplitude.
#      (c) Applying the same flips / axis-swap from params.
#      (d) Normalising with the same pre-augmentation mean/std.
#
#    Nearest-neighbour vs. linear interpolation causes small boundary
#    differences, so we require Pearson r > 0.90 over valid voxels.
#    A failure (r << 1) indicates that the zoom was applied to wrong axes.
# ---------------------------------------------------------------------------

def test_dipping_layers_topology_preserved():
    """
    Augmented y must correlate strongly (r > 0.90) with an analytical
    ground truth built from the layer-ID array using nearest-neighbour zoom.
    """
    np.random.seed(17)   # makes augment_pair_3d deterministic

    vol, layer_ids, amplitudes = make_uniform_layers(
        shape=(128, 128, 128), n_layers=8, dip_x_deg=12.0, seed=17
    )

    # Pass cube in (x, y, z) order as required by augment_pair_3d.
    cube_xyz = np.transpose(vol, (1, 2, 0)).copy()

    # Keep scales >= 1 so extraction size equals the full 128^3 input; this
    # removes random subvolume offsets and preserves analytical comparability.
    _, y_arr, mask, params = augment_pair_3d(
        cube_xyz,
        target_shape=(128, 128, 128),
        normalize=True,
        target_std=1.0,
        z_stretch_range=(1.0, 1.25),
        xy_stretch_range=(1.0, 1.25),
        phase_range=(0.0, 0.0),
        time_to_depth=False,   # exclude t2d so GT stays analytically tractable
    )

    # params['stretch_factors'] = (x_scale, y_scale, z_scale) as named in augment_pair_3d
    x_scale, y_scale, z_scale = params['stretch_factors']

    # Apply zoom to layer IDs (nearest-neighbour = order=0 to preserve integer IDs)
    # Data axes are (z, x, y) → zoom factors must be (z_scale, x_scale, y_scale)
    zoomed_ids_f = scipy_zoom(
        layer_ids.astype(np.float32),
        (z_scale, x_scale, y_scale),
        order=0,
    )
    zoomed_ids = crop_or_pad_to_shape(
        zoomed_ids_f.astype(np.int32), vol.shape
    )

    # Refill amplitudes from remapped layer IDs
    gt = np.zeros(vol.shape, dtype=np.float32)
    for li, amp in enumerate(amplitudes):
        gt[zoomed_ids == li] = amp

    # Apply same spatial transforms as augment_pair_3d
    if params['flip_x']:
        gt = np.flip(gt, axis=1).copy()
    if params['flip_y']:
        gt = np.flip(gt, axis=2).copy()
    if params['swap_xy']:
        gt = np.swapaxes(gt, 1, 2).copy()

    # Normalise with pre-augmentation stats (identical to what augment_pair_3d used)
    norm_mean = params['norm_mean']
    norm_std = params['norm_std']
    if norm_std > 0:
        gt = (gt - norm_mean) / norm_std * params['target_std']

    valid_y = y_arr[mask].ravel()
    valid_gt = gt[mask].ravel()

    if valid_y.size < 200:
        pytest.skip("Too few valid voxels for a meaningful correlation check.")

    corr = float(np.corrcoef(valid_y, valid_gt)[0, 1])
    assert corr > 0.90, (
        f"Augmented y vs analytical ground truth: r={corr:.3f}.  "
        "Low correlation suggests the zoom is being applied to the wrong axes."
    )
