"""
Test: x/y correspondence after augmentation + masking.

Verifies that every non-zero voxel in the masked input (x) has an identical
value at the same (z, i, j) location in the label (y).  Non-zero positions in
x are exactly the unmasked voxels that the model sees — they must match y
because x is derived from y before masking, and masking only zeroes voxels
without touching the ones it preserves.

The test forces every augmentation branch to fire (fixed seed chosen so all
random decisions land True/do_*) and verifies the invariant holds.

Usage:
    uv run pytest tests/test_augmentation_pair.py -v
    # or with an explicit data path:
    DATA_PATH=/path/to/dataset.zarr uv run pytest tests/test_augmentation_pair.py -v
"""

import os
import numpy as np
import pytest
import zarr

from synthoseis_pre_train.augmentation import augment_pair_3d
from synthoseis_pre_train.masking import create_mask_3d, apply_mask_to_seismic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_FAKE_DATA_ROOT = "/Users/donaldpg/synthoseis/fake_data"


def _discover_zarr_path() -> str:
    """Return the path to the first complete model_data.zarr found under
    _FAKE_DATA_ROOT, or the legacy hard-coded path if none exists."""
    import glob
    candidates = sorted(glob.glob(f"{_FAKE_DATA_ROOT}/seismic__*/model_data.zarr"))
    return candidates[0] if candidates else (
        f"{_FAKE_DATA_ROOT}/seismic__2026.27392078__300ph7b/model_data.zarr"
    )


DEFAULT_DATA_PATH = _discover_zarr_path()
ARRAY_KEY = "seismicCubes_cumsum_fullstack"
SAMPLE_SHAPE = (128, 128, 128)  # (z, x, y) after transpose
TRACE_MASK_RATIO = 0.07


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_subvolume(zarr_path: str) -> np.ndarray:
    """Load the full zarr cube in (x, y, z) order for passing to augment_pair_3d.

    augment_pair_3d expects a full-size (x, y, z) volume and handles extraction
    internally. Pre-extracting a 128³ subvolume here would prevent the function
    from over-sampling for squeeze scales.
    """
    root = zarr.open(zarr_path, mode="r")
    assert ARRAY_KEY in root, f"Array key '{ARRAY_KEY}' not found in {zarr_path}"
    cube = root[ARRAY_KEY][:]  # shape (300, 300, 1499) on disk → (x, y, z)
    return cube.astype(np.float32)


def _force_all_augmentations(data: np.ndarray) -> tuple:
    """
    Call augment_pair_3d with a fixed seed that forces every augmentation branch
    (stretch, time-to-depth, all flips, swap_xy, noise) to be applied.

    We patch numpy's random state so the probability draws all return values
    that trigger each branch, then restore it afterwards.
    """
    # Pre-compute a seed sequence whose uniform/rand draws satisfy:
    #   sx, sy, sz  ∈ (0.8, 1.2)  — any value works
    #   rand() < 0.6 → True        (time-to-depth)
    #   velocity_grad ∈ (0.3, 0.8)
    #   rand() < 0.5 → True × 3   (flip_x, flip_y, swap_xy)
    #   rand() < 0.4 → True        (noise)
    #
    # Easiest approach: mock the draws by subclassing.

    class _ForcedRNG:
        """Returns controlled values to force all branches True."""
        def __init__(self):
            self._uniform_calls = 0
            self._rand_calls = 0

        def uniform(self, low=0.0, high=1.0, size=None):
            self._uniform_calls += 1
            mid = (low + high) / 2.0
            if size is not None:
                return np.full(size, mid, dtype=np.float64)
            return float(mid)

        def rand(self, *args):
            self._rand_calls += 1
            # Always return 0.0 → satisfies < 0.6, < 0.5, < 0.4
            if args:
                return np.zeros(args, dtype=np.float64)
            return 0.0

    rng = _ForcedRNG()

    # Temporarily replace np.random functions used inside augment_pair_3d
    import synthoseis_pre_train.augmentation as aug_mod
    orig_uniform = np.random.uniform
    orig_rand    = np.random.rand

    np.random.uniform = rng.uniform
    np.random.rand    = rng.rand

    try:
        result = augment_pair_3d(
            data,
            z_stretch_range=(0.8, 1.2),
            xy_stretch_range=(0.8, 1.2),
            time_to_depth=True,
            normalize=True,
            target_std=1.0,
        )
    finally:
        np.random.uniform = orig_uniform
        np.random.rand    = orig_rand

    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAugmentationPairCorrespondence:
    """x and y must agree at every non-zero (unmasked) position in x."""

    @pytest.fixture(scope="class")
    def zarr_path(self):
        path = os.environ.get("DATA_PATH", DEFAULT_DATA_PATH)
        if not os.path.exists(path):
            pytest.skip(f"Test data not found at {path}")
        return path

    @pytest.fixture(scope="class")
    def sample(self, zarr_path):
        return load_subvolume(zarr_path)

    def _run_pair_and_mask(self, data: np.ndarray, seed: int):
        """Produce (masked_x, y) for one random seed."""
        np.random.seed(seed)
        x, y, _geom_mask, params = augment_pair_3d(
            data.copy(),
            z_stretch_range=(0.8, 1.2),
            xy_stretch_range=(0.8, 1.2),
            time_to_depth=True,
            normalize=True,
            target_std=1.0,
        )

        # Apply input-only masking (peak/trough + trace clusters) to x
        mask = create_mask_3d(x, trace_mask_ratio=TRACE_MASK_RATIO)
        masked_x, _, mask = apply_mask_to_seismic(x, mask)
        return masked_x, y, mask, params

    # ------------------------------------------------------------------
    # Test 1: Random seed — statistical augmentation
    # ------------------------------------------------------------------
    def test_nonzero_x_matches_y_random_seed(self, sample):
        """Non-zero voxels in masked x equal the corresponding y voxels."""
        masked_x, y, mask, params = self._run_pair_and_mask(sample, seed=0)

        nonzero = masked_x != 0.0
        assert nonzero.any(), "All voxels were masked — nothing to compare"

        mismatches = ~np.isclose(masked_x[nonzero], y[nonzero], rtol=0, atol=1e-5)
        n_mismatch = mismatches.sum()
        n_total    = nonzero.sum()

        assert n_mismatch == 0, (
            f"seed=0: {n_mismatch}/{n_total} non-zero x voxels differ from y.\n"
            f"  max |Δ| = {np.abs(masked_x[nonzero][mismatches] - y[nonzero][mismatches]).max():.6f}\n"
            f"  params  = {params}"
        )

    # ------------------------------------------------------------------
    # Test 2: All augmentations forced on
    # ------------------------------------------------------------------
    def test_nonzero_x_matches_y_all_augmentations_forced(self, sample):
        """
        With every augmentation branch forced (stretch, t2d, all flips,
        swap_xy, noise=zeros), non-zero x voxels must still equal y.
        """
        x_aug, y_aug, _geom_mask, params = _force_all_augmentations(sample.copy())

        # Apply masking to x only
        mask = create_mask_3d(x_aug, trace_mask_ratio=TRACE_MASK_RATIO)
        masked_x, _, mask = apply_mask_to_seismic(x_aug, mask)

        nonzero = masked_x != 0.0
        assert nonzero.any(), "All voxels were masked — nothing to compare"

        mismatches = ~np.isclose(masked_x[nonzero], y_aug[nonzero], rtol=0, atol=1e-5)
        n_mismatch = mismatches.sum()
        n_total    = nonzero.sum()

        assert n_mismatch == 0, (
            f"forced-all: {n_mismatch}/{n_total} non-zero x voxels differ from y.\n"
            f"  max |Δ| = {np.abs(masked_x[nonzero][mismatches] - y_aug[nonzero][mismatches]).max():.6f}\n"
            f"  params  = {params}"
        )

    # ------------------------------------------------------------------
    # Test 3: Multiple seeds for robustness
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("seed", [1, 2, 3, 7, 42])
    def test_nonzero_x_matches_y_multiple_seeds(self, sample, seed):
        """Property holds across several independent random seeds."""
        masked_x, y, mask, params = self._run_pair_and_mask(sample, seed=seed)

        nonzero = masked_x != 0.0
        assert nonzero.any(), f"seed={seed}: all voxels masked"

        mismatches = ~np.isclose(masked_x[nonzero], y[nonzero], rtol=0, atol=1e-5)
        n_mismatch = mismatches.sum()

        assert n_mismatch == 0, (
            f"seed={seed}: {n_mismatch}/{nonzero.sum()} non-zero x voxels differ from y.\n"
            f"  params = {params}"
        )

    # ------------------------------------------------------------------
    # Test 4: Masked positions in x are exactly zero
    # ------------------------------------------------------------------
    def test_masked_positions_are_zero(self, sample):
        """apply_mask_to_seismic must zero every position where mask=False."""
        masked_x, y, mask, _ = self._run_pair_and_mask(sample, seed=5)

        masked_positions = ~mask
        assert masked_positions.any(), "No positions were masked"

        non_zero_in_masked = masked_x[masked_positions] != 0.0
        assert not non_zero_in_masked.any(), (
            f"{non_zero_in_masked.sum()} masked positions in x are not zero"
        )

    # ------------------------------------------------------------------
    # Test 5: Shapes are consistent
    # ------------------------------------------------------------------
    def test_shapes_consistent(self, sample):
        """x, y, and mask must all be the same shape as the input sample."""
        masked_x, y, mask, _ = self._run_pair_and_mask(sample, seed=99)
        assert masked_x.shape == SAMPLE_SHAPE, f"x shape {masked_x.shape} != {SAMPLE_SHAPE}"
        assert y.shape       == SAMPLE_SHAPE, f"y shape {y.shape} != {SAMPLE_SHAPE}"
        assert mask.shape    == SAMPLE_SHAPE, f"mask shape {mask.shape} != {SAMPLE_SHAPE}"
