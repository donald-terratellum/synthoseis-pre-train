"""Unit tests for trace-center sampling methods in create_mask_3d."""

import numpy as np
import pytest

from synthoseis_pre_train.masking import _select_cluster_centers, create_mask_3d


@pytest.mark.parametrize(
    "method",
    ["mitchell_best_candidate", "poisson_disc", "uniform_random"],
)
def test_create_mask_3d_supports_all_center_methods(method: str):
    """Each supported center method should produce a valid boolean mask."""
    # Build a small synthetic seismic cube with enough traces for cluster tests.
    seismic = np.random.default_rng(123).normal(size=(8, 16, 16)).astype(np.float32)

    # Run masking with deterministic settings and full in-cluster masking.
    mask = create_mask_3d(
        seismic,
        trace_mask_ratio=0.2,
        cluster_prob=1.0,
        random_seed=7,
        center_selection_method=method,
    )

    # Validate basic mask shape/type contract.
    assert mask.shape == seismic.shape
    assert mask.dtype == np.bool_

    # Expect at least one masked voxel for this non-trivial setup.
    assert (~mask).any()


def test_create_mask_3d_method_aliases_are_supported():
    """Common aliases should normalize to the canonical methods."""
    # Build fixed synthetic data so comparisons are stable.
    seismic = np.random.default_rng(1234).normal(size=(6, 12, 12)).astype(np.float32)

    # Mitchell alias should behave exactly like canonical spelling for same seed.
    mask_alias = create_mask_3d(
        seismic,
        random_seed=9,
        center_selection_method="mitchell",
    )
    mask_canon = create_mask_3d(
        seismic,
        random_seed=9,
        center_selection_method="mitchell_best_candidate",
    )

    # Uniform alias should behave exactly like canonical spelling for same seed.
    mask_uniform_alias = create_mask_3d(
        seismic,
        random_seed=11,
        center_selection_method="uniform",
    )
    mask_uniform_canon = create_mask_3d(
        seismic,
        random_seed=11,
        center_selection_method="uniform_random",
    )

    # Alias normalization should preserve exact output with deterministic seeds.
    assert np.array_equal(mask_alias, mask_canon)
    assert np.array_equal(mask_uniform_alias, mask_uniform_canon)


def test_create_mask_3d_default_is_random_mixture():
    """Default center-selection behavior should match explicit random-mixture mode."""
    # Build fixed synthetic data for deterministic comparison.
    seismic = np.random.default_rng(99).normal(size=(8, 14, 13)).astype(np.float32)

    # Compare default invocation vs explicit canonical method.
    default_mask = create_mask_3d(seismic, random_seed=21)
    explicit_mask = create_mask_3d(
        seismic,
        random_seed=21,
        center_selection_method="random_mixture",
    )

    # Both calls should produce identical masks with the same seed.
    assert np.array_equal(default_mask, explicit_mask)


def test_create_mask_3d_reproducible_per_method():
    """A fixed random seed should make each method deterministic."""
    # Build fixed synthetic data and test each sampling mode independently.
    seismic = np.random.default_rng(3).normal(size=(7, 10, 11)).astype(np.float32)
    methods = ["mitchell_best_candidate", "poisson_disc", "uniform_random"]

    # Compare two runs per method using identical arguments.
    for method in methods:
        mask_a = create_mask_3d(
            seismic,
            trace_mask_ratio=0.25,
            cluster_prob=0.8,
            random_seed=42,
            center_selection_method=method,
        )
        mask_b = create_mask_3d(
            seismic,
            trace_mask_ratio=0.25,
            cluster_prob=0.8,
            random_seed=42,
            center_selection_method=method,
        )
        assert np.array_equal(mask_a, mask_b)


def test_create_mask_3d_rejects_invalid_center_method():
    """Unknown center-selection methods should raise a clear ValueError."""
    # Build synthetic data for method validation.
    seismic = np.random.default_rng(1).normal(size=(5, 8, 8)).astype(np.float32)

    # Ensure unsupported method names fail fast with a helpful message.
    with pytest.raises(ValueError, match="center_selection_method"):
        create_mask_3d(seismic, center_selection_method="not_a_method")


@pytest.mark.parametrize(
    "method",
    ["uniform_random", "mitchell_best_candidate", "poisson_disc"],
)
def test_masked_fraction_is_close_to_target_after_clustering(method: str):
    """Final masked fraction should be close to requested trace_mask_ratio."""
    # Use z=2 to isolate cluster masking behavior from peak/trough preservation.
    seismic = np.zeros((2, 128, 128), dtype=np.float32)

    # Request 15% final masking with 80% in-cluster probability.
    mask = create_mask_3d(
        seismic,
        target_masked_fraction=0.15,
        cluster_prob=0.8,
        random_seed=123,
        center_selection_method=method,
    )

    # create_mask_3d semantics: False means masked.
    masked_fraction = float((~mask[0]).mean())

    # Stochastic tolerance chosen to allow method-specific variation.
    assert 0.12 <= masked_fraction <= 0.18


def test_poisson_disc_has_global_coverage():
    """Poisson-disc masking should not collapse into one local region."""
    # Use z=2 to isolate center distribution and cluster masking behavior.
    seismic = np.zeros((2, 128, 128), dtype=np.float32)

    # Build a deterministic Poisson-disc mask.
    mask = create_mask_3d(
        seismic,
        target_masked_fraction=0.15,
        cluster_prob=0.8,
        random_seed=125,
        center_selection_method="poisson_disc",
    )

    # Convert to 2D masked image where 1 means masked.
    masked = (~mask[0]).astype(np.uint8)

    # Partition into 4 quadrants and require each to contain masked traces.
    q1 = masked[:64, :64].sum()
    q2 = masked[:64, 64:].sum()
    q3 = masked[64:, :64].sum()
    q4 = masked[64:, 64:].sum()
    quadrant_counts = [int(q1), int(q2), int(q3), int(q4)]

    assert all(count > 0 for count in quadrant_counts)


def test_poisson_disc_centers_span_full_axis_ranges():
    """Poisson-disc centers should span nearly the full width and height."""
    rng = np.random.default_rng(125)
    centers = _select_cluster_centers(
        width=128,
        height=128,
        n_centers=280,
        method="poisson_disc",
        rng=rng,
    )

    xs = np.array([c[0] for c in centers], dtype=np.int64)
    ys = np.array([c[1] for c in centers], dtype=np.int64)

    assert int(xs.min()) <= 5
    assert int(xs.max()) >= 122
    assert int(ys.min()) <= 5
    assert int(ys.max()) >= 122


@pytest.mark.parametrize("cluster_shape", [3, 5, 7])
def test_create_mask_3d_supports_odd_cluster_shapes(cluster_shape: int):
    """Cluster shape argument should support odd sizes such as 3, 5, and 7."""
    seismic = np.zeros((2, 64, 64), dtype=np.float32)
    mask = create_mask_3d(
        seismic,
        target_masked_fraction=0.15,
        cluster_prob=0.8,
        cluster_shape=cluster_shape,
        random_seed=7,
        center_selection_method="uniform_random",
    )
    assert mask.shape == seismic.shape
    assert mask.dtype == np.bool_


def test_create_mask_3d_rejects_even_cluster_shape():
    """Even cluster sizes should be rejected because there is no center trace."""
    seismic = np.zeros((2, 32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="cluster_shape"):
        create_mask_3d(seismic, cluster_shape=4)
