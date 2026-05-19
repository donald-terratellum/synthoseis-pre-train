"""
Seismic 3D Masking Strategies
=============================
Implements masking for seismic pre-training:
- Peak/trough preservation along vertical axis
- Random trace masking with 3x3 cluster patterns
- Mask-value infill with zero or Gaussian noise
"""

import numpy as np
from typing import Tuple, Optional


def _generate_triangular_kernel(length: int, power: float) -> np.ndarray:
    """Generate normalized triangular kernel [1,3,5,...,peak,...,5,3,1]**power."""
    if length < 1 or length % 2 == 0:
        raise ValueError("kernel length must be a positive odd integer")
    if power <= 0:
        raise ValueError("kernel power must be > 0")

    mid = (length + 1) // 2
    kernel = np.concatenate([
        np.arange(1, 2 * mid, 2, dtype=np.float64),
        np.arange(2 * mid - 3, 0, -2, dtype=np.float64),
    ])
    kernel = np.power(kernel, float(power))
    kernel /= kernel.sum()
    return kernel


def _prefilter_along_z(
    seismic_data: np.ndarray,
    kernel_length: int = 19,
    kernel_power: float = 1.6,
) -> np.ndarray:
    """Prefilter seismic along z only, for extrema index detection."""
    kernel = _generate_triangular_kernel(kernel_length, kernel_power)
    # Keep this temporary array float64 to match strict-neighbor comparisons.
    return np.apply_along_axis(
        lambda trace: np.convolve(trace.astype(np.float64, copy=False), kernel, mode="same"),
        axis=0,
        arr=seismic_data,
    )


def _expected_cluster_footprint(cluster_shape: int) -> float:
    """Approximate expected unique footprint of a single cluster center.

    For small cluster shapes this is simply cluster_shape**2, but this helper
    keeps the name explicit for possible future refinements.
    """
    return float(cluster_shape * cluster_shape)


def _estimate_center_count_for_target_mask_ratio(
    width: int,
    height: int,
    target_mask_fraction: float,
    cluster_shape: int,
    cluster_prob: float,
) -> int:
    """Estimate how many cluster centers are needed to reach target masked fraction.

    Use a simple Poisson-approximation: expected masked traces per center =
    footprint * cluster_prob. Solve n_centers * footprint * cluster_prob / (W*H) ~= target.
    """
    total_traces = width * height
    footprint = _expected_cluster_footprint(cluster_shape)
    if footprint * cluster_prob <= 0:
        return 0
    est = target_mask_fraction * total_traces / (footprint * cluster_prob)
    return max(1, int(round(est)))


def create_mask_3d(
    seismic_data: np.ndarray,
    target_masked_fraction: Optional[float] = None,
    trace_mask_ratio: Optional[float] = None,
    cluster_prob: float = 0.8,
    cluster_shape: int = 3,
    center_selection_method: str = "random_mixture",
    random_seed: Optional[int] = None,
    extrema_prefilter_kernel_length: int = 19,
    extrema_prefilter_power: float = 1.6,
) -> np.ndarray:
    """
    Create a 3D mask for seismic data with clustered trace masking.

    Backwards-compatible: callers may pass either ``trace_mask_ratio`` (legacy)
    or ``target_masked_fraction`` (preferred). If both are provided,
    ``target_masked_fraction`` takes precedence.

    Args:
        seismic_data: 3D seismic array (z, x, y) - depth, width, height
        target_masked_fraction: fraction of traces to mask (0-1). If None,
            ``trace_mask_ratio`` is used. Default ~0.15 when both are None.
        trace_mask_ratio: legacy alias for target_masked_fraction.
        cluster_prob: per-voxel probability inside a chosen cluster to be masked
        cluster_shape: side length of square cluster (odd integer, e.g. 3)
        center_selection_method: sampling method for cluster centers
        random_seed: Optional RNG seed
        extrema_prefilter_kernel_length: odd kernel length for temporary
            peak/trough index prefilter along z
        extrema_prefilter_power: kernel exponent for temporary peak/trough
            index prefilter along z

    Returns:
        mask: Boolean array where True = preserve, False = masked
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Backward-compatible alias handling
    if target_masked_fraction is None:
        if trace_mask_ratio is None:
            target_masked_fraction = 0.15
        else:
            target_masked_fraction = float(trace_mask_ratio)

    # Basic validation / coercion
    if cluster_shape < 1:
        raise ValueError("cluster_shape must be >= 1")
    if cluster_shape % 2 == 0:
        raise ValueError("cluster_shape must be odd")

    shape = seismic_data.shape  # (z, x, y)
    z, x, y = shape
    mask = np.ones(shape, dtype=bool)

    # Preserve local peaks/troughs along depth axis (z) when possible
    if z > 2:
        # Filter only for index detection; raw seismic values remain untouched.
        prefiltered = _prefilter_along_z(
            seismic_data,
            kernel_length=int(extrema_prefilter_kernel_length),
            kernel_power=float(extrema_prefilter_power),
        )

        nonzero_z = np.where(seismic_data.any(axis=(1, 2)))[0]
        boundary_lo = None
        boundary_hi = None
        if nonzero_z.size > 0:
            z_lo = int(nonzero_z[0])
            z_hi = int(nonzero_z[-1])
            if z_lo > 0 or z_hi < z - 1:
                work = prefiltered.copy()
                if z_lo > 0:
                    work[z_lo - 1] = work[z_lo]
                    boundary_lo = z_lo - 1
                if z_hi < z - 1:
                    work[z_hi + 1] = work[z_hi]
                    boundary_hi = z_hi + 1
            else:
                work = prefiltered
        else:
            work = prefiltered

        # is_peak = (work[1:-1, :, :] > work[:-2, :, :]) & (work[1:-1, :, :] > work[2:, :, :])
        # is_trough = (work[1:-1, :, :] < work[:-2, :, :]) & (work[1:-1, :, :] < work[2:, :, :])
        is_peak = (work[1:-1, :, :] >= work[:-2, :, :]) & (work[1:-1, :, :] >= work[2:, :, :])
        is_trough = (work[1:-1, :, :] <= work[:-2, :, :]) & (work[1:-1, :, :] <= work[2:, :, :])

        peaks_troughs = is_peak | is_trough
        mask[1:-1, :, :] = peaks_troughs
        mask[0, :, :] = False
        mask[-1, :, :] = False
        if boundary_lo is not None:
            mask[boundary_lo, :, :] = False
        if boundary_hi is not None:
            mask[boundary_hi, :, :] = False

        # Explicitly release temporary prefiltered volumes.
        del prefiltered
        if work is not seismic_data:
            del work

    # Clustered trace masking
    total_traces = x * y
    n_centers = _estimate_center_count_for_target_mask_ratio(
        width=x, height=y, target_mask_fraction=target_masked_fraction,
        cluster_shape=cluster_shape, cluster_prob=cluster_prob
    )

    # select centers
    rng = np.random.default_rng(random_seed)
    centers = _select_cluster_centers(x, y, n_centers, center_selection_method, rng)

    half = cluster_shape // 2
    for cx, cy in centers:
        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                xx = cx + dx
                yy = cy + dy
                if 0 <= xx < x and 0 <= yy < y:
                    if np.random.random() < cluster_prob:
                        mask[:, xx, yy] = False

    return mask


def apply_mask_to_seismic(
    seismic_data: np.ndarray,
    mask: np.ndarray,
    fill_value: float = 0.0,
    fill_method: str = "zero",
    noise_std: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply mask to seismic data.
    
    Args:
        seismic_data: 3D seismic array (x, y, z)
        mask: Boolean mask (True = preserve, False = mask)
        fill_value: Value to fill masked positions when fill_method="zero"
        fill_method: Infill strategy for masked voxels: "zero" or "gaussian"
        noise_std: Standard deviation for Gaussian infill with mean 0
    
    Returns:
        masked_data: Seismic data with masked positions filled
        original_data: Original full seismic data for reconstruction loss
        mask: Boolean mask used to identify masked voxels
    """
    if fill_method not in ("zero", "gaussian"):
        raise ValueError("fill_method must be one of: zero, gaussian")
    if noise_std < 0:
        raise ValueError("noise_std must be >= 0")

    masked_data = seismic_data.copy()
    if fill_method == "gaussian":
        masked_count = int((~mask).sum())
        if masked_count > 0:
            noise = np.random.normal(loc=0.0, scale=float(noise_std), size=masked_count)
            masked_data[~mask] = noise.astype(masked_data.dtype, copy=False)
    else:
        masked_data[~mask] = fill_value
    
    original_data = seismic_data.copy()
    return masked_data, original_data, mask


def normalize_seismic(
    seismic_data: np.ndarray,
    target_std: float = 1.0
) -> Tuple[np.ndarray, float, float]:
    """
    Normalize seismic data to have specified standard deviation.
    
    Args:
        seismic_data: 3D seismic array
        target_std: Target standard deviation (default: 1.0)
    
    Returns:
        normalized: Normalized seismic data
        mean: Original mean (for denormalization)
        std: Original std (for denormalization)
    """
    mean = np.mean(seismic_data)
    mean = 0.0 # Centering to zero mean for seismic data
    std = np.std(seismic_data)
    
    if std > 0:
        normalized = (seismic_data - mean) / std * target_std
    else:
        normalized = seismic_data
    
    return normalized.astype(np.float32), mean, std


# ---------------------------------------------------------------------------
# Public helper: cluster-center sampling used by `create_mask_3d`
# The original implementation exposed several internal helpers; tests
# import `_select_cluster_centers` directly. Provide a clear top-level
# implementation here so it is importable and deterministic given a
# numpy.random.Generator.
# ---------------------------------------------------------------------------

def _indices_to_xy(indices: np.ndarray, width: int) -> list[tuple[int, int]]:
    flat = np.asarray(indices, dtype=np.int64).reshape(-1)
    return [(int(idx % width), int(idx // width)) for idx in flat]


def _sample_uniform_random_centers(width: int, height: int, n_centers: int, rng: np.random.Generator):
    population = width * height
    n = max(0, min(int(n_centers), population))
    if n == 0:
        return []
    indices = rng.choice(population, size=n, replace=False)
    return _indices_to_xy(indices, width)


def _sample_mitchell_best_candidate_centers(width: int, height: int, n_centers: int, rng: np.random.Generator, n_candidates: int = 10):
    population = width * height
    n = max(0, min(int(n_centers), population))
    if n == 0:
        return []
    if n == 1:
        return _indices_to_xy(np.array([int(rng.integers(0, population))]), width)

    xs = np.arange(population, dtype=np.float32) % float(width)
    ys = np.arange(population, dtype=np.float32) // float(width)

    selected = np.zeros(population, dtype=bool)
    first = int(rng.integers(0, population))
    selected[first] = True
    selected_indices = [first]

    nearest_dist2 = (xs - xs[first]) ** 2 + (ys - ys[first]) ** 2
    nearest_dist2[first] = 0.0

    for _ in range(1, n):
        available = np.flatnonzero(~selected)
        if available.size == 0:
            break
        c = min(int(n_candidates), int(available.size))
        candidate_idx = rng.choice(available, size=c, replace=False)
        best = int(candidate_idx[np.argmax(nearest_dist2[candidate_idx])])
        selected[best] = True
        selected_indices.append(best)
        dist2_new = (xs - xs[best]) ** 2 + (ys - ys[best]) ** 2
        nearest_dist2 = np.minimum(nearest_dist2, dist2_new)
        nearest_dist2[selected] = 0.0

    return _indices_to_xy(np.asarray(selected_indices, dtype=np.int64), width)


def _sample_poisson_disc_centers(width: int, height: int, n_centers: int, rng: np.random.Generator):
    # Simple coverage-oriented approximation: tile the domain with a grid
    # sized to roughly match the target count, then jitter and select a
    # deterministic subset. This produces broad coverage similar to
    # Bridson-style Poisson-disc sampling but is faster and deterministic
    # for our unit-test needs.
    population = width * height
    n = max(0, min(int(n_centers), population))
    if n == 0:
        return []

    # Estimate grid spacing to get approximately n cells
    approx_cells = max(1, int(np.sqrt(max(1, population / n))))
    step_x = max(1, int(width // max(1, approx_cells)))
    step_y = max(1, int(height // max(1, approx_cells)))

    candidates = []
    for gx in range(0, width, step_x):
        for gy in range(0, height, step_y):
            # jitter inside cell
            jx = int(min(width - 1, max(0, gx + int(rng.integers(0, step_x)))))
            jy = int(min(height - 1, max(0, gy + int(rng.integers(0, step_y)))))
            candidates.append((jx, jy))

    # If we have fewer candidates than needed, fall back to uniform sampling
    if len(candidates) <= n:
        # Fill remaining with random unique picks
        remaining = n - len(candidates)
        if remaining > 0:
            flat = rng.choice(population, size=remaining, replace=False)
            candidates.extend(_indices_to_xy(flat, width))

    # Shuffle deterministically and pick first n
    rng.shuffle(candidates)
    return candidates[:n]


def _select_cluster_centers(width: int, height: int, n_centers: int, method: str, rng: np.random.Generator):
    # Normalize method aliases
    normalized = (method or "").strip().lower().replace("-", "_")
    if normalized in ("random_mixture", "random_choice", "mixed", "mixture", "random"):
        normalized = "random_mixture"
    if normalized in ("mitchell", "best_candidate"):
        normalized = "mitchell_best_candidate"
    if normalized in ("poisson", "bridson"):
        normalized = "poisson_disc"
    if normalized not in ("random_mixture", "mitchell_best_candidate", "poisson_disc", "uniform_random"):
        # allow explicit canonical uniform
        if normalized == "uniform":
            normalized = "uniform_random"
        else:
            raise ValueError("center_selection_method must be one of: random_mixture, mitchell_best_candidate, poisson_disc, uniform_random")

    if normalized == "random_mixture":
        # mix strategies by picking one of the three
        choice = int(rng.integers(0, 3))
        if choice == 0:
            return _sample_uniform_random_centers(width, height, n_centers, rng)
        elif choice == 1:
            return _sample_mitchell_best_candidate_centers(width, height, n_centers, rng)
        else:
            return _sample_poisson_disc_centers(width, height, n_centers, rng)

    if normalized == "mitchell_best_candidate":
        # split into two passes for diversity (matches prior behaviour)
        n1 = n_centers // 2
        n2 = n_centers - n1
        return _sample_mitchell_best_candidate_centers(width, height, n1, rng) + _sample_mitchell_best_candidate_centers(width, height, n2, rng)

    if normalized == "poisson_disc":
        return _sample_poisson_disc_centers(width, height, n_centers, rng)

    # uniform_random
    return _sample_uniform_random_centers(width, height, n_centers, rng)
