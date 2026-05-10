"""
Seismic 3D Masking Strategies
=============================
Implements masking for seismic pre-training:
- Peak/trough preservation along vertical axis
- Random trace masking with 3x3 cluster patterns
- Zero-masking for masked voxels
"""

import numpy as np
from typing import Literal, Optional, Tuple


CenterSelectionMethod = Literal[
    "random_mixture",
    "mitchell_best_candidate",
    "poisson_disc",
    "uniform_random",
]


def _normalize_center_selection_method(method: str) -> CenterSelectionMethod:
    """Normalize method aliases to the canonical center-sampling method name.

    Args:
        method: User-provided center sampling method string.

    Returns:
        Canonical method name.

    Raises:
        ValueError: If the method name is not recognized.
    """
    # Normalize case/format so CLI-style and shorthand names map cleanly.
    normalized = method.strip().lower().replace("-", "_")

    # Accept common aliases while keeping one canonical internal spelling.
    aliases: dict[str, CenterSelectionMethod] = {
        "random_mixture": "random_mixture",
        "random_choice": "random_mixture",
        "mixed": "random_mixture",
        "mixture": "random_mixture",
        "random": "random_mixture",
        "mitchell": "mitchell_best_candidate",
        "best_candidate": "mitchell_best_candidate",
        "mitchell_best_candidate": "mitchell_best_candidate",
        "poisson": "poisson_disc",
        "bridson": "poisson_disc",
        "poisson_disc": "poisson_disc",
        "uniform": "uniform_random",
        "uniform_random": "uniform_random",
    }

    # Return the canonical value or fail with an explicit list of valid options.
    if normalized not in aliases:
        raise ValueError(
            "center_selection_method must be one of: "
            "random_mixture, mitchell_best_candidate, poisson_disc, uniform_random"
        )
    return aliases[normalized]


def _indices_to_xy(indices: np.ndarray, width: int) -> list[tuple[int, int]]:
    """Convert flattened linear indices to (x, y) integer coordinates."""
    # Flatten and convert to Python ints for stable downstream iteration.
    flat = np.asarray(indices, dtype=np.int64).reshape(-1)
    return [(int(idx % width), int(idx // width)) for idx in flat]


def _sample_uniform_random_centers(
    width: int,
    height: int,
    n_centers: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Sample center indices uniformly without replacement (current baseline)."""
    # Clamp to the finite number of available traces.
    population = width * height
    n = max(0, min(int(n_centers), population))
    if n == 0:
        return []

    # Draw unique flattened indices, then map to (x, y) coordinates.
    indices = rng.choice(population, size=n, replace=False)
    return _indices_to_xy(indices, width)


def _sample_mitchell_best_candidate_centers(
    width: int,
    height: int,
    n_centers: int,
    rng: np.random.Generator,
    n_candidates: int = 10,
) -> list[tuple[int, int]]:
    """Sample centers via Mitchell's best-candidate algorithm on a discrete grid.

    This follows Mike Bostock's published description by iteratively selecting
    the candidate that maximizes distance to the current point set.
    """
    # Clamp requested count and handle trivial requests early.
    population = width * height
    n = max(0, min(int(n_centers), population))
    if n == 0:
        return []
    if n == 1:
        return _indices_to_xy(np.array([rng.integers(0, population)]), width)

    # Build flattened grid coordinates for fast vectorized distance updates.
    xs = np.arange(population, dtype=np.float32) % float(width)
    ys = np.arange(population, dtype=np.float32) // float(width)

    # Seed with one random point, then track selected points in a boolean mask.
    selected = np.zeros(population, dtype=bool)
    first = int(rng.integers(0, population))
    selected[first] = True
    selected_indices = [first]

    # Track each trace's nearest squared distance to the selected set.
    nearest_dist2 = (xs - xs[first]) ** 2 + (ys - ys[first]) ** 2
    nearest_dist2[first] = 0.0

    # Grow the point set by repeatedly choosing the best of random candidates.
    for _ in range(1, n):
        # Pick candidate traces from currently unselected traces.
        available = np.flatnonzero(~selected)
        if available.size == 0:
            break
        c = min(int(n_candidates), int(available.size))
        candidate_idx = rng.choice(available, size=c, replace=False)

        # Choose candidate maximizing nearest-neighbor distance.
        best = int(candidate_idx[np.argmax(nearest_dist2[candidate_idx])])
        selected[best] = True
        selected_indices.append(best)

        # Update nearest-distance field with distances to the newly selected point.
        dist2_new = (xs - xs[best]) ** 2 + (ys - ys[best]) ** 2
        nearest_dist2 = np.minimum(nearest_dist2, dist2_new)
        nearest_dist2[selected] = 0.0

    # Convert final flattened indices to (x, y) center coordinates.
    return _indices_to_xy(np.asarray(selected_indices, dtype=np.int64), width)


def _sample_poisson_disc_centers(
    width: int,
    height: int,
    n_centers: int,
    rng: np.random.Generator,
    k: int = 30,
) -> list[tuple[int, int]]:
    """Sample centers using Bridson-style Poisson-disc sampling in 2D.

    This follows Mike Bostock's published Bridson algorithm description with an
    active list, annulus proposals in [r, 2r], and neighbor-cell distance checks.
    """
    # Clamp requested count and handle trivial geometry edge cases.
    population = width * height
    n = max(0, min(int(n_centers), population))
    if n == 0 or width <= 0 or height <= 0:
        return []
    if population == 1:
        return [(0, 0)]

    # Estimate radius from target density (slightly reduced to ease filling).
    area = float(width * height)
    target = max(float(n), 1.0)
    radius = max(1.0, np.sqrt(area / (np.pi * target)) * 0.90)
    radius2 = radius * radius

    # Create acceleration grid where each cell side is r / sqrt(2).
    cell = radius / np.sqrt(2.0)
    grid_w = max(1, int(np.ceil(width / cell)))
    grid_h = max(1, int(np.ceil(height / cell)))
    grid = -np.ones((grid_h, grid_w), dtype=np.int64)

    # Keep continuous points, active list, and accepted integer trace centers.
    points: list[tuple[float, float]] = []
    active: list[int] = []
    centers: list[tuple[int, int]] = []
    center_set: set[tuple[int, int]] = set()

    def _is_valid_point(px: float, py: float) -> bool:
        """Check if a candidate point obeys the Poisson min-distance rule."""
        gx = min(grid_w - 1, int(px / cell))
        gy = min(grid_h - 1, int(py / cell))
        x0g = max(0, gx - 2)
        x1g = min(grid_w - 1, gx + 2)
        y0g = max(0, gy - 2)
        y1g = min(grid_h - 1, gy + 2)
        for yy in range(y0g, y1g + 1):
            for xx in range(x0g, x1g + 1):
                pidx = int(grid[yy, xx])
                if pidx < 0:
                    continue
                qx, qy = points[pidx]
                if (px - qx) ** 2 + (py - qy) ** 2 < radius2:
                    return False
        return True

    def _insert_point(px: float, py: float) -> None:
        """Insert a validated point into the active list, grid, and center set."""
        points.append((px, py))
        new_idx = len(points) - 1
        active.append(new_idx)
        gx = min(grid_w - 1, int(px / cell))
        gy = min(grid_h - 1, int(py / cell))
        grid[gy, gx] = new_idx
        center = (min(width - 1, int(px)), min(height - 1, int(py)))
        if center not in center_set:
            centers.append(center)
            center_set.add(center)

    # Seed with an initial random point.
    x0 = float(rng.uniform(0.0, width))
    y0 = float(rng.uniform(0.0, height))
    _insert_point(x0, y0)

    # Grow samples to exhaustion. When the active list empties, attempt to
    # start a new valid seed so coverage is not confined to one local region.
    while True:
        if not active:
            seeded = False
            for _ in range(1024):
                sx = float(rng.uniform(0.0, width))
                sy = float(rng.uniform(0.0, height))
                if _is_valid_point(sx, sy):
                    _insert_point(sx, sy)
                    seeded = True
                    break
            if not seeded:
                break

        # Pick a random active seed point.
        active_i = int(rng.integers(0, len(active)))
        seed_idx = active[active_i]
        sx, sy = points[seed_idx]
        accepted = False

        # Propose up to k annulus candidates around this seed.
        for _ in range(k):
            ang = float(rng.uniform(0.0, 2.0 * np.pi))
            rad = float(rng.uniform(radius, 2.0 * radius))
            cx = sx + rad * np.cos(ang)
            cy = sy + rad * np.sin(ang)

            # Reject candidates outside the domain bounds.
            if not (0.0 <= cx < width and 0.0 <= cy < height):
                continue

            # Enforce Poisson minimum distance against nearby accepted points.
            if not _is_valid_point(cx, cy):
                continue

            # Accept candidate, register it, and keep it active.
            _insert_point(cx, cy)
            accepted = True

        # Retire seed points that cannot produce a valid candidate.
        if not accepted:
            active.pop(active_i)

    # If we generated more than needed, pick a deterministic random subset.
    if len(centers) > n:
        keep_idx = rng.choice(len(centers), size=n, replace=False)
        centers = [centers[int(i)] for i in np.asarray(keep_idx, dtype=np.int64)]

    # Backfill shortfalls with uniform random unique centers.
    if len(centers) < n:
        all_indices = np.arange(population, dtype=np.int64)
        used_indices = {cy * width + cx for cx, cy in center_set}
        remaining = np.array([idx for idx in all_indices if int(idx) not in used_indices], dtype=np.int64)
        need = min(n - len(centers), int(remaining.size))
        if need > 0:
            fill = rng.choice(remaining, size=need, replace=False)
            centers.extend(_indices_to_xy(fill, width))

    # Return exactly n centers.
    return centers[:n]


def _expected_cluster_footprint(width: int, height: int, cluster_shape: int) -> float:
    """Return expected valid-trace count in a 3x3 cluster on a bounded grid.

    The expectation is over uniformly random center locations. Edge centers have
    clipped neighborhoods, so this value is slightly below 9 on finite grids.
    """
    if width <= 0 or height <= 0:
        return 0.0
    half = cluster_shape // 2

    # For odd cluster width k=2h+1 on axis length W, expected valid count is:
    #   E[count] = k - h(h+1)/W
    expected_x = float(cluster_shape) - (float(half * (half + 1)) / float(width))
    expected_y = float(cluster_shape) - (float(half * (half + 1)) / float(height))
    return expected_x * expected_y


def _estimate_center_count_for_target_mask_ratio(
    width: int,
    height: int,
    trace_mask_ratio: float,
    cluster_prob: float,
    cluster_shape: int,
) -> int:
    """Estimate cluster-center count needed for target final masked fraction."""
    total_traces = width * height
    if total_traces <= 0:
        return 0
    if trace_mask_ratio <= 0.0 or cluster_prob <= 0.0:
        return 0

    footprint = _expected_cluster_footprint(width, height, cluster_shape)
    if footprint <= 0.0:
        return 0

    # Poisson-coverage approximation for final masked fraction:
    #   p_masked ~= 1 - exp(-(n_centers * footprint * cluster_prob)/N)
    # Solve for n_centers with p_masked = trace_mask_ratio.
    target = min(max(trace_mask_ratio, 0.0), 1.0 - 1e-9)
    est = -float(total_traces) * np.log(1.0 - target) / (footprint * float(cluster_prob))
    n_centers = int(np.round(est))
    return max(0, min(n_centers, total_traces))


def _select_cluster_centers(
    width: int,
    height: int,
    n_centers: int,
    method: str,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Select cluster-center coordinates using the requested sampling method."""
    # Normalize aliases first so all control flow uses canonical names.
    canonical = _normalize_center_selection_method(method)

    # Randomly choose one of the 3 core methods with equal probability.
    if canonical == "random_mixture":
        method_idx = int(rng.integers(0, 3))
        if method_idx == 0:
            chosen = "uniform_random"
        elif method_idx == 1:
            chosen = "mitchell_best_candidate"
        else:
            chosen = "poisson_disc"
        return _select_cluster_centers(width, height, n_centers, chosen, rng)

    # Use Mitchell best-candidate in two independent half-size passes by default.
    # Combining both pass outputs allows center overlap between passes.
    if canonical == "mitchell_best_candidate":
        n1 = n_centers // 2
        n2 = n_centers - n1
        pass1 = _sample_mitchell_best_candidate_centers(width, height, n1, rng)
        pass2 = _sample_mitchell_best_candidate_centers(width, height, n2, rng)
        return pass1 + pass2

    # Use Bridson Poisson-disc when explicitly requested.
    if canonical == "poisson_disc":
        return _sample_poisson_disc_centers(width, height, n_centers, rng)

    # Fall back to the original uniform-random center selection.
    return _sample_uniform_random_centers(width, height, n_centers, rng)


def create_mask_3d(
    seismic_data: np.ndarray,
    cluster_prob: float = 0.8,
    target_masked_fraction: float = 0.15,
    cluster_shape: int = 3,
    random_seed: Optional[int] = None,
    center_selection_method: str = "random_mixture",
) -> np.ndarray:
    """
    Create a 3D mask for seismic data.
    
    Args:
        seismic_data: 3D seismic array (z, x, y) - depth, width, height
        cluster_prob: Probability a trace in a 3x3 cluster is masked (0.8 = 80%)
        target_masked_fraction: Target final masked fraction after accounting for
            center count, cluster size, and in-cluster masking probability.
            This value determines the effective final masked fraction.
        cluster_shape: Odd cluster edge size (e.g. 3, 5, 7). Defines a
            cluster_shape x cluster_shape neighborhood around each center.
        random_seed: Optional random seed for reproducibility
        center_selection_method: Method used to pick 3x3 cluster centers.
            Supported values: "random_mixture" (default),
            "mitchell_best_candidate", "poisson_disc", "uniform_random".
    
    Returns:
        mask: Boolean array where True = preserve, False = mask (zero)
    """
    # Validate user-facing masking probabilities.
    if not (0.0 <= cluster_prob <= 1.0):
        raise ValueError("cluster_prob must be in [0, 1]")
    if not (0.0 <= target_masked_fraction <= 1.0):
        raise ValueError("target_masked_fraction must be in [0, 1]")
    if cluster_shape <= 0 or cluster_shape % 2 == 0:
        raise ValueError("cluster_shape must be a positive odd integer")

    # Use the explicit target masked fraction provided by the caller.
    effective_target_mask = float(target_masked_fraction)

    # Build a local RNG so reproducibility is explicit and side-effect free.
    rng = np.random.default_rng(random_seed)

    # Initialize dimensions and start with all voxels preserved.
    shape = seismic_data.shape  # (z, x, y)
    z, x, y = shape
    mask = np.ones(shape, dtype=bool)
    
    # Step 1: Identify local peaks and troughs along vertical (z) axis (axis 0)
    # Preserve only voxels that are local extrema relative to immediate neighbors in z axis
    # A peak: value > previous AND value > next
    # A trough: value < previous AND value < next
    if z > 2:
        # When a squeeze augmentation leaves zero-filled planes at the top or bottom
        # of the cube, the 0 → nonzero transition at z_lo (or nonzero → 0 at z_hi)
        # makes every voxel in the first/last real plane appear as a local extremum
        # (any nonzero value is trivially > 0 or < 0 relative to the zero neighbor).
        # Fix: before running the detector, temporarily fill the immediately adjacent
        # zero plane with a copy of the real boundary plane.  After detection those
        # filled planes are forced to False so they are never exposed in the input.
        nonzero_z = np.where(seismic_data.any(axis=(1, 2)))[0]
        boundary_lo = None
        boundary_hi = None
        if nonzero_z.size > 0:
            z_lo = int(nonzero_z[0])
            z_hi = int(nonzero_z[-1])
            if z_lo > 0 or z_hi < z - 1:
                work = seismic_data.copy()
                if z_lo > 0:
                    work[z_lo - 1] = work[z_lo]   # fill adjacent zero plane
                    boundary_lo = z_lo - 1
                if z_hi < z - 1:
                    work[z_hi + 1] = work[z_hi]   # fill adjacent zero plane
                    boundary_hi = z_hi + 1
            else:
                work = seismic_data
        else:
            work = seismic_data

        is_peak = (work[1:-1, :, :] > work[:-2, :, :]) & \
                  (work[1:-1, :, :] > work[2:, :, :])
        is_trough = (work[1:-1, :, :] < work[:-2, :, :]) & \
                    (work[1:-1, :, :] < work[2:, :, :])
        
        # Create mask for peaks and troughs (middle z indices)
        peaks_troughs = is_peak | is_trough
        
        # Mask all voxels except peaks and troughs
        mask[1:-1, :, :] = peaks_troughs
        # Edge voxels (z=0 and z=z-1) are not preserved (no neighbors for comparison)
        mask[0, :, :] = False
        mask[-1, :, :] = False

        # Force the temporarily-filled zero planes back to masked
        if boundary_lo is not None:
            mask[boundary_lo, :, :] = False
        if boundary_hi is not None:
            mask[boundary_hi, :, :] = False
    
    # Step 2: Choose center count to target FINAL masked fraction after
    # random in-cluster masking and 3x3 neighborhood expansion.
    n_traces_to_mask = _estimate_center_count_for_target_mask_ratio(
        width=x,
        height=y,
        trace_mask_ratio=effective_target_mask,
        cluster_prob=cluster_prob,
        cluster_shape=cluster_shape,
    )

    # Step 3: Select cluster centers using the requested trace sampling method.
    centers = _select_cluster_centers(
        width=x,
        height=y,
        n_centers=n_traces_to_mask,
        method=center_selection_method,
        rng=rng,
    )

    # Step 4: Apply cluster masking around each selected center.
    half = cluster_shape // 2
    for trace_x, trace_y in centers:
        # Iterate through offsets in the cluster_shape x cluster_shape neighborhood.
        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                cx = trace_x + dx
                cy = trace_y + dy

                # Only consider valid in-bounds trace coordinates.
                if 0 <= cx < x and 0 <= cy < y:
                    # Retain existing probabilistic masking within each cluster.
                    if float(rng.random()) < cluster_prob:
                        # Mask entire vertical trace (all z for this x,y)
                        mask[:, cx, cy] = False
    
    return mask


def apply_mask_to_seismic(
    seismic_data: np.ndarray,
    mask: np.ndarray,
    fill_value: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply mask to seismic data.
    
    Args:
        seismic_data: 3D seismic array (x, y, z)
        mask: Boolean mask (True = preserve, False = mask)
        fill_value: Value to fill masked positions (default: 0)
    
    Returns:
        masked_data: Seismic data with masked positions filled
        original_data: Original full seismic data for reconstruction loss
        mask: Boolean mask used to identify masked voxels
    """
    masked_data = seismic_data.copy()
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
