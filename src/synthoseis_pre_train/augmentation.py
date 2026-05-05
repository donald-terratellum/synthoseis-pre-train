"""
Seismic 3D Data Augmentation
=============================
Implements domain-specific augmentations for seismic data:
- Stretch/squeeze in x, y, z axes
- Time-to-depth conversion simulation
- Edge masking for squeeze operations
"""

import numpy as np
from typing import Tuple, Optional, List
from scipy.ndimage import zoom


def _next_regular(n: int) -> int:
    """Return the smallest integer >= n whose only prime factors are 2, 3, and 5.

    These "5-smooth" (regular) sizes make numpy FFT use pure split-radix passes
    and avoid the slow O(n log n) fallback that fires on prime-length inputs.
    """
    if n <= 1:
        return 1
    m = n
    while True:
        x = m
        while x % 2 == 0: x //= 2
        while x % 3 == 0: x //= 3
        while x % 5 == 0: x //= 5
        if x == 1:
            return m
        m += 1


def phase_rotation_3d(data: np.ndarray, phase_deg: float) -> np.ndarray:
    """Apply a constant phase rotation to every trace (z-axis) of a 3D seismic volume.

    A single angle phi is applied uniformly — every (x,y) trace is rotated by
    the same phase.  The full (z, x, y) array is transformed in one vectorized
    rfft call so there are no Python-level trace loops.

    The z-axis is zero-padded to the next 5-smooth size before the FFT to keep
    numpy in its efficient split-radix code path, then the result is trimmed back
    to the original z length.

    Args:
        data:      3D array (z, x, y), float32 or float64
        phase_deg: Rotation angle in degrees; U(-180, 180) gives full-cycle coverage

    Returns:
        Rotated array, same shape and dtype as input
    """
    z_size = data.shape[0]
    pad_z  = _next_regular(z_size)

    # rfft along axis=0; shape after: (pad_z//2 + 1, x, y)
    F = np.fft.rfft(data, n=pad_z, axis=0)

    # Rotate all frequency components by the same phase angle
    F *= np.exp(1j * np.deg2rad(phase_deg))

    # Inverse rfft, trim padding, preserve original dtype
    rotated = np.fft.irfft(F, n=pad_z, axis=0)[:z_size]
    return rotated.astype(data.dtype)


def stretch_squeeze_3d(
    data: np.ndarray,
    scale_factors: Tuple[float, float, float],
    axes: Tuple[str, str, str] = ('x', 'y', 'z'),
    mask_edges: bool = True,
    random_edge_position: bool = True,
    output_shape: Optional[Tuple[int, int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply stretch/squeeze transformation to 3D seismic data.
    
    Args:
        data: 3D seismic array (z, x, y)
        scale_factors: (sz, sx, sy) scale factors > 1 = stretch, < 1 = squeeze
        axes: Axis labels (for documentation)
        mask_edges: If True, mask edge regions when squeezing
        random_edge_position: If True, place masked edge at random position
    
    Returns:
        transformed: Stretched/squeezed data
        edge_mask: Boolean mask where True = edge region (masked)
    """
    sz, sx, sy = scale_factors
    
    # Apply zoom (interpolation).
    # Data axes are (z, x, y); zoom_factors must match that order: (sz, sx, sy).
    zoom_factors = (sz, sx, sy)
    transformed = zoom(data, zoom_factors, order=1)
    
    # Create edge mask for squeezed dimensions
    edge_mask = np.ones_like(transformed, dtype=bool)
    
    if mask_edges:
        original_shape = data.shape
        
        for axis_idx, scale_val in enumerate((sz, sx, sy)):
            if scale_val < 1.0:  # Squeezing
                new_size = int(original_shape[axis_idx] * scale_val)
                
                if random_edge_position:
                    # Random position for the actual data within larger array
                    max_offset = transformed.shape[axis_idx] - new_size
                    if max_offset > 0:
                        offset = np.random.randint(0, max_offset)
                    else:
                        offset = 0
                else:
                    offset = 0
                
                # Create mask for edge regions
                if offset > 0:
                    edge_mask[tuple([slice(None)] * axis_idx + [slice(0, offset)])] = False
                if offset + new_size < transformed.shape[axis_idx]:
                    edge_mask[tuple([slice(None)] * axis_idx + [slice(offset + new_size, None)])] = False
    
    final_shape = output_shape if output_shape is not None else original_shape
    transformed = crop_or_pad_to_shape(transformed, final_shape)
    edge_mask = crop_or_pad_to_shape(edge_mask, final_shape)
    return transformed, edge_mask


def crop_or_pad_to_shape(data: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Crop or pad an array to the target shape using centered cropping/padding.
    """
    current_shape = data.shape
    if current_shape == target_shape:
        return data

    # Crop to target shape
    slices = []
    pads = []
    for current, target in zip(current_shape, target_shape):
        if current > target:
            start = (current - target) // 2
            slices.append(slice(start, start + target))
            pads.append((0, 0))
        else:
            slices.append(slice(None))
            pad_total = target - current
            pads.append((pad_total // 2, pad_total - pad_total // 2))

    cropped = data[tuple(slices)]
    if any(pad != (0, 0) for pad in pads):
        pad_value = 0 if cropped.dtype != bool else False
        cropped = np.pad(cropped, pads, constant_values=pad_value)

    return cropped


def time_to_depth_simulation(
    data: np.ndarray,
    velocity_gradient: float = 0.5,
    depth_range: Tuple[float, float] = (0.0, 3000.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate time-to-depth conversion stretch/squeeze.
    
    In seismic, velocity typically increases with depth, causing:
    - Squeezing at shallow depths (time > depth)
    - Stretching at deep depths (time < depth)
    
    Args:
        data: 3D seismic array with z as time axis
        velocity_gradient: Rate of velocity increase (higher = more stretch/squeeze)
        depth_range: (min_depth, max_depth) in meters
    
    Returns:
        depth_data: Data stretched to simulate depth domain
        depth_mask: Mask for edge regions
    """
    z_size = data.shape[0]

    # Create depth-dependent stretch factor
    # At shallow: squeeze (factor < 1), at deep: stretch (factor > 1)
    z_indices = np.linspace(0, 1, z_size)
    stretch_factors = 1.0 + velocity_gradient * (z_indices - 0.5)

    # Compute stretched sample positions (same for every trace — only z-dependent).
    # Where these exceed [0, z_size-1], interpolation is out-of-bounds; those
    # z-planes are masked.  Clip to avoid runaway extrapolation values.
    original_indices = np.arange(z_size)
    stretched_indices = original_indices * stretch_factors
    valid_z = (stretched_indices >= 0) & (stretched_indices <= z_size - 1)
    stretched_indices_clipped = np.clip(stretched_indices, 0, z_size - 1)

    # Apply depth-dependent resampling along z for every (x, y) trace
    from scipy.interpolate import interp1d
    depth_data = np.zeros_like(data)
    depth_mask = np.ones_like(data, dtype=bool)

    for i in range(data.shape[1]):  # x
        for j in range(data.shape[2]):  # y
            trace = data[:, i, j]
            if len(trace) > 1:
                interp_func = interp1d(original_indices, trace,
                                       kind='linear',
                                       bounds_error=False,
                                       fill_value=(trace[0], trace[-1]))
                depth_data[:, i, j] = interp_func(stretched_indices_clipped)

    # Mask z-planes where stretched_indices were out of bounds
    depth_mask[~valid_z, :, :] = False

    return depth_data, depth_mask


def random_augmentation_3d(
    data: np.ndarray,
    z_stretch_range: Tuple[float, float] = (0.667, 1.5),
    xy_stretch_range: Tuple[float, float] = (0.8, 1.25),
    time_to_depth: bool = True,
    normalize: bool = True,
    target_std: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Apply random augmentations to seismic data.
    
    Args:
        data: 3D seismic array (x, y, z)
        z_stretch_range:  (min, max) scale factor for the z (time/depth) axis
        xy_stretch_range: (min, max) scale factor for the x and y spatial axes
        time_to_depth: Whether to apply time-to-depth simulation
        normalize: Whether to normalize to target std
        target_std: Target standard deviation after normalization
    
    Returns:
        augmented: Augmented seismic data
        combined_mask: Combined mask for all augmentations
        params: Dictionary of augmentation parameters used
    """
    x, y, z = data.shape
    
    # Independent random stretch/squeeze per axis with axis-specific ranges
    sx = np.random.uniform(*xy_stretch_range)
    sy = np.random.uniform(*xy_stretch_range)
    sz = np.random.uniform(*z_stretch_range)
    
    # Apply stretch/squeeze
    augmented, stretch_mask = stretch_squeeze_3d(
        data, (sx, sy, sz), mask_edges=True
    )
    
    # Apply time-to-depth simulation
    if time_to_depth and np.random.rand() < 0.6:
        velocity_grad = np.random.uniform(0.3, 0.8)
        augmented, td_mask = time_to_depth_simulation(
            augmented, velocity_gradient=velocity_grad
        )
        combined_mask = stretch_mask & td_mask
    else:
        velocity_grad = None
        combined_mask = stretch_mask

    # Additional random augmentations that preserve the shape
    if np.random.rand() < 0.5:
        augmented = np.flip(augmented, axis=0).copy()
    if np.random.rand() < 0.5:
        augmented = np.flip(augmented, axis=1).copy()
    if np.random.rand() < 0.5:
        augmented = np.flip(augmented, axis=2).copy()
    if np.random.rand() < 0.5:
        augmented = np.swapaxes(augmented, 1, 2).copy()  # swap x and y

    if np.random.rand() < 0.4:
        noise_scale = np.std(augmented) * 0.02
        augmented = augmented + np.random.normal(0.0, noise_scale, augmented.shape)

    if normalize:
        mean = np.mean(augmented)
        mean = 0.0 # Centering to zero mean for seismic data
        std = np.std(augmented)
        if std > 0:
            augmented = (augmented - mean) / std * target_std

    params = {
        'stretch_factors': (sx, sy, sz),
        'velocity_gradient': velocity_grad,
        'flip_x': bool(np.random.rand() < 0.5),
        'flip_y': bool(np.random.rand() < 0.5),
        'flip_z': bool(np.random.rand() < 0.5),
        'noise_added': np.random.rand() < 0.4,
        'normalized': normalize,
        'target_std': target_std
    }
    
    return augmented, combined_mask, params


def augment_pair_3d(
    cube: np.ndarray,
    target_shape: Tuple[int, int, int] = (128, 128, 128),
    z_artifact_margin: int = 0,
    z_stretch_range: Tuple[float, float] = (0.667, 1.5),
    xy_stretch_range: Tuple[float, float] = (0.8, 1.25),
    phase_range: Tuple[float, float] = (-180.0, 180.0),
    time_to_depth: bool = True,
    normalize: bool = True,
    target_std: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Produce an (x, y) training pair with shared geometric augmentations.

    Extraction and stretch/squeeze are coordinated: for squeeze axes the
    subvolume is extracted at (int(target/scale)+1) so that after zooming the
    full target_shape is filled with valid seismic data, eliminating the
    zero-padded masked borders that the old fixed-size extraction produced.

    Args:
        cube:            Full zarr volume in (x, y, z) axis order
        target_shape:    Output shape in (z, x, y) training order
        z_artifact_margin: z-indices to exclude at the deep end of the zarr
        z_stretch_range:  (min, max) scale factor for the z (time/depth) axis
        xy_stretch_range: (min, max) scale factor for the x and y spatial axes
        phase_range:    (min_deg, max_deg) constant phase shift range in degrees
        time_to_depth:   Whether to optionally apply time-to-depth simulation
        normalize:       Whether to normalise to target_std
        target_std:      Target standard deviation for normalisation

    Returns:
        x:             Input array (z, x, y) — geometrically augmented + optional noise
        y:             Label array (z, x, y) — geometrically augmented, clean
        combined_mask: Edge mask produced by geometric transforms
        params:        Dict recording all sampled parameters
    """
    # --- Sample all random decisions up front (independent ranges per axis) ---
    sx = np.random.uniform(*xy_stretch_range)
    sy = np.random.uniform(*xy_stretch_range)
    sz = np.random.uniform(*z_stretch_range)
    phase_deg = np.random.uniform(*phase_range)
    do_t2d = time_to_depth and np.random.rand() < 0.6
    velocity_grad = np.random.uniform(0.3, 0.8) if do_t2d else None
    do_flip_x  = np.random.rand() < 0.5
    do_flip_y  = np.random.rand() < 0.5
    do_swap_xy = np.random.rand() < 0.5

    # --- Extract a scale-aware subvolume so squeezed axes have no masked edges ---
    # For squeeze (scale < 1): extract int(target/scale)+1 so zoom produces >= target
    # valid samples → no zero-padding needed in crop_or_pad_to_shape.
    # For stretch (scale >= 1): extract target; zoom produces > target → center crop.
    target_z, target_x, target_y = target_shape
    ext_x = int(target_x / sx) + 1 if sx < 1.0 else target_x
    ext_y = int(target_y / sy) + 1 if sy < 1.0 else target_y
    ext_z = int(target_z / sz) + 1 if sz < 1.0 else target_z
    # extract_random_subvolume expects (x, y, z) zarr order
    raw = extract_random_subvolume(
        cube, (ext_x, ext_y, ext_z), z_artifact_margin=z_artifact_margin
    ).astype(np.float32)
    data = np.transpose(raw, (2, 0, 1))  # (ext_x, ext_y, ext_z) → (ext_z, ext_x, ext_y)

    # --- Phase rotation along z before normalisation ---
    # Applied on the raw extraction so normalisation sees the rotated amplitudes.
    data = phase_rotation_3d(data, phase_deg)

    # --- Derive normalisation statistics from extracted data BEFORE augmentation ---
    norm_mean = float(np.mean(data))
    norm_mean = 0.0 # Centering to zero mean for seismic data
    norm_std  = float(np.std(data))

    # --- Apply geometric transforms once → clean augmented volume ---
    # output_shape ensures the result is cropped to target_shape (not extraction shape).
    clean, stretch_mask = stretch_squeeze_3d(
        data, (sz, sx, sy), mask_edges=True, output_shape=target_shape
    )

    if do_t2d:
        clean, td_mask = time_to_depth_simulation(clean, velocity_gradient=velocity_grad)
        combined_mask = stretch_mask & td_mask
    else:
        combined_mask = stretch_mask

    # Apply the same flip/swap to both the data AND the mask so they stay aligned
    if do_flip_x:
        clean = np.flip(clean, axis=1).copy()
        combined_mask = np.flip(combined_mask, axis=1).copy()
    if do_flip_y:
        clean = np.flip(clean, axis=2).copy()
        combined_mask = np.flip(combined_mask, axis=2).copy()
    if do_swap_xy:
        clean = np.swapaxes(clean, 1, 2).copy()
        combined_mask = np.swapaxes(combined_mask, 1, 2).copy()

    # --- Normalise using pre-augmentation statistics (same for both x and y) ---
    if normalize and norm_std > 0:
        clean = (clean - norm_mean) / norm_std * target_std

    # --- Zero out interpolation edge artifacts in both x and y ---
    # combined_mask is False where stretch/squeeze or t2d left unreliable edge regions.
    # Applying it here ensures x and y share the same valid region; the dataloader's
    # trace masking (x-only) will add further zeros on top.
    clean[~combined_mask] = 0.0

    # --- Trim boundary z-planes to prevent false peak/trough extrema ---
    # After zeroing squeeze edges, the first and last z-planes that contain any
    # non-zero data sit at a 0→nonzero boundary. The peak/trough detector in
    # create_mask_3d will fire on those transition voxels, producing the spurious
    # horizontal band of extrema visible in TensorBoard.
    # Fix: mask out the single shallowest and deepest non-zero z-planes (all x,y)
    # in both clean and combined_mask so those transitions are never seen.
    nonzero_z = np.where(clean.any(axis=(1, 2)))[0]  # z-planes with any non-zero
    if nonzero_z.size >= 2:
        z_min, z_max = int(nonzero_z[0]), int(nonzero_z[-1])
        clean[z_min, :, :] = 0.0
        clean[z_max, :, :] = 0.0
        combined_mask[z_min, :, :] = False
        combined_mask[z_max, :, :] = False

    # y is the clean normalised result; x is identical before trace masking
    y = clean
    x = clean.copy()

    params = {
        'stretch_factors':  (sx, sy, sz),
        'phase_deg':        phase_deg,
        'velocity_gradient': velocity_grad,
        'flip_x':    do_flip_x,
        'flip_y':    do_flip_y,
        'swap_xy':   do_swap_xy,
        'normalized':  normalize,
        'target_std':  target_std,
        'norm_mean':   norm_mean,   # required to denormalise during inference
        'norm_std':    norm_std,    # required to denormalise during inference
    }

    return x.astype(np.float32), y.astype(np.float32), combined_mask, params


def extract_random_subvolume(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int],
    random_seed: Optional[int] = None,
    z_artifact_margin: int = 0,
) -> np.ndarray:
    """
    Extract random subvolume from larger seismic volume.
    
    Args:
        volume: Large 3D seismic array (x, y, z) in zarr axis order
        target_shape: (x, y, z) shape of subvolume
        random_seed: Optional random seed
        z_artifact_margin: Number of z-indices at the deep end to exclude.
            The last valid z start position is capped so that
            start_z + target_z <= vol_z - z_artifact_margin.
    
    Returns:
        subvolume: Random subvolume of target_shape
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    vol_shape = volume.shape
    
    # Calculate valid starting positions; cap z upper bound for artifact margin
    max_x = vol_shape[0] - target_shape[0]
    max_y = vol_shape[1] - target_shape[1]
    max_z = (vol_shape[2] - z_artifact_margin) - target_shape[2]
    
    if max_x < 0 or max_y < 0 or max_z < 0:
        raise ValueError(f"Target shape {target_shape} larger than usable volume {vol_shape} "
                         f"(z_artifact_margin={z_artifact_margin})")
    
    start_x = np.random.randint(0, max_x + 1)
    start_y = np.random.randint(0, max_y + 1)
    start_z = np.random.randint(0, max_z + 1)
    
    subvolume = volume[
        start_x:start_x + target_shape[0],
        start_y:start_y + target_shape[1],
        start_z:start_z + target_shape[2]
    ]
    
    return subvolume
