"""
Seismic 3D Masking Strategies
=============================
Implements masking for seismic pre-training:
- Peak/trough preservation along vertical axis
- Random trace masking with 3x3 cluster patterns
- Zero-masking for masked voxels
"""

import numpy as np
from typing import Tuple, Optional


def create_mask_3d(
    seismic_data: np.ndarray,
    trace_mask_ratio: float = 0.07,
    cluster_prob: float = 0.8,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Create a 3D mask for seismic data.
    
    Args:
        seismic_data: 3D seismic array (z, x, y) - depth, width, height
        trace_mask_ratio: Fraction of traces to potentially mask (0.07 = 7%)
        cluster_prob: Probability a trace in a 3x3 cluster is masked (0.8 = 80%)
        random_seed: Optional random seed for reproducibility
    
    Returns:
        mask: Boolean array where True = preserve, False = mask (zero)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
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
    
    # Step 2: Random trace masking with 3x3 clusters
    n_traces_to_mask = int(x * y * trace_mask_ratio)
    
    # Select random trace positions (x, y pairs) - x is width (axis 1), y is height (axis 2)
    trace_indices = np.random.choice(x * y, size=n_traces_to_mask, replace=False)
    
    for idx in trace_indices:
        trace_x = idx % x  # x coordinate (width)
        trace_y = idx // x  # y coordinate (height)
        
        # Create 3x3 cluster around this trace
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                cx = trace_x + dx
                cy = trace_y + dy
                
                # Check bounds
                if 0 <= cx < x and 0 <= cy < y:
                    # Randomly mask this trace with probability cluster_prob
                    if np.random.random() < cluster_prob:
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
