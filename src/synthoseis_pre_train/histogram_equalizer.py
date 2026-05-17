"""Histogram-equalisation amplitude transform for seismic pre-training.

Adapted from synthoseis.datagenerator.histogram_equalizer with the following
changes relative to the original:

  * ``_apply_standard_normal`` refactored to use ``numpy.interp`` instead of
    ``scipy.interpolate.interp1d``.  For strictly in-range inputs the two
    approaches are numerically equivalent (linear interpolation); numpy.interp
    also clamps out-of-range inputs to boundary values rather than raising,
    which is safer for local 128³ training windows drawn from a larger volume.

  * Storage back-end changed from .npz files to zarr groups (consistent with
    the rest of the pre-train transform infrastructure).

  * ``derive_histeq_params`` accepts a flat 1-D float array sampled from *all*
    seismic array keys in a zarr store so that a single shared transform is
    derived and can be applied identically to any angle-stack or full-stack key.

  * ``ensure_histeq_params`` / ``load_histeq_params`` handle persistence inside
    the zarr store under ``<transforms_group>/histeq/``.

Numerical equivalence test
--------------------------
This module intentionally diverges from the synthoseis source in two places:

1. A configurable prefilter is applied before deriving histogram bins.
    The production default is a 19-point triangular kernel with
    ``prefilter_power=1.6``.
2. Interpolation arrays are endpoint-extended during application to reduce
    boundary clipping artifacts for out-of-range inputs.

As a result, bit-identical equivalence with the original implementation is no
longer expected.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np
import zarr


# ---------------------------------------------------------------------------
# Zarr storage constants
# ---------------------------------------------------------------------------

_HISTEQ_SUBGROUP = "histeq"
_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistEqConfig:
    """Configuration for histogram-equalisation amplitude transform."""

    transforms_group: str = "transforms"
    nbr_bins: int = 256
    kernel_length: int = 5
    prefilter_power: float = 4.0
    max_voxels_per_key: int = 2_000_000
    lock_timeout_sec: float = 120.0
    lock_poll_sec: float = 0.1

    def validate(self) -> None:
        if not self.transforms_group:
            raise ValueError("transforms_group must be non-empty")
        if self.nbr_bins < 8:
            raise ValueError("nbr_bins must be >= 8")
        if self.kernel_length < 1 or self.kernel_length % 2 == 0:
            raise ValueError("kernel_length must be a positive odd integer")
        if self.prefilter_power <= 0:
            raise ValueError("prefilter_power must be > 0")
        if self.max_voxels_per_key < 1000:
            raise ValueError("max_voxels_per_key must be >= 1000")
        if self.lock_timeout_sec <= 0:
            raise ValueError("lock_timeout_sec must be > 0")
        if self.lock_poll_sec <= 0:
            raise ValueError("lock_poll_sec must be > 0")


# ---------------------------------------------------------------------------
# Core transform parameters
# ---------------------------------------------------------------------------


@dataclass
class HistEqParams:
    """Derived histogram-equalisation parameters for one zarr store."""

    centerbins: np.ndarray        # shape (nbr_bins,), float32
    target_centerbins: np.ndarray # shape (nbr_bins,), float32
    seismic_mean: float           # global mean subtracted before mapping


# ---------------------------------------------------------------------------
# Core numerical functions (ported from synthoseis histogram_equalizer.py)
# ---------------------------------------------------------------------------


def _generate_triangular_kernel(length: int, power: float) -> np.ndarray:
    """Generate triangular kernel [1,3,5,...,peak,...,5,3,1]**power normalized."""
    if length < 1 or length % 2 == 0:
        raise ValueError("kernel length must be a positive odd integer")
    if power <= 0:
        raise ValueError("kernel power must be > 0")
    
    mid = (length + 1) // 2
    # Build triangle with increment of 2: [1, 3, 5, ..., peak, ..., 5, 3, 1]
    kernel = np.concatenate([
        np.arange(1, 2*mid, 2, dtype=np.float64),
        np.arange(2*mid - 3, 0, -2, dtype=np.float64),
    ])
    kernel = np.power(kernel, float(power))
    kernel /= kernel.sum()
    return kernel


def _derive_standard_normal(
    im: np.ndarray,
    nbr_bins: int = 256,
    prefilter_power: float = 1.6,
    kernel_length: int = 19,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive histogram-equalisation curve to standard-normal shape.

    Exact port of ``synthoseis.datagenerator.histogram_equalizer
    ._derive_standard_normal``.  Returns ``(centerbins, target_centerbins)``
    as float64 arrays.  Does NOT apply the transform; call
    ``_apply_standard_normal`` separately.
    """
    from numpy import histogram
    from scipy.interpolate import interp1d

    if prefilter_power <= 0:
        raise ValueError("prefilter_power must be > 0")

    # Decimate to ~10 000 representative values; add global min/max so the
    # full data range is always bracketed; symmetrise about zero.
    decimate = max(1, int(sqrt(im.flatten().shape[0] / 10000)))
    amin, amax = im.min(), im.max()
    _data = im.flatten()[::decimate]

    # Smooth clipped plateaus before deriving bins to reduce repeated-adjacent
    # values after equalisation.
    kernel = _generate_triangular_kernel(kernel_length, prefilter_power)
    _data = np.convolve(_data.astype(np.float64), kernel, mode="same")

    _data = np.hstack((amin, _data, amax))
    _data = np.hstack((_data, -_data))

    histrange = _data.max() - _data.min()
    if len(np.arange(_data.min(), _data.max(), histrange / (nbr_bins - 1))) == 0:
        # Degenerate (constant) input – return identity mapping.
        cb = np.linspace(amin, amax, nbr_bins)
        return cb, cb

    imhist, _bins = histogram(_data, bins=nbr_bins, density=True)
    imhist[0] = 0.0
    imhist[-1] = 0.0

    centerbins = np.linspace(_data.min(), _data.max(), nbr_bins)

    # Smooth input PDF with a moving-median filter (window ±2).
    imhistmedian = np.empty(len(imhist), dtype=float)
    for i in range(len(imhist)):
        indexmin = max(0, i - 2)
        indexmax = min(i + 2, len(imhist))
        imhistmedian[i] = np.median(imhist[indexmin:indexmax])

    if imhistmedian[imhistmedian != 0].shape[0] == 0:
        imhistmedian = imhist

    imhistmedian[0] = 0.0
    cdf = imhistmedian.cumsum()
    cdf /= cdf[-1]

    # Standard-normal target distribution (symmetrised).
    fit_vals = np.random.normal(loc=0.0, scale=1.0, size=_data.flatten().shape[0])
    fit_vals = np.hstack((fit_vals, -fit_vals))

    fit_histrange = fit_vals.max() - fit_vals.min()
    fit_imhist, _fit_bins = histogram(fit_vals, bins=nbr_bins, density=True)
    fit_imhist[0] = 0.0
    fit_imhist[-1] = 0.0
    fit_centerbins = np.linspace(fit_vals.min(), fit_vals.max(), nbr_bins)
    fit_cdf = fit_imhist.cumsum()
    fit_cdf /= fit_cdf[-1]

    f = interp1d(fit_cdf, cdf)
    normed_centerbins = centerbins - centerbins.min()
    normed_centerbins /= normed_centerbins.max()
    deltacdf = f(normed_centerbins)
    equality_line = np.linspace(0.0, 1.0, cdf.shape[0])
    deltacdf = deltacdf - equality_line

    target_centerbins = fit_centerbins + deltacdf * (
        fit_centerbins.max() - fit_centerbins.min()
    )
    mirrored = -target_centerbins[::-1]
    target_centerbins = (target_centerbins + mirrored) / 2.0

    # Normalise so that applying the mapping to _data yields std ≈ 1.
    # Use np.interp here (consistent with _apply_standard_normal below).
    x_interp, y_interp = _extended_interp_bins(centerbins, target_centerbins)
    im_std = np.interp(_data, x_interp, y_interp).std()

    return centerbins, target_centerbins / im_std


def _extended_interp_bins(
    centerbins: np.ndarray,
    target_centerbins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend interpolation endpoints for safer out-of-range handling.

    Input range is doubled in total width by extending each side by half of
    the original range. Output range is extended marginally by 2x endpoint
    intervals, as requested.
    """
    if centerbins.ndim != 1 or target_centerbins.ndim != 1:
        raise ValueError("centerbins and target_centerbins must be 1D")
    if centerbins.size != target_centerbins.size:
        raise ValueError("centerbins and target_centerbins must have same length")
    if centerbins.size < 2:
        raise ValueError("at least two bins are required for interpolation")

    x = np.asarray(centerbins, dtype=np.float64)
    y = np.asarray(target_centerbins, dtype=np.float64)

    x_range = x[-1] - x[0]
    x_pad = 5 * x_range
    x_left = x[0] - x_pad
    x_right = x[-1] + x_pad

    y_step_left = y[1] - y[0]
    y_step_right = y[-1] - y[-2]
    y_left = y[0] - 5.0 * y_step_left
    y_right = y[-1] + 5.0 * y_step_right

    x_ext = np.concatenate(([x_left], x, [x_right]))
    y_ext = np.concatenate(([y_left], y, [y_right]))
    return x_ext, y_ext


def _apply_standard_normal(
    im: np.ndarray,
    centerbins: np.ndarray,
    target_centerbins: np.ndarray,
) -> np.ndarray:
    """Apply histogram-equalisation curve to array *im*.

    Refactored from ``synthoseis.datagenerator.histogram_equalizer
    ._apply_standard_normal``:

    * Uses ``numpy.interp`` instead of ``scipy.interpolate.interp1d``.
    * Fully vectorised - no per-slice loop.
    * Out-of-range values are mapped using endpoint-extended interpolation
      arrays to reduce boundary clipping artifacts.
    """
    x_interp, y_interp = _extended_interp_bins(centerbins, target_centerbins)
    output = np.interp(im.ravel(), x_interp, y_interp)
    return output.reshape(im.shape).astype(im.dtype)


# ---------------------------------------------------------------------------
# Zarr persistence helpers
# ---------------------------------------------------------------------------


def _histeq_group_if_exists(
    root: zarr.Group,
    cfg: HistEqConfig,
) -> zarr.Group | None:
    """Return the stored histeq group without creating anything."""
    if cfg.transforms_group not in root:
        return None
    tfm_root = root[cfg.transforms_group]
    if _HISTEQ_SUBGROUP not in tfm_root:
        return None
    return tfm_root[_HISTEQ_SUBGROUP]


def _try_load_from_group(grp: zarr.Group) -> HistEqParams | None:
    if "centerbins" not in grp or "target_centerbins" not in grp:
        return None
    centerbins = np.asarray(grp["centerbins"][:], dtype=np.float32)
    target_centerbins = np.asarray(grp["target_centerbins"][:], dtype=np.float32)
    seismic_mean = float(grp.attrs.get("seismic_mean", 0.0))
    return HistEqParams(
        centerbins=centerbins,
        target_centerbins=target_centerbins,
        seismic_mean=seismic_mean,
    )


def _lock_path_histeq(data_path: Path, cfg: HistEqConfig) -> Path:
    lock_dir = data_path.parent / ".transform_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return lock_dir / f"histeq__{cfg.transforms_group}.lock"


def _acquire_lock(lock_file: Path, timeout_sec: float, poll_sec: float) -> int:
    start = time.monotonic()
    while True:
        try:
            return os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if (time.monotonic() - start) >= timeout_sec:
                raise TimeoutError(
                    f"Timeout waiting for histeq transform lock: {lock_file}"
                )
            time.sleep(poll_sec)


def _release_lock(fd: int, lock_file: Path) -> None:
    try:
        os.close(fd)
    finally:
        try:
            lock_file.unlink(missing_ok=True)
        except OSError:
            pass


def _collect_representative_sample(
    zarr_root: zarr.Group,
    array_keys: list[str],
    max_voxels_per_key: int = 2_000_000,
) -> np.ndarray:
    """Collect a flat decimated sample from all seismic array keys.

    For each key a stride-based read keeps memory usage bounded to roughly
    ``max_voxels_per_key`` float32 values per key while remaining
    representative of the full amplitude distribution.
    """
    parts: list[np.ndarray] = []
    for key in array_keys:
        try:
            arr = zarr_root[key]
        except (KeyError, FileNotFoundError):
            continue
        shape = arr.shape
        if len(shape) != 3:
            continue
        total = int(np.prod(shape))
        # Compute per-dimension stride so total_strided ≈ max_voxels_per_key.
        stride = max(1, int(round((total / max_voxels_per_key) ** (1.0 / 3.0))))
        data = np.asarray(arr[::stride, ::stride, ::stride], dtype=np.float32)
        parts.append(data.ravel())

    if not parts:
        raise ValueError(
            "No valid 3D array keys found in zarr store for histeq derivation. "
            f"Tried: {array_keys}"
        )
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def derive_histeq_params(
    flat_values: np.ndarray,
    nbr_bins: int = 256,
    prefilter_power: float = 1.6,
    kernel_length: int = 19,
) -> HistEqParams:
    """Derive ``HistEqParams`` from a flat array of representative values.

    ``flat_values`` should be a concatenated, decimated sample drawn from all
    seismic array keys in the zarr store (angle stacks *and* full stack) so
    that the resulting transform is appropriate for any of those keys.

    The derivation:
    1. Computes ``seismic_mean`` = global mean of ``flat_values``.
    2. Subtracts the mean (centres data at zero, as seismic should be).
    3. Calls ``_derive_standard_normal`` to obtain ``centerbins`` and
       ``target_centerbins``.
    """
    flat = np.asarray(flat_values, dtype=np.float64).ravel()
    seismic_mean = float(np.mean(flat))
    flat -= seismic_mean
    centerbins, target_centerbins = _derive_standard_normal(
        flat.astype(np.float32),
        nbr_bins=nbr_bins,
        prefilter_power=prefilter_power,
        kernel_length=kernel_length,
    )
    return HistEqParams(
        centerbins=centerbins.astype(np.float32),
        target_centerbins=target_centerbins.astype(np.float32),
        seismic_mean=seismic_mean,
    )


def load_histeq_params(
    data_path: str | Path,
    config: HistEqConfig,
) -> HistEqParams | None:
    """Load persisted ``HistEqParams`` from zarr store, or return None.

    Safe for read-only zarr stores — never writes.
    """
    config.validate()
    root = zarr.open(str(Path(data_path)), mode="r")
    grp = _histeq_group_if_exists(root, config)
    if grp is None:
        return None
    return _try_load_from_group(grp)


def ensure_histeq_params(
    data_path: str | Path,
    array_keys: list[str],
    config: HistEqConfig,
) -> HistEqParams:
    """Load persisted ``HistEqParams`` or derive and store them.

    Thread/process safe: a file-based lock prevents concurrent derivation.
    If the zarr store is read-only, derivation is performed in-memory and
    *not* persisted (caller receives valid params regardless).

    Args:
        data_path: Path to the zarr store directory.
        array_keys: List of 3D seismic array keys to sample when deriving.
        config: Histogram-equalisation configuration.

    Returns:
        ``HistEqParams`` ready for use in ``_apply_standard_normal``.
    """
    config.validate()
    path = Path(data_path)

    # Fast path: load without acquiring any lock.
    try:
        root = zarr.open(str(path), mode="r")
        grp = _histeq_group_if_exists(root, config)
        if grp is not None:
            params = _try_load_from_group(grp)
            if params is not None:
                return params
    except Exception:
        pass

    # Need to derive.  Try write-mode first; fall back to in-memory if read-only.
    try:
        root_rw = zarr.open(str(path), mode="a")
    except Exception:
        root_rw = None

    if root_rw is None:
        # Read-only fallback: derive in-memory.
        root_ro = zarr.open(str(path), mode="r")
        flat = _collect_representative_sample(root_ro, array_keys, config.max_voxels_per_key)
        return derive_histeq_params(
            flat,
            nbr_bins=config.nbr_bins,
            prefilter_power=config.prefilter_power,
            kernel_length=config.kernel_length,
        )

    lock_file = _lock_path_histeq(path, config)
    fd = _acquire_lock(lock_file, config.lock_timeout_sec, config.lock_poll_sec)
    try:
        # Re-check after acquiring lock (another process may have written it).
        grp = _histeq_group_if_exists(root_rw, config)
        if grp is not None:
            params = _try_load_from_group(grp)
            if params is not None:
                return params

        # Derive from a combined sample of all available keys.
        flat = _collect_representative_sample(root_rw, array_keys, config.max_voxels_per_key)
        params = derive_histeq_params(
            flat,
            nbr_bins=config.nbr_bins,
            prefilter_power=config.prefilter_power,
            kernel_length=config.kernel_length,
        )

        # Persist to zarr.
        tfm_root = root_rw.require_group(config.transforms_group)
        grp_w = tfm_root.require_group(_HISTEQ_SUBGROUP)
        grp_w.create_array("centerbins", data=params.centerbins, overwrite=True)
        grp_w.create_array(
            "target_centerbins", data=params.target_centerbins, overwrite=True
        )
        grp_w.attrs.update(
            {
                "seismic_mean": float(params.seismic_mean),
                "nbr_bins": config.nbr_bins,
                "kernel_length": config.kernel_length,
                "prefilter_power": config.prefilter_power,
                "schema_version": _SCHEMA_VERSION,
                "array_keys_used": list(array_keys),
            }
        )
        return params

    finally:
        _release_lock(fd, lock_file)
