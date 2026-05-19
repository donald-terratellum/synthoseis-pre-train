"""Quantile-based amplitude transforms persisted per zarr array key.

This module provides a reusable API to derive/load/apply a quantile-to-normal
mapping per array key in a zarr store.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
import os
import re
import time
from typing import Any

import numpy as np
from scipy.stats import norm
import zarr


_SCHEMA_VERSION = 1
_TRANSFORM_TYPE = "quantile_to_normal"


@dataclass(frozen=True)
class QuantileNormalConfig:
    """Configuration for quantile-to-normal amplitude transform."""

    epsilon: float = 1e-6
    symmetry_mode: str = "strict_odd"  # choices: strict_odd, independent
    transforms_group: str = "transforms"
    lock_timeout_sec: float = 120.0
    lock_poll_sec: float = 0.1

    def validate(self) -> None:
        if not (0.0 < float(self.epsilon) < 0.5):
            raise ValueError("epsilon must be in (0, 0.5)")
        if self.symmetry_mode not in ("strict_odd", "independent"):
            raise ValueError("symmetry_mode must be 'strict_odd' or 'independent'")
        if not self.transforms_group:
            raise ValueError("transforms_group must be a non-empty string")
        if self.lock_timeout_sec <= 0:
            raise ValueError("lock_timeout_sec must be > 0")
        if self.lock_poll_sec <= 0:
            raise ValueError("lock_poll_sec must be > 0")


class QuantileNormalTransform:
    """In-memory transform with forward/inverse lookup mapping."""

    def __init__(
        self,
        x_lut: np.ndarray,
        z_lut: np.ndarray,
        *,
        symmetry_mode: str,
        metadata: dict[str, Any],
    ):
        self.x_lut = np.asarray(x_lut, dtype=np.float32)
        self.z_lut = np.asarray(z_lut, dtype=np.float32)
        self.symmetry_mode = symmetry_mode
        self.metadata = metadata

        if self.x_lut.ndim != 1 or self.z_lut.ndim != 1:
            raise ValueError("x_lut and z_lut must be 1D arrays")
        if self.x_lut.size != self.z_lut.size:
            raise ValueError("x_lut and z_lut must have the same length")
        if self.x_lut.size < 2:
            raise ValueError("x_lut/z_lut must contain at least 2 entries")

    def forward(self, values: np.ndarray) -> np.ndarray:
        """Map amplitudes to normal space (float32)."""
        x = np.asarray(values, dtype=np.float32)
        if self.symmetry_mode == "strict_odd":
            signs = np.sign(x)
            mag = np.abs(x)
            mag_z = np.interp(mag, self.x_lut, self.z_lut).astype(np.float32)
            return (signs * mag_z).astype(np.float32)

        out = np.interp(x, self.x_lut, self.z_lut).astype(np.float32)
        return out

    def inverse(self, values: np.ndarray) -> np.ndarray:
        """Map normal-space values back to amplitude space (float32)."""
        z = np.asarray(values, dtype=np.float32)
        if self.symmetry_mode == "strict_odd":
            signs = np.sign(z)
            mag = np.abs(z)
            mag_x = np.interp(mag, self.z_lut, self.x_lut).astype(np.float32)
            return (signs * mag_x).astype(np.float32)

        out = np.interp(z, self.z_lut, self.x_lut).astype(np.float32)
        return out


def _safe_array_key_name(array_key: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "__", array_key)
    digest = sha1(array_key.encode("utf-8")).hexdigest()[:10]
    return f"{safe}__{digest}"


def _config_tag(cfg: QuantileNormalConfig) -> str:
    eps_tag = f"{float(cfg.epsilon):.1e}".replace("+", "")
    return f"{_TRANSFORM_TYPE}__{cfg.symmetry_mode}__eps_{eps_tag}"


def _lock_path(data_path: Path, array_key: str, cfg: QuantileNormalConfig) -> Path:
    lock_dir = data_path.parent / ".transform_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_name = f"{_safe_array_key_name(array_key)}__{_config_tag(cfg)}.lock"
    return lock_dir / lock_name


def _acquire_lock(lock_file: Path, timeout_sec: float, poll_sec: float) -> int:
    start = time.monotonic()
    while True:
        try:
            return os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if (time.monotonic() - start) >= timeout_sec:
                raise TimeoutError(f"Timeout waiting for transform lock: {lock_file}")
            time.sleep(poll_sec)


def _release_lock(fd: int, lock_file: Path) -> None:
    try:
        os.close(fd)
    finally:
        try:
            lock_file.unlink(missing_ok=True)
        except OSError:
            pass


def _build_transform(values: np.ndarray, cfg: QuantileNormalConfig) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    vals = np.asarray(values, dtype=np.float32).ravel()
    if vals.size == 0:
        raise ValueError("Cannot derive transform from an empty array")

    drift_mean = float(np.mean(vals, dtype=np.float64))
    drift_std = float(np.std(vals, dtype=np.float64))

    if cfg.symmetry_mode == "strict_odd":
        x_lut = np.sort(np.abs(vals).astype(np.float64, copy=False))
        q = np.linspace(0.0, 1.0, x_lut.size, dtype=np.float64)
        p = np.clip(0.5 + 0.5 * q, cfg.epsilon, 1.0 - cfg.epsilon)
        z_lut = norm.ppf(p)
    else:
        x_lut = np.sort(vals.astype(np.float64, copy=False))
        q = np.linspace(0.0, 1.0, x_lut.size, dtype=np.float64)
        p = np.clip(q, cfg.epsilon, 1.0 - cfg.epsilon)
        z_lut = norm.ppf(p)

    x_lut = x_lut.astype(np.float32)
    z_lut = z_lut.astype(np.float32)

    # Ensure strict monotonic interpolation domains.
    x_lut = np.maximum.accumulate(x_lut)
    z_lut = np.maximum.accumulate(z_lut)

    metadata = {
        "schema_version": _SCHEMA_VERSION,
        "transform_type": _TRANSFORM_TYPE,
        "symmetry_mode": cfg.symmetry_mode,
        "epsilon": float(cfg.epsilon),
        "dtype": "float32",
        "voxel_count": int(vals.size),
        "source_mean": drift_mean,
        "source_std": drift_std,
        "source_abs_mean_drift": abs(drift_mean),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return x_lut, z_lut, metadata


def _transform_group(root: zarr.Group, array_key: str, cfg: QuantileNormalConfig) -> zarr.Group:
    transforms_root = root.require_group(cfg.transforms_group)
    key_group = transforms_root.require_group(_safe_array_key_name(array_key))
    return key_group.require_group(_config_tag(cfg))


def _transform_group_if_exists(root: zarr.Group, array_key: str, cfg: QuantileNormalConfig) -> zarr.Group | None:
    """Return existing transform group without creating anything."""
    if cfg.transforms_group not in root:
        return None
    transforms_root = root[cfg.transforms_group]

    key_name = _safe_array_key_name(array_key)
    if key_name not in transforms_root:
        return None
    key_group = transforms_root[key_name]

    cfg_tag = _config_tag(cfg)
    if cfg_tag not in key_group:
        return None

    return key_group[cfg_tag]


def _try_load_existing(root: zarr.Group, array_key: str, cfg: QuantileNormalConfig) -> QuantileNormalTransform | None:
    grp = _transform_group_if_exists(root, array_key, cfg)
    if grp is None:
        return None
    if "x_lut" not in grp or "z_lut" not in grp:
        return None

    x_lut = np.asarray(grp["x_lut"][:], dtype=np.float32)
    z_lut = np.asarray(grp["z_lut"][:], dtype=np.float32)
    metadata = dict(grp.attrs.asdict())
    metadata["array_key"] = array_key
    return QuantileNormalTransform(
        x_lut=x_lut,
        z_lut=z_lut,
        symmetry_mode=str(metadata.get("symmetry_mode", cfg.symmetry_mode)),
        metadata=metadata,
    )


def derive_quantile_normal_transform(
    *,
    array_key: str,
    array_values: np.ndarray,
    config: QuantileNormalConfig,
) -> QuantileNormalTransform:
    """Derive a quantile transform in-memory without persisting to zarr."""
    config.validate()
    x_lut, z_lut, metadata = _build_transform(array_values, config)
    metadata["array_key"] = array_key
    return QuantileNormalTransform(
        x_lut=x_lut,
        z_lut=z_lut,
        symmetry_mode=config.symmetry_mode,
        metadata=metadata,
    )


def ensure_quantile_normal_transform(
    *,
    data_path: str | Path,
    array_key: str,
    array_values: np.ndarray,
    config: QuantileNormalConfig,
) -> QuantileNormalTransform:
    """Load existing transform or lazily derive/store it for a zarr array key."""
    config.validate()
    path = Path(data_path)

    root = zarr.open(str(path), mode="a")
    existing = _try_load_existing(root, array_key, config)
    if existing is not None:
        return existing

    lock_file = _lock_path(path, array_key, config)
    fd = _acquire_lock(lock_file, config.lock_timeout_sec, config.lock_poll_sec)
    try:
        # Re-check after obtaining lock (another process might have written it).
        root = zarr.open(str(path), mode="a")
        existing = _try_load_existing(root, array_key, config)
        if existing is not None:
            return existing

        x_lut, z_lut, metadata = _build_transform(array_values, config)
        metadata["array_key"] = array_key

        grp = _transform_group(root, array_key, config)
        grp.attrs.update(metadata)
        grp.create_array("x_lut", data=x_lut, overwrite=True)
        grp.create_array("z_lut", data=z_lut, overwrite=True)

        return QuantileNormalTransform(
            x_lut=x_lut,
            z_lut=z_lut,
            symmetry_mode=config.symmetry_mode,
            metadata=metadata,
        )
    finally:
        _release_lock(fd, lock_file)


def load_quantile_normal_transform(
    *,
    data_path: str | Path,
    array_key: str,
    config: QuantileNormalConfig,
) -> QuantileNormalTransform | None:
    """Load an already-derived transform for a zarr array key, if present."""
    config.validate()
    root = zarr.open(str(Path(data_path)), mode="r")
    return _try_load_existing(root, array_key, config)
