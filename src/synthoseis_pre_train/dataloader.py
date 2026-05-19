"""
Seismic Data Loader
====================
Loads and processes seismic data from Zarr format for training.
"""

import numpy as np
import zarr
from typing import Tuple, Optional, List, Callable
from pathlib import Path
import random
import re

from synthoseis_pre_train.transforms import (
    QuantileNormalConfig,
    QuantileNormalTransform,
    derive_quantile_normal_transform,
    ensure_quantile_normal_transform,
    load_quantile_normal_transform,
)
from synthoseis_pre_train.histogram_equalizer import (
    HistEqConfig,
    HistEqParams,
    _apply_standard_normal as _histeq_apply,
    ensure_histeq_params,
    load_histeq_params,
)

# Zarr z-axis (axis 2) has known artifacts in the last N indices.
# Restrict random subvolume starts so the deepest sampled z-index is < z_size - Z_ARTIFACT_MARGIN.
Z_ARTIFACT_MARGIN = 2


def _first_last_positive_indices(arr_1d: np.ndarray) -> tuple[int, int] | None:
    """Return first/last indices where 1D array is > 0; None if none found."""
    idx = np.where(arr_1d > 0)[0]
    if idx.size == 0:
        return None
    return int(idx[0]), int(idx[-1])


def _derive_bounds_from_abs_energy(input_data: np.ndarray) -> tuple[int, int, int, int, int, int]:
    """Derive z/x/y bounds from absolute-energy support of a (z, x, y) array."""
    abs_input = np.abs(input_data)

    z_sum = np.sum(np.sum(abs_input, axis=1), axis=1)  # [z]
    x_sum = np.sum(np.sum(abs_input, axis=0), axis=1)  # [x]
    y_sum = np.sum(np.sum(abs_input, axis=0), axis=0)  # [y]

    z_bounds = _first_last_positive_indices(z_sum)
    x_bounds = _first_last_positive_indices(x_sum)
    y_bounds = _first_last_positive_indices(y_sum)

    if z_bounds is None or x_bounds is None or y_bounds is None:
        # Fallback to full sample if a degenerate all-zero sample appears.
        z, x, y = input_data.shape
        return 0, z - 1, 0, x - 1, 0, y - 1

    z_min, z_max = z_bounds
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    return z_min, z_max, x_min, x_max, y_min, y_max


def _dilate_binary_2d(mask_2d: np.ndarray, radius: int = 1) -> np.ndarray:
    """Binary dilation in XY using an all-ones neighborhood of size (2r+1)^2."""
    if radius <= 0:
        return mask_2d.astype(bool)

    padded = np.pad(mask_2d.astype(bool), pad_width=radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask_2d, dtype=bool)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            xs = radius + dx
            ys = radius + dy
            out |= padded[xs:xs + mask_2d.shape[0], ys:ys + mask_2d.shape[1]]
    return out


def _compact_histeq_key_name(key: str) -> str:
    """Shorten verbose seismic key names for stdout readability."""
    shortened = str(key)
    for prefix in ("seismicCubes_cumsum", "seismicCubes"):
        if shortened.startswith(prefix):
            shortened = "*" + shortened[len(prefix):]
            break
    return shortened


class SeismicDataset:
    """
    Dataset for seismic pre-training with masking and augmentation.
    """
    
    def __init__(
        self,
        data_path: str,
        sample_shape: Tuple[int, int, int] = (128, 128, 128),
        trace_mask_ratio: float = 0.07,
        target_masked_fraction: Optional[float] = None,
        cluster_shape: int = 3,
        center_selection_method: str = "random_mixture",
        mask_fill_method: str = "zero",
        mask_noise_std: float = 1e-2,
        augment: bool = True,
        normalize: bool = True,
        target_std: float = 1.0,
        cache_in_memory: bool = False,
        array_key: Optional[str] = None,
        array_keys: Optional[List[str]] = None,
        amplitude_transform: str = "example_stdev_scaling",
        quantile_symmetry_mode: str = "strict_odd",
        quantile_epsilon: float = 1e-6,
        transforms_group: str = "transforms",
        enable_cluster_mask_expansion: bool = True,
        histogram_nbr_bins: int = 256,
    ):
        """
        Args:
            data_path: Path to Zarr seismic data
            sample_shape: Shape of each training sample (x, y, z)
            trace_mask_ratio: Ratio of traces to mask
            target_masked_fraction: Preferred masked-trace fraction override
            cluster_shape: Odd cluster edge size for trace masking
            center_selection_method: Cluster-center sampling strategy
            mask_fill_method: Masked-voxel infill method (zero or gaussian)
            mask_noise_std: Standard deviation for gaussian mask infill
            augment: Whether to apply data augmentation
            normalize: Whether to normalize samples
            target_std: Target standard deviation after normalization
            cache_in_memory: Whether to cache all data in memory
            array_key: Single specific 3D array key to use (legacy; takes precedence)
            array_keys: List of 3D array keys to randomly sample from each __getitem__
            amplitude_transform: Preprocessing transform mode. Preferred:
                "example_stdev_scaling" (per-sample std scaling),
                "dataset_quantile_scaling" (per-zarr-key quantile->normal), or
                "histogram_equalization" (joint histeq across all keys, ported from synthoseis).
                Backward-compatible aliases are accepted:
                "standardize" -> "example_stdev_scaling",
                "quantile_normal" -> "dataset_quantile_scaling".
            quantile_symmetry_mode: Quantile mode: "strict_odd" (default) or "independent"
            quantile_epsilon: Epsilon used when mapping quantiles to normal-space
            transforms_group: Zarr subgroup name used to store persisted transforms
            enable_cluster_mask_expansion: Whether to run bounds+dilation mask expansion
                used by cluster-focused loss weighting.
            histogram_nbr_bins: Number of histogram bins for histeq derivation (default 256).
        """
        self.data_path = Path(data_path)
        self.sample_shape = sample_shape
        self.trace_mask_ratio = trace_mask_ratio
        self.target_masked_fraction = target_masked_fraction
        self.cluster_shape = int(cluster_shape)
        self.center_selection_method = center_selection_method
        self.mask_fill_method = str(mask_fill_method)
        self.mask_noise_std = float(mask_noise_std)
        self.augment = augment
        self.normalize = normalize
        self.target_std = target_std
        if self.cluster_shape < 1 or self.cluster_shape % 2 == 0:
            raise ValueError("cluster_shape must be a positive odd integer")
        if self.mask_fill_method not in ("zero", "gaussian"):
            raise ValueError("mask_fill_method must be one of: zero, gaussian")
        if self.mask_noise_std < 0:
            raise ValueError("mask_noise_std must be >= 0")
        _legacy_alias = {
            "standardize": "example_stdev_scaling",
            "quantile_normal": "dataset_quantile_scaling",
        }
        amplitude_transform = _legacy_alias.get(str(amplitude_transform), str(amplitude_transform))
        _valid_transforms = ("example_stdev_scaling", "dataset_quantile_scaling", "histogram_equalization")
        if amplitude_transform not in _valid_transforms:
            raise ValueError(
                f"amplitude_transform must be one of: {', '.join(_valid_transforms)}. "
                "Aliases: standardize->example_stdev_scaling, "
                "quantile_normal->dataset_quantile_scaling"
            )
        self.amplitude_transform = str(amplitude_transform)
        # Exactly one normalization path is active when normalize=True.
        self.apply_standardize = self.normalize and (self.amplitude_transform == "example_stdev_scaling")
        self.apply_quantile = self.normalize and (self.amplitude_transform == "dataset_quantile_scaling")
        self.apply_histeq = self.normalize and (self.amplitude_transform == "histogram_equalization")
        self.quantile_config = QuantileNormalConfig(
            epsilon=quantile_epsilon,
            symmetry_mode=quantile_symmetry_mode,
            transforms_group=transforms_group,
        )
        self.histeq_config = HistEqConfig(
            transforms_group=transforms_group,
            nbr_bins=int(histogram_nbr_bins),
        )
        self.enable_cluster_mask_expansion = bool(enable_cluster_mask_expansion)
        self._transform_cache: dict[str, QuantileNormalTransform] = {}
        self._reported_derived_transform_keys: set[str] = set()
        self._reported_forward_transform_keys: set[str] = set()
        self._reported_readonly_fallback_keys: set[str] = set()
        self._histeq_params: HistEqParams | None = None
        self._histeq_reported: bool = False
        
        # Load zarr data
        self.zarr_data = zarr.open(str(data_path), mode='r')

        all_3d_keys = [
            key for key in self.zarr_data.array_keys()
            if len(self.zarr_data[key].shape) == 3
        ]

        # Resolve which keys to use: single key > explicit list > all 3D keys
        if array_key is not None:
            candidate_keys = [array_key]
        elif array_keys is not None:
            candidate_keys = list(array_keys)
        else:
            candidate_keys = all_3d_keys

        # Keep only keys that actually exist and are 3D in this zarr
        self.available_cubes = [k for k in candidate_keys if k in all_3d_keys]
        if not self.available_cubes:
            raise ValueError(
                f"None of the requested array keys found as 3D arrays in {data_path}.\n"
                f"  Requested: {candidate_keys}\n"
                f"  Available: {all_3d_keys}"
            )
        
        # Cache if requested
        self.cached_data = None
        if cache_in_memory:
            self._cache_data()
    
    def _cache_data(self):
        """Cache all data in memory."""
        print("Caching seismic data in memory...")
        self.cached_data = []
        for cube_name in self.available_cubes:
            cube = self.zarr_data[cube_name][:]
            self.cached_data.append((cube_name, cube))
        print(f"Cached {len(self.cached_data)} cubes")

    def _get_dataset_prefix(self) -> str:
        """Extract dataset prefix for logging (e.g., run_0194)."""
        # Search path components from nearest to farthest for explicit run_####.
        # This handles layouts like .../seismic__...__run_0194/model_data/data.zarr.
        candidates = [self.data_path.name, self.data_path.stem]
        candidates.extend(parent.name for parent in self.data_path.parents)

        for token in candidates:
            token = str(token).replace(".zarr", "")
            match = re.search(r"run_\d+", token)
            if match:
                return match.group(0)

        # Last fallback: nearest parent folder name.
        parent_name = self.data_path.parent.name.replace(".zarr", "")
        return parent_name or self.data_path.stem.replace(".zarr", "")

    def _display_key(self, cube_name: str) -> str:
        dataset_prefix = self._get_dataset_prefix()
        return f"{dataset_prefix}/{cube_name}" if dataset_prefix else cube_name

    def _get_or_build_histeq_params(self) -> HistEqParams:
        """Return cached histeq params, loading or deriving them on first call.

        A single set of params is derived from *all* available seismic array
        keys in this zarr store (angle stacks and full stack combined), so the
        same mapping is applied uniformly regardless of which key supplies the
        128³ training window.
        """
        if self._histeq_params is not None:
            return self._histeq_params

        # Try loading persisted params first (avoids re-derivation).
        params = load_histeq_params(self.data_path, self.histeq_config)
        derived = False
        if params is None:
            try:
                params = ensure_histeq_params(
                    data_path=self.data_path,
                    array_keys=self.available_cubes,
                    config=self.histeq_config,
                )
            except Exception as exc:
                msg = str(exc).lower()
                if "read-only" not in msg and "read only" not in msg:
                    raise
                # Read-only zarr: derive in-memory without persisting.
                from synthoseis_pre_train.histogram_equalizer import (
                    _collect_representative_sample,
                    derive_histeq_params,
                )
                import zarr as _zarr
                root_ro = _zarr.open(str(self.data_path), mode="r")
                flat = _collect_representative_sample(
                    root_ro, self.available_cubes, self.histeq_config.max_voxels_per_key
                )
                params = derive_histeq_params(flat, nbr_bins=self.histeq_config.nbr_bins)
                if not self._histeq_reported:
                    dataset_prefix = self._get_dataset_prefix()
                    print(
                        f"     . Read-only zarr for '{dataset_prefix}': "
                        "derived histeq transform in-memory (not persisted)"
                    )
            derived = True

        self._histeq_params = params

        if derived and not self._histeq_reported:
            dataset_prefix = self._get_dataset_prefix()
            compact_keys = [_compact_histeq_key_name(key) for key in self.available_cubes]
            print(
                f"     . Derived histogram-equalisation transform for '{dataset_prefix}'\n"
                f"       .. (nbr_bins={self.histeq_config.nbr_bins}, seismic_mean={params.seismic_mean:.4e}\n"
                f"       .. keys={compact_keys})"
            )
            self._histeq_reported = True
        elif not derived and not self._histeq_reported:
            dataset_prefix = self._get_dataset_prefix()
            print(
                f"     . Loaded persisted histeq transform for '{dataset_prefix}'\n"
                f"       .. (seismic_mean={params.seismic_mean:.4e})"
            )
            self._histeq_reported = True

        return params

    def _get_or_build_transform(self, cube_name: str, cube_source) -> tuple[QuantileNormalTransform, bool]:
        transform = self._transform_cache.get(cube_name)
        if transform is not None:
            return transform, False

        try:
            transform = load_quantile_normal_transform(
                data_path=self.data_path,
                array_key=cube_name,
                config=self.quantile_config,
            )
        except Exception:
            # Test doubles and lightweight read adapters may not implement full
            # zarr-group semantics. Fall back to derive path in that case.
            transform = None
        derived = False
        if transform is None:
            try:
                transform = ensure_quantile_normal_transform(
                    data_path=self.data_path,
                    array_key=cube_name,
                    # `cube_source` can be a zarr array or a numpy array. When a persisted
                    # transform already exists this is not materialized. If derivation is
                    # needed, ensure_quantile_normal_transform will read full values once.
                    array_values=cube_source,
                    config=self.quantile_config,
                )
            except Exception as exc:
                msg = str(exc).lower()
                if "read-only" not in msg and "read only" not in msg:
                    raise
                display_key = self._display_key(cube_name)
                if cube_name not in self._reported_readonly_fallback_keys:
                    print(
                        f"     . Read-only zarr for key '{display_key}': deriving non-persistent quantile transform in-memory"
                    )
                    self._reported_readonly_fallback_keys.add(cube_name)
                transform = derive_quantile_normal_transform(
                    array_key=cube_name,
                    array_values=np.asarray(cube_source[:], dtype=np.float32),
                    config=self.quantile_config,
                )
            derived = True

        self._transform_cache[cube_name] = transform

        if derived and cube_name not in self._reported_derived_transform_keys:
            drift = float(transform.metadata.get("source_abs_mean_drift", 0.0))
            symmetry = transform.metadata.get("symmetry_mode", self.quantile_config.symmetry_mode)
            display_key = self._display_key(cube_name)
            print(
                f"     . Derived quantile transform for key '{display_key}' "
                f"(symmetry={symmetry}, eps={self.quantile_config.epsilon:.1e}, "
                f"abs_mean_drift={drift:.3e})"
            )
            self._reported_derived_transform_keys.add(cube_name)

        return transform, derived
    
    def __len__(self) -> int:
        """Return number of samples per epoch.

        Samples are drawn randomly with replacement (overlapping subvolumes),
        so the dataset size is a training hyperparameter, not a physical limit.
        The count of all valid starting positions across all cubes is used
        (stride-1 grid),
        which can be tens of millions — effectively unlimited random sampling.
        """
        # Count valid start positions without loading cube data.
        # Axis 2 (z_zarr) is capped by Z_ARTIFACT_MARGIN to avoid deep artifacts.
        total = 0
        still_available = []
        for cube_name in self.available_cubes:
            try:
                shape = self.zarr_data[cube_name].shape
            except (KeyError, FileNotFoundError, OSError):
                continue

            still_available.append(cube_name)
            positions = 1
            for ax, (dim_size, sample_size) in enumerate(zip(shape, self.sample_shape)):
                effective_size = dim_size - Z_ARTIFACT_MARGIN if ax == 2 else dim_size
                positions *= max(1, effective_size - sample_size + 1)
            total += positions

        # Keep only keys that still exist so future __getitem__ retries stay focused.
        if len(still_available) != len(self.available_cubes):
            self.available_cubes = still_available

        return total
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a single training sample.
        
        Returns:
            input_data: Masked input for the model
            target: Original full data for reconstruction loss
            mask: Boolean mask used to identify masked voxels
        """
        # Select random cube — retry if the zarr key has been deleted on disk
        if self.cached_data:
            cube_name, cube_source = random.choice(self.cached_data)
        else:
            candidates = list(self.available_cubes)
            random.shuffle(candidates)
            cube_source = None
            cube_name = None
            for cube_name in candidates:
                try:
                    # Keep this as a zarr array handle; subvolume extraction below
                    # reads only the requested window instead of the full cube.
                    cube_source = self.zarr_data[cube_name]
                    break
                except (KeyError, FileNotFoundError, OSError):
                    continue
            if cube_source is None:
                raise RuntimeError(
                    f"All array keys unavailable in zarr store "
                    "(zarr may have been deleted during training)\n"
                    f"  Tried: {candidates}"
                )
        
        # Extract random subvolume and augment.
        # augment_pair_3d handles extraction internally for the augment path so
        # that squeezed axes are extracted at a larger size and zoomed cleanly to
        # target_shape, avoiding zero-padded masked borders.
        from synthoseis_pre_train.augmentation import extract_random_subvolume, augment_pair_3d

        if self.augment:
            # CLI/dataset sample_shape is (x, y, z), but augment_pair_3d expects
            # target_shape in training order (z, x, y).
            target_shape_zxy = (
                int(self.sample_shape[2]),
                int(self.sample_shape[0]),
                int(self.sample_shape[1]),
            )
            # `cube_source` is in zarr (x, y, z) order — augment_pair_3d expects that.
            input_data, target, geom_mask, _ = augment_pair_3d(
                cube_source,
                target_shape=target_shape_zxy,
                z_artifact_margin=Z_ARTIFACT_MARGIN,
                normalize=self.apply_standardize,
                target_std=self.target_std,
            )
        else:
            raw = extract_random_subvolume(
                cube_source, self.sample_shape, z_artifact_margin=Z_ARTIFACT_MARGIN
            ).astype(np.float32)
            raw = np.transpose(raw, (2, 0, 1))  # (x, y, z) → (z, x, y)
            target = raw
            if self.apply_standardize:
                from synthoseis_pre_train.masking import normalize_seismic
                target, _, _ = normalize_seismic(target, self.target_std)
            input_data = target.copy()
            geom_mask = np.ones(target.shape, dtype=bool)  # no squeeze edges

        if self.apply_quantile:
            if cube_name is None:
                raise RuntimeError("Internal error: cube_name missing for quantile transform")
            transform, _derived = self._get_or_build_transform(cube_name, cube_source)
            target = transform.forward(target)
            if cube_name not in self._reported_forward_transform_keys:
                print(f"     . Applied forward quantile transform to {self._display_key(cube_name)}")
                self._reported_forward_transform_keys.add(cube_name)
            input_data = target.copy()

        if self.apply_histeq:
            params = self._get_or_build_histeq_params()
            target = _histeq_apply(
                target - params.seismic_mean,
                params.centerbins,
                params.target_centerbins,
            )
            input_data = target.copy()

        # Peak/trough preservation + trace masking — applied to input (x) only.
        # create_mask_3d receives and returns (z, x, y) — no transpose needed.
        from synthoseis_pre_train.masking import create_mask_3d, apply_mask_to_seismic
        trace_mask = create_mask_3d(
            input_data,
            target_masked_fraction=self.target_masked_fraction,
            trace_mask_ratio=self.trace_mask_ratio,
            cluster_shape=self.cluster_shape,
            center_selection_method=self.center_selection_method,
        )
        input_data, _, trace_mask = apply_mask_to_seismic(
            input_data,
            trace_mask,
            fill_method=self.mask_fill_method,
            noise_std=self.mask_noise_std,
        )

        if self.enable_cluster_mask_expansion:
            # Expand masked traces by XY adjacency inside bounds derived from input support.
            # This mirrors diagnostic logic: derive a blue bounds cuboid, detect fully blank
            # traces across z within that cuboid, then include adjacent XY traces.
            z_min, z_max, x_min, x_max, y_min, y_max = _derive_bounds_from_abs_energy(input_data)
            bounded = input_data[z_min:z_max + 1, x_min:x_max + 1, y_min:y_max + 1]
            blank_xy = np.sum(np.abs(bounded), axis=0) <= 1e-12  # [x_span, y_span]
            blank_xy_adj = _dilate_binary_2d(blank_xy, radius=1)

            expanded_trace_mask = np.ones_like(trace_mask, dtype=bool)
            expanded_block = expanded_trace_mask[z_min:z_max + 1, x_min:x_max + 1, y_min:y_max + 1]
            expanded_block[:, blank_xy_adj] = False

            # Union original masked traces with adjacency-expanded blank traces.
            trace_mask = trace_mask & expanded_trace_mask
            input_data, _, trace_mask = apply_mask_to_seismic(
                input_data,
                trace_mask,
                fill_method=self.mask_fill_method,
                noise_std=self.mask_noise_std,
            )

        # Combine with geometric mask: exclude squeeze/t2d edge artifacts from loss.
        # mask=True means "visible to model / included in loss".
        mask = trace_mask & geom_mask

        return (
            input_data.astype(np.float32),
            target.astype(np.float32),
            mask
        )


def create_dataloader(
    data_path: str,
    batch_size: int = 4,
    sample_shape: Tuple[int, int, int] = (128, 128, 128),
    num_workers: int = 0,
    pin_memory: bool = True,
    array_key: Optional[str] = None,
    array_keys: Optional[List[str]] = None,
    **dataset_kwargs
):
    """
    Create a PyTorch DataLoader for seismic data.
    
    Args:
        data_path: Path to Zarr seismic data
        batch_size: Batch size
        sample_shape: Shape of each sample
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for CUDA
        array_key: Single specific 3D array key (legacy)
        array_keys: List of 3D array keys to randomly sample from
        **dataset_kwargs: Additional arguments for SeismicDataset
    
    Returns:
        DataLoader instance
    """
    try:
        import torch
        from torch.utils.data import DataLoader as TorchDataLoader
    except ImportError:
        print("PyTorch not installed. Returning dataset directly.")
        return SeismicDataset(data_path, sample_shape, array_key=array_key, **dataset_kwargs)
    
    dataset = SeismicDataset(
        data_path,
        sample_shape,
        array_key=array_key,
        array_keys=array_keys,
        **dataset_kwargs
    )
    
    loader = TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return loader


def create_merged_dataloader(
    data_paths: List[str],
    batch_size: int = 4,
    sample_shape: Tuple[int, int, int] = (128, 128, 128),
    num_workers: int = 0,
    pin_memory: bool = True,
    array_key: Optional[str] = None,
    array_keys: Optional[List[str]] = None,
    **dataset_kwargs,
):
    """Create one DataLoader over a ConcatDataset spanning multiple zarr stores.

    Each source path is opened as a SeismicDataset with the same sampling and
    preprocessing configuration.  The resulting ConcatDataset is shuffled at
    DataLoader level, which mixes samples from all input datasets throughout each
    epoch.

    Args:
        data_paths: List of zarr store paths to merge.
        batch_size: Batch size.
        sample_shape: Shape of each sample.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory for CUDA.
        array_key: Single specific 3D array key (legacy).
        array_keys: List of 3D array keys to randomly sample from.
        **dataset_kwargs: Additional arguments for SeismicDataset.

    Returns:
        DataLoader over torch.utils.data.ConcatDataset.

    Raises:
        ValueError: If no paths are provided or all dataset paths fail to open.
    """
    if not data_paths:
        raise ValueError("data_paths must contain at least one path.")

    try:
        from torch.utils.data import ConcatDataset, DataLoader as TorchDataLoader
    except ImportError:
        raise ImportError("PyTorch not installed. create_merged_dataloader requires torch.")

    datasets = []
    failed = []

    for data_path in data_paths:
        try:
            dataset = SeismicDataset(
                data_path=data_path,
                sample_shape=sample_shape,
                array_key=array_key,
                array_keys=array_keys,
                **dataset_kwargs,
            )
            datasets.append(dataset)
        except Exception as exc:
            failed.append(f"{data_path}: {exc}")

    if failed:
        import warnings
        warnings.warn(
            f"Skipped {len(failed)} dataset(s) that could not be opened:\n"
            + "\n".join(f"  {msg}" for msg in failed),
            stacklevel=2,
        )

    if not datasets:
        raise ValueError(
            "No datasets could be opened. Check that data_paths are valid zarr stores."
        )

    merged_dataset = ConcatDataset(datasets)
    merged_loader = TorchDataLoader(
        merged_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return merged_loader
