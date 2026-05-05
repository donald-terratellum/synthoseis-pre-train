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

# Zarr z-axis (axis 2) has known artifacts in the last N indices.
# Restrict random subvolume starts so the deepest sampled z-index is < z_size - Z_ARTIFACT_MARGIN.
Z_ARTIFACT_MARGIN = 2


class SeismicDataset:
    """
    Dataset for seismic pre-training with masking and augmentation.
    """
    
    def __init__(
        self,
        data_path: str,
        sample_shape: Tuple[int, int, int] = (128, 128, 128),
        trace_mask_ratio: float = 0.07,
        augment: bool = True,
        normalize: bool = True,
        target_std: float = 1.0,
        cache_in_memory: bool = False,
        array_key: Optional[str] = None,
        array_keys: Optional[List[str]] = None,
    ):
        """
        Args:
            data_path: Path to Zarr seismic data
            sample_shape: Shape of each training sample (x, y, z)
            trace_mask_ratio: Ratio of traces to mask
            augment: Whether to apply data augmentation
            normalize: Whether to normalize samples
            target_std: Target standard deviation after normalization
            cache_in_memory: Whether to cache all data in memory
            array_key: Single specific 3D array key to use (legacy; takes precedence)
            array_keys: List of 3D array keys to randomly sample from each __getitem__
        """
        self.data_path = Path(data_path)
        self.sample_shape = sample_shape
        self.trace_mask_ratio = trace_mask_ratio
        self.augment = augment
        self.normalize = normalize
        self.target_std = target_std
        
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
            self.cached_data.append(cube)
        print(f"Cached {len(self.cached_data)} cubes")
    
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
            cube = random.choice(self.cached_data)
        else:
            candidates = list(self.available_cubes)
            random.shuffle(candidates)
            cube = None
            for cube_name in candidates:
                try:
                    cube = self.zarr_data[cube_name][:]
                    break
                except (KeyError, FileNotFoundError, OSError):
                    continue
            if cube is None:
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
            # cube is still in zarr (x, y, z) order — augment_pair_3d expects that.
            input_data, target, geom_mask, _ = augment_pair_3d(
                cube,
                target_shape=self.sample_shape,
                z_artifact_margin=Z_ARTIFACT_MARGIN,
                normalize=self.normalize,
                target_std=self.target_std,
            )
        else:
            raw = extract_random_subvolume(
                cube, self.sample_shape, z_artifact_margin=Z_ARTIFACT_MARGIN
            ).astype(np.float32)
            raw = np.transpose(raw, (2, 0, 1))  # (x, y, z) → (z, x, y)
            target = raw
            if self.normalize:
                from synthoseis_pre_train.masking import normalize_seismic
                target, _, _ = normalize_seismic(target, self.target_std)
            input_data = target.copy()
            geom_mask = np.ones(target.shape, dtype=bool)  # no squeeze edges

        # Peak/trough preservation + trace masking — applied to input (x) only.
        # create_mask_3d receives and returns (z, x, y) — no transpose needed.
        from synthoseis_pre_train.masking import create_mask_3d, apply_mask_to_seismic
        trace_mask = create_mask_3d(input_data, trace_mask_ratio=self.trace_mask_ratio)
        input_data, _, trace_mask = apply_mask_to_seismic(input_data, trace_mask)

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
