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
import json

# Zarr z-axis (axis 2) has known artifacts in the last N indices.
# Restrict random subvolume starts so the deepest sampled z-index is < z_size - Z_ARTIFACT_MARGIN.
Z_ARTIFACT_MARGIN = 2
DEFAULT_GEOLOGIC_SCORE_KEYS = ["geological_score", "geologic_score"]
DEFAULT_GEO_POINTS_JSON = "geologic_score_selected_points.json"
DEFAULT_GEO_VAL_CENTER_JSON = "geologic_score_val_center.json"


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
        geologic_score_sampling: bool = True,
        geologic_score_min: float = 0.5,
        geologic_score_key_candidates: Optional[List[str]] = None,
        geologic_points_json_name: str = DEFAULT_GEO_POINTS_JSON,
        geologic_val_center_json_name: str = DEFAULT_GEO_VAL_CENTER_JSON,
        geologic_target_points: int = 500,
        geologic_candidate_count: int = 3000,
        geologic_candidate_probes: int = 24,
        geologic_dist_thresh_start: int = 96,
        geologic_dist_thresh_floor: int = 32,
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
        self.geologic_score_sampling = bool(geologic_score_sampling)
        self.geologic_score_min = float(geologic_score_min)
        self.geologic_score_key_candidates = (
            list(geologic_score_key_candidates)
            if geologic_score_key_candidates is not None
            else list(DEFAULT_GEOLOGIC_SCORE_KEYS)
        )
        self.geologic_points_json_path = self.data_path.parent / geologic_points_json_name
        self.geologic_val_center_json_path = self.data_path.parent / geologic_val_center_json_name
        self.geologic_target_points = int(geologic_target_points)
        self.geologic_candidate_count = int(geologic_candidate_count)
        self.geologic_candidate_probes = int(geologic_candidate_probes)
        self.geologic_dist_thresh_start = int(geologic_dist_thresh_start)
        self.geologic_dist_thresh_floor = int(geologic_dist_thresh_floor)
        self._ranked_points_xyz: Optional[np.ndarray] = None
        self._ranked_point_scores: Optional[np.ndarray] = None
        self._fixed_val_center_xyz: Optional[Tuple[int, int, int]] = None
        self._score_key: Optional[str] = None
        
        # Load zarr data
        self.zarr_data = zarr.open(str(data_path), mode='r')

        all_3d_keys = [
            key for key in self.zarr_data.array_keys()
            if len(self.zarr_data[key].shape) == 3 and key not in DEFAULT_GEOLOGIC_SCORE_KEYS
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

        if self.geologic_score_sampling:
            self._init_geologic_score_sampling()
    
    def _cache_data(self):
        """Cache all data in memory."""
        print("Caching seismic data in memory...")
        self.cached_data = []
        for cube_name in self.available_cubes:
            cube = self.zarr_data[cube_name][:]
            self.cached_data.append(cube)
        print(f"Cached {len(self.cached_data)} cubes")

    def _init_geologic_score_sampling(self) -> None:
        """Load or build ranked geologic-score points and optional fixed val center."""
        self._score_key = self._resolve_score_key()
        if self._score_key is None:
            print(
                f"WARNING: geologic score key not found in {self.data_path}; "
                "falling back to random crop centers."
            )
            self.geologic_score_sampling = False
            return

        points, scores = self._load_or_build_ranked_points()
        if len(points) == 0:
            print(
                f"WARNING: no ranked points were available for {self.data_path}; "
                "falling back to random crop centers."
            )
            self.geologic_score_sampling = False
            return

        self._ranked_points_xyz = points
        self._ranked_point_scores = scores

        # Validation loaders (augment=False) keep one persistent center per dataset.
        if not self.augment:
            self._fixed_val_center_xyz = self._load_or_create_fixed_val_center()

    def _resolve_score_key(self) -> Optional[str]:
        keys = set(self.zarr_data.array_keys())
        for key in self.geologic_score_key_candidates:
            if key in keys and len(self.zarr_data[key].shape) == 3:
                return key
        return None

    def _triangular_rank_index(self, n: int) -> int:
        if n <= 0:
            raise ValueError("n must be > 0")
        idx = int(np.random.triangular(0.0, 0.0, float(n)))
        return min(max(idx, 0), n - 1)

    def _draw_center_from_ranked_points(self) -> Tuple[int, int, int]:
        assert self._ranked_points_xyz is not None
        idx = self._triangular_rank_index(len(self._ranked_points_xyz))
        x, y, z = self._ranked_points_xyz[idx]
        return int(x), int(y), int(z)

    def _score_usable_shape(self) -> Tuple[int, int, int]:
        assert self._score_key is not None
        score_shape = self.zarr_data[self._score_key].shape
        cube_shapes = [self.zarr_data[name].shape for name in self.available_cubes]
        if not cube_shapes:
            cube_shape = score_shape
        else:
            cube_shape = cube_shapes[0]
        return tuple(min(int(a), int(b)) for a, b in zip(score_shape, cube_shape))

    def _center_bounds(self, shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        # Keep selected center at least half sample size from each boundary.
        margin = np.asarray([s // 2 for s in self.sample_shape], dtype=np.int32)
        low = margin
        high = np.asarray(shape, dtype=np.int32) - margin - 1
        if np.any(high < low):
            raise ValueError(
                f"Dataset shape {shape} too small for sample shape {self.sample_shape}"
            )
        return low, high

    def _random_point_within(self, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        return np.array(
            [np.random.randint(int(lo), int(hi) + 1) for lo, hi in zip(low, high)],
            dtype=np.int32,
        )

    def _best_candidate_point(
        self,
        existing: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
    ) -> np.ndarray:
        probes = np.stack(
            [self._random_point_within(low, high) for _ in range(self.geologic_candidate_probes)],
            axis=0,
        )
        if existing.size == 0:
            return probes[0]
        diffs = probes[:, None, :].astype(np.float32) - existing[None, :, :].astype(np.float32)
        min_distances = np.sqrt(np.sum(diffs * diffs, axis=2)).min(axis=1)
        return probes[int(np.argmax(min_distances))]

    def _generate_candidate_points(self, shape: Tuple[int, int, int]) -> np.ndarray:
        low, high = self._center_bounds(shape)
        points: list[np.ndarray] = []
        existing = np.empty((0, 3), dtype=np.int32)
        for _ in range(self.geologic_candidate_count):
            point = self._best_candidate_point(existing, low, high)
            points.append(point)
            existing = np.asarray(points, dtype=np.int32)
        return existing

    def _select_ranked_points(
        self,
        score_values: np.ndarray,
        points_xyz: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        keep = score_values >= float(self.geologic_score_min)
        points = points_xyz[keep]
        scores = score_values[keep]
        if len(points) == 0:
            return np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.float32)

        order = np.argsort(scores)[::-1]
        ordered_points = points[order]
        ordered_scores = scores[order]

        accepted_points: list[np.ndarray] = [ordered_points[0]]
        accepted_scores: list[float] = [float(ordered_scores[0])]
        remaining = list(range(1, len(ordered_points)))
        dist_thresh = int(self.geologic_dist_thresh_start)

        while remaining and len(accepted_points) < self.geologic_target_points:
            added = False
            next_remaining: list[int] = []
            for idx in remaining:
                point = ordered_points[idx]
                reference = np.asarray(accepted_points[-3:], dtype=np.int32)
                if len(accepted_points) < 3:
                    reference = np.asarray(accepted_points, dtype=np.int32)
                diffs = reference.astype(np.float32) - point.astype(np.float32)
                min_dist = float(np.sqrt(np.sum(diffs * diffs, axis=1)).min())
                if min_dist > dist_thresh:
                    accepted_points.append(point)
                    accepted_scores.append(float(ordered_scores[idx]))
                    added = True
                    if len(accepted_points) >= self.geologic_target_points:
                        break
                else:
                    next_remaining.append(idx)
            remaining = next_remaining
            if len(accepted_points) >= self.geologic_target_points:
                break
            if not added:
                if dist_thresh <= self.geologic_dist_thresh_floor:
                    break
                dist_thresh = max(self.geologic_dist_thresh_floor, dist_thresh - 8)

        return (
            np.asarray(accepted_points, dtype=np.int32),
            np.asarray(accepted_scores, dtype=np.float32),
        )

    def _load_or_build_ranked_points(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.geologic_points_json_path.exists():
            payload = json.loads(self.geologic_points_json_path.read_text(encoding="utf-8"))
            points = np.asarray(payload.get("selected_points", []), dtype=np.int32)
            scores = np.asarray(payload.get("selected_scores", []), dtype=np.float32)
            if len(points) > 0 and len(points) == len(scores):
                return points, scores

        assert self._score_key is not None
        shape = self._score_usable_shape()
        points = self._generate_candidate_points(shape)
        score_arr = self.zarr_data[self._score_key]
        score_values = np.asarray(score_arr[points[:, 0], points[:, 1], points[:, 2]], dtype=np.float32)
        selected_points, selected_scores = self._select_ranked_points(score_values, points)

        payload = {
            "dataset": str(self.data_path),
            "score_key": self._score_key,
            "sample_shape": [int(v) for v in self.sample_shape],
            "min_score": float(self.geologic_score_min),
            "selected_points": selected_points.astype(int).tolist(),
            "selected_scores": selected_scores.astype(float).tolist(),
        }
        self.geologic_points_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return selected_points, selected_scores

    def _load_or_create_fixed_val_center(self) -> Tuple[int, int, int]:
        if self.geologic_val_center_json_path.exists():
            payload = json.loads(self.geologic_val_center_json_path.read_text(encoding="utf-8"))
            point = payload.get("center_xyz")
            if isinstance(point, list) and len(point) == 3:
                return int(point[0]), int(point[1]), int(point[2])

        center_xyz = self._draw_center_from_ranked_points()
        payload = {
            "dataset": str(self.data_path),
            "center_xyz": [int(v) for v in center_xyz],
            "source_points_json": str(self.geologic_points_json_path),
        }
        self.geologic_val_center_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return center_xyz
    
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

        center_xyz: Optional[Tuple[int, int, int]] = None
        if self.geologic_score_sampling and self._ranked_points_xyz is not None:
            if self.augment:
                center_xyz = self._draw_center_from_ranked_points()
            else:
                center_xyz = self._fixed_val_center_xyz
        
        # Extract random subvolume and augment.
        # augment_pair_3d handles extraction internally for the augment path so
        # that squeezed axes are extracted at a larger size and zoomed cleanly to
        # target_shape, avoiding zero-padded masked borders.
        from synthoseis_pre_train.augmentation import (
            extract_random_subvolume,
            extract_centered_subvolume,
            augment_pair_3d,
        )

        if self.augment:
            # cube is still in zarr (x, y, z) order — augment_pair_3d expects that.
            input_data, target, geom_mask, _ = augment_pair_3d(
                cube,
                target_shape=self.sample_shape,
                center_xyz=center_xyz,
                z_artifact_margin=Z_ARTIFACT_MARGIN,
                normalize=self.normalize,
                target_std=self.target_std,
            )
        else:
            if center_xyz is None:
                raw = extract_random_subvolume(
                    cube, self.sample_shape, z_artifact_margin=Z_ARTIFACT_MARGIN
                ).astype(np.float32)
            else:
                raw = extract_centered_subvolume(
                    cube,
                    self.sample_shape,
                    center_xyz=center_xyz,
                    z_artifact_margin=Z_ARTIFACT_MARGIN,
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
