"""Unit tests for create_merged_dataloader.

These tests monkeypatch SeismicDataset so they do not depend on real zarr files.
"""

import numpy as np
import pytest

from synthoseis_pre_train import dataloader as dl


class _FakeSeismicDataset:
    """Simple in-memory stand-in for SeismicDataset."""

    def __init__(
        self,
        data_path: str,
        sample_shape=(128, 128, 128),
        trace_mask_ratio: float = 0.07,
        augment: bool = True,
        normalize: bool = True,
        target_std: float = 1.0,
        cache_in_memory: bool = False,
        array_key=None,
        array_keys=None,
    ):
        if "bad" in data_path:
            raise ValueError(f"invalid dataset path: {data_path}")
        self.data_path = data_path
        self.sample_shape = sample_shape
        self.available_cubes = ["cube0"]

    def __len__(self):
        return 8

    def __getitem__(self, idx):
        shape = (128, 128, 128)
        input_data = np.ones(shape, dtype=np.float32)
        target = np.ones(shape, dtype=np.float32) * 2.0
        mask = np.ones(shape, dtype=bool)
        return input_data, target, mask


def _patch_dataset(monkeypatch):
    monkeypatch.setattr(dl, "SeismicDataset", _FakeSeismicDataset)


def test_merged_dataloader_returns_dataloader(monkeypatch):
    _patch_dataset(monkeypatch)
    loader = dl.create_merged_dataloader(["good_a", "good_b"], batch_size=2)

    from torch.utils.data import DataLoader

    assert isinstance(loader, DataLoader)


def test_merged_len_is_sum_of_dataset_lens(monkeypatch):
    _patch_dataset(monkeypatch)
    loader = dl.create_merged_dataloader(["good_a", "good_b", "good_c"])
    assert len(loader.dataset) == 24


def test_merged_batch_shape(monkeypatch):
    _patch_dataset(monkeypatch)
    loader = dl.create_merged_dataloader(["good_a", "good_b"], batch_size=2)
    input_data, target, mask = next(iter(loader))

    assert tuple(input_data.shape) == (2, 128, 128, 128)
    assert tuple(target.shape) == (2, 128, 128, 128)
    assert tuple(mask.shape) == (2, 128, 128, 128)


def test_merged_three_tensors_per_batch(monkeypatch):
    _patch_dataset(monkeypatch)
    loader = dl.create_merged_dataloader(["good_a", "good_b"], batch_size=2)
    batch = next(iter(loader))

    assert isinstance(batch, (tuple, list))
    assert len(batch) == 3


def test_merged_single_path_equivalent_to_create_dataloader(monkeypatch):
    _patch_dataset(monkeypatch)
    merged = dl.create_merged_dataloader(["good_a"], batch_size=2)
    single = dl.create_dataloader("good_a", batch_size=2)

    assert len(merged) == len(single)


def test_merged_empty_paths_raises(monkeypatch):
    _patch_dataset(monkeypatch)
    with pytest.raises(ValueError, match="data_paths must contain"):
        dl.create_merged_dataloader([])


def test_merged_all_bad_paths_raises(monkeypatch):
    _patch_dataset(monkeypatch)
    with pytest.raises(ValueError, match="No datasets could be opened"):
        dl.create_merged_dataloader(["bad_a", "bad_b"])


def test_merged_one_bad_path_warns_and_continues(monkeypatch):
    _patch_dataset(monkeypatch)
    with pytest.warns(UserWarning, match="Skipped 1 dataset"):
        loader = dl.create_merged_dataloader(["good_a", "bad_b"])

    assert len(loader.dataset) == 8


def test_merged_length_scales_with_dataset_count(monkeypatch):
    _patch_dataset(monkeypatch)
    loader = dl.create_merged_dataloader(["good_a", "good_b", "good_c"])
    assert len(loader.dataset) == 24
