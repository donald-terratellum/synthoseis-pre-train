import numpy as np

import synthoseis_pre_train.dataloader as dl


class _FakeArray:
    def __init__(self, arr: np.ndarray):
        self._arr = arr
        self.shape = arr.shape

    def __getitem__(self, key):
        return self._arr[key]


class _FakeZarrStore:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def array_keys(self):
        return ["cube"]

    def __getitem__(self, key):
        if key != "cube":
            raise KeyError(key)
        return _FakeArray(self._arr)


def _patch_masking_noop(monkeypatch):
    def _create_mask_3d(
        x,
        target_masked_fraction=None,
        trace_mask_ratio=0.07,
        cluster_shape=3,
        center_selection_method="random_mixture",
        random_seed=None,
    ):
        return np.ones_like(x, dtype=bool)

    def _apply_mask_to_seismic(x, mask, fill_value=0.0, fill_method="zero", noise_std=1e-2):
        return x, x.copy(), mask

    monkeypatch.setattr("synthoseis_pre_train.masking.create_mask_3d", _create_mask_3d)
    monkeypatch.setattr("synthoseis_pre_train.masking.apply_mask_to_seismic", _apply_mask_to_seismic)


def test_non_augment_non_cubic_shape_returns_zxy(monkeypatch):
    # zarr is (x, y, z)
    fake_cube = np.random.randn(160, 120, 80).astype(np.float32)
    monkeypatch.setattr(dl.zarr, "open", lambda *args, **kwargs: _FakeZarrStore(fake_cube))
    _patch_masking_noop(monkeypatch)

    ds = dl.SeismicDataset(
        data_path="ignored.zarr",
        sample_shape=(128, 96, 64),  # (x, y, z)
        augment=False,
        normalize=False,
    )

    inp, tgt, mask = ds[0]
    assert inp.shape == (64, 128, 96)
    assert tgt.shape == (64, 128, 96)
    assert mask.shape == (64, 128, 96)


def test_augment_non_cubic_shape_uses_zxy_target_shape(monkeypatch):
    # zarr is (x, y, z)
    fake_cube = np.random.randn(160, 120, 80).astype(np.float32)
    monkeypatch.setattr(dl.zarr, "open", lambda *args, **kwargs: _FakeZarrStore(fake_cube))
    _patch_masking_noop(monkeypatch)

    seen = {}

    def _augment_pair_3d(cube, target_shape, z_artifact_margin=0, normalize=True, target_std=1.0):
        seen["target_shape"] = tuple(target_shape)
        z, x, y = target_shape
        base = np.ones((z, x, y), dtype=np.float32)
        geom = np.ones((z, x, y), dtype=bool)
        return base, base.copy(), geom, {}

    monkeypatch.setattr("synthoseis_pre_train.augmentation.augment_pair_3d", _augment_pair_3d)

    ds = dl.SeismicDataset(
        data_path="ignored.zarr",
        sample_shape=(128, 96, 64),  # (x, y, z)
        augment=True,
        normalize=False,
    )

    inp, tgt, mask = ds[0]
    assert seen["target_shape"] == (64, 128, 96)
    assert inp.shape == (64, 128, 96)
    assert tgt.shape == (64, 128, 96)
    assert mask.shape == (64, 128, 96)


def test_quantile_mode_disables_augmentation_standardization(monkeypatch):
    fake_cube = np.random.randn(96, 96, 96).astype(np.float32)
    monkeypatch.setattr(dl.zarr, "open", lambda *args, **kwargs: _FakeZarrStore(fake_cube))
    _patch_masking_noop(monkeypatch)

    seen = {}

    def _augment_pair_3d(cube, target_shape, z_artifact_margin=0, normalize=True, target_std=1.0):
        seen["normalize"] = normalize
        z, x, y = target_shape
        base = np.ones((z, x, y), dtype=np.float32)
        geom = np.ones((z, x, y), dtype=bool)
        return base, base.copy(), geom, {}

    monkeypatch.setattr("synthoseis_pre_train.augmentation.augment_pair_3d", _augment_pair_3d)

    class _IdentityTransform:
        metadata = {"source_abs_mean_drift": 0.0}

        def forward(self, x):
            return x

    monkeypatch.setattr(
        dl,
        "ensure_quantile_normal_transform",
        lambda **kwargs: _IdentityTransform(),
    )

    ds = dl.SeismicDataset(
        data_path="ignored.zarr",
        sample_shape=(64, 64, 64),
        augment=True,
        normalize=True,
        amplitude_transform="quantile_normal",
    )
    _ = ds[0]
    assert seen["normalize"] is False
