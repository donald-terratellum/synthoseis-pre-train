import numpy as np
import zarr

from synthoseis_pre_train.dataloader import SeismicDataset
from synthoseis_pre_train.transforms import (
    QuantileNormalConfig,
    ensure_quantile_normal_transform,
    load_quantile_normal_transform,
)


def test_strict_odd_forward_inverse_roundtrip(tmp_path):
    data_path = tmp_path / "model_data.zarr"
    root = zarr.open(str(data_path), mode="w")
    base = np.linspace(-3.0, 3.0, 10000, dtype=np.float32)
    root.create_array("cube", data=base.reshape(100, 100, 1), overwrite=True)

    cfg = QuantileNormalConfig(epsilon=1e-6, symmetry_mode="strict_odd")
    transform = ensure_quantile_normal_transform(
        data_path=data_path,
        array_key="cube",
        array_values=base,
        config=cfg,
    )

    x = np.array([-2.2, -0.8, 0.0, 0.8, 2.2], dtype=np.float32)
    y = transform.forward(x)
    y_neg = transform.forward(-x)
    np.testing.assert_allclose(y_neg, -y, rtol=0.0, atol=1e-5)

    x_rec = transform.inverse(y)
    np.testing.assert_allclose(x_rec, x, rtol=0.0, atol=5e-3)


def test_existing_transform_is_reused(tmp_path):
    data_path = tmp_path / "model_data.zarr"
    root = zarr.open(str(data_path), mode="w")
    arr = np.random.randn(16, 12, 8).astype(np.float32)
    root.create_array("cube", data=arr, overwrite=True)

    cfg = QuantileNormalConfig(epsilon=1e-6, symmetry_mode="independent")
    first = ensure_quantile_normal_transform(
        data_path=data_path,
        array_key="cube",
        array_values=arr,
        config=cfg,
    )

    # Re-run with very different values; existing transform should be reused.
    second = ensure_quantile_normal_transform(
        data_path=data_path,
        array_key="cube",
        array_values=np.ones_like(arr, dtype=np.float32) * 123.0,
        config=cfg,
    )

    np.testing.assert_array_equal(first.x_lut, second.x_lut)
    np.testing.assert_array_equal(first.z_lut, second.z_lut)


def test_per_array_key_transform_is_distinct(tmp_path):
    data_path = tmp_path / "model_data.zarr"
    root = zarr.open(str(data_path), mode="w")
    arr_a = np.random.normal(loc=0.0, scale=1.0, size=(20, 20, 20)).astype(np.float32)
    arr_b = np.random.normal(loc=5.0, scale=3.0, size=(20, 20, 20)).astype(np.float32)
    root.create_array("cube_a", data=arr_a, overwrite=True)
    root.create_array("cube_b", data=arr_b, overwrite=True)

    cfg = QuantileNormalConfig(epsilon=1e-6, symmetry_mode="independent")
    ta = ensure_quantile_normal_transform(
        data_path=data_path,
        array_key="cube_a",
        array_values=arr_a,
        config=cfg,
    )
    tb = ensure_quantile_normal_transform(
        data_path=data_path,
        array_key="cube_b",
        array_values=arr_b,
        config=cfg,
    )

    assert not np.allclose(ta.x_lut, tb.x_lut)


def test_dataset_lazy_derives_quantile_transform(monkeypatch, tmp_path):
    data_path = tmp_path / "model_data.zarr"
    root = zarr.open(str(data_path), mode="w")
    cube = np.random.randn(64, 64, 64).astype(np.float32)
    root.create_array("cube", data=cube, overwrite=True)

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

    ds = SeismicDataset(
        data_path=str(data_path),
        sample_shape=(32, 32, 32),
        augment=False,
        normalize=True,
        array_key="cube",
        amplitude_transform="quantile_normal",
        quantile_symmetry_mode="strict_odd",
        quantile_epsilon=1e-6,
    )

    inp, tgt, mask = ds[0]
    assert inp.dtype == np.float32
    assert tgt.dtype == np.float32
    assert mask.dtype == bool

    loaded = load_quantile_normal_transform(
        data_path=data_path,
        array_key="cube",
        config=QuantileNormalConfig(epsilon=1e-6, symmetry_mode="strict_odd"),
    )
    assert loaded is not None
