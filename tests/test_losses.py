"""Unit tests for seismic loss functions."""

import torch

from synthoseis_pre_train.losses import SSIMMSELoss3D, SlidingWindowStatsLoss3D


def test_ssim_mse_is_near_zero_for_perfect_match():
    loss_fn = SSIMMSELoss3D(
        data_range=30.0,
        window_size=16,
        sigma=16.0 / 6.0,
        alpha=1.0 / 6.0,
        min_valid_ratio=0.5,
    )
    target = torch.randn(1, 1, 32, 32, 32, dtype=torch.float32)
    pred = target.clone()
    valid = torch.ones_like(target)

    loss = loss_fn(pred, target, valid_mask=valid)
    assert float(loss) < 1e-6


def test_ssim_mse_increases_with_error():
    loss_fn = SSIMMSELoss3D(
        data_range=30.0,
        window_size=16,
        sigma=16.0 / 6.0,
        alpha=1.0 / 6.0,
        min_valid_ratio=0.5,
    )
    target = torch.randn(1, 1, 32, 32, 32, dtype=torch.float32)
    valid = torch.ones_like(target)

    loss_clean = loss_fn(target.clone(), target, valid_mask=valid)
    loss_noisy = loss_fn(target + 0.5 * torch.randn_like(target), target, valid_mask=valid)

    assert float(loss_noisy) > float(loss_clean)


def test_ssim_mse_respects_valid_mask():
    loss_fn = SSIMMSELoss3D(
        data_range=30.0,
        window_size=16,
        sigma=16.0 / 6.0,
        alpha=1.0 / 6.0,
        min_valid_ratio=0.5,
    )
    target = torch.zeros(1, 1, 32, 32, 32, dtype=torch.float32)
    pred = target.clone()

    # Inject large mismatch in a subregion but mark it invalid.
    pred[:, :, :8, :8, :8] = 20.0
    valid = torch.ones_like(target)
    valid[:, :, :8, :8, :8] = 0.0

    masked_loss = loss_fn(pred, target, valid_mask=valid)

    # Without masking this region, loss should be larger.
    unmasked_loss = loss_fn(pred, target, valid_mask=torch.ones_like(target))

    assert float(masked_loss) < float(unmasked_loss)


def test_ssim_mse_alpha_zero_matches_masked_mse():
    loss_fn = SSIMMSELoss3D(alpha=0.0)
    target = torch.randn(1, 1, 16, 16, 16, dtype=torch.float32)
    pred = target + 0.25 * torch.randn_like(target)
    valid = torch.ones_like(target)
    valid[:, :, :4, :4, :4] = 0.0

    loss = loss_fn(pred, target, valid_mask=valid)
    # SSIMMSELoss3D rescales zero-centered seismic amplitudes to [0, 1]
    # before computing both MSE and SSIM terms.
    pred_scaled = torch.clamp(pred / loss_fn.data_range + 0.5, 0.0, 1.0)
    target_scaled = torch.clamp(target / loss_fn.data_range + 0.5, 0.0, 1.0)
    expected = (((pred_scaled - target_scaled) ** 2) * valid).sum() / valid.sum().clamp_min(1.0)

    assert torch.allclose(loss, expected, atol=1e-7, rtol=1e-5)


def test_ssim_mse_alpha_one_skips_mse_term():
    target = torch.zeros(1, 1, 16, 16, 16, dtype=torch.float32)
    pred = torch.ones_like(target)
    valid = torch.ones_like(target)

    pure_ssim = SSIMMSELoss3D(alpha=1.0)(pred, target, valid_mask=valid)
    mixed = SSIMMSELoss3D(alpha=0.5)(pred, target, valid_mask=valid)

    assert float(pure_ssim) <= float(mixed)


def test_sliding_stats_is_near_zero_for_perfect_match():
    loss_fn = SlidingWindowStatsLoss3D(window_size=(9, 9, 9))
    target = torch.randn(1, 1, 20, 20, 20, dtype=torch.float32)
    pred = target.clone()
    valid = torch.ones_like(target)

    loss = loss_fn(pred, target, valid_mask=valid)
    assert float(loss) < 1e-6


def test_sliding_stats_increases_when_local_std_mismatches():
    loss_fn = SlidingWindowStatsLoss3D(window_size=(9, 9, 9))
    target = torch.randn(1, 1, 20, 20, 20, dtype=torch.float32)
    valid = torch.ones_like(target)

    loss_clean = loss_fn(target.clone(), target, valid_mask=valid)
    # Reduce prediction variance to create local std-ratio error.
    pred_low_std = 0.3 * target
    loss_low_std = loss_fn(pred_low_std, target, valid_mask=valid)

    assert float(loss_low_std) > float(loss_clean)


def test_sliding_stats_all_voxels_ignores_valid_mask_holes():
    target = torch.randn(1, 1, 20, 20, 20, dtype=torch.float32)
    pred = target.clone()
    pred[:, :, :6, :6, :6] += 5.0

    valid = torch.ones_like(target)
    valid[:, :, :6, :6, :6] = 0.0

    masked_loss = SlidingWindowStatsLoss3D(window_size=(9, 9, 9), apply_to_all_voxels=False)(
        pred, target, valid_mask=valid
    )
    all_voxels_loss = SlidingWindowStatsLoss3D(window_size=(9, 9, 9), apply_to_all_voxels=True)(
        pred, target, valid_mask=valid
    )

    assert float(all_voxels_loss) > float(masked_loss)


def test_sliding_stats_minmax_terms_are_near_zero_for_perfect_match():
    loss_fn = SlidingWindowStatsLoss3D(
        window_size=(9, 9, 9),
        mean_weight=0.0,
        std_weight=0.0,
        min_weight=1.0,
        max_weight=1.0,
    )
    target = torch.randn(1, 1, 20, 20, 20, dtype=torch.float32)
    pred = target.clone()
    valid = torch.ones_like(target)

    loss = loss_fn(pred, target, valid_mask=valid)
    assert float(loss) < 1e-6


def test_sliding_stats_minmax_terms_increase_with_local_extrema_mismatch():
    loss_fn = SlidingWindowStatsLoss3D(
        window_size=(9, 9, 9),
        mean_weight=0.0,
        std_weight=0.0,
        min_weight=1.0,
        max_weight=1.0,
    )
    target = torch.randn(1, 1, 20, 20, 20, dtype=torch.float32)
    valid = torch.ones_like(target)

    clean_loss = loss_fn(target.clone(), target, valid_mask=valid)
    pred = target.clone()
    pred[:, :, 4:8, 4:8, 4:8] += 3.0
    pred[:, :, 10:14, 10:14, 10:14] -= 3.0
    mismatch_loss = loss_fn(pred, target, valid_mask=valid)

    assert float(mismatch_loss) > float(clean_loss)


def test_sliding_stats_mae_mse_terms_match_direct_masked_reductions():
    target = torch.randn(1, 1, 16, 16, 16, dtype=torch.float32)
    pred = target + 0.25 * torch.randn_like(target)
    valid = torch.ones_like(target)
    valid[:, :, :4, :4, :4] = 0.0

    loss_fn = SlidingWindowStatsLoss3D(
        window_size=(9, 9, 9),
        mean_weight=0.0,
        std_weight=0.0,
        min_weight=0.0,
        max_weight=0.0,
        mae_weight=1.0,
        mse_weight=1.0,
        apply_to_all_voxels=False,
    )
    loss = loss_fn(pred, target, valid_mask=valid)

    denom = valid.sum().clamp_min(1.0)
    expected_mae = (torch.abs(pred - target) * valid).sum() / denom
    expected_mse = (((pred - target) ** 2) * valid).sum() / denom
    expected = expected_mae + expected_mse

    assert torch.allclose(loss, expected, atol=1e-6, rtol=1e-5)
