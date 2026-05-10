"""Unit tests for seismic loss functions."""

import torch

from synthoseis_pre_train.losses import SSIMMSELoss3D


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
    expected = (((pred - target) ** 2) * valid).sum() / valid.sum().clamp_min(1.0)

    assert torch.allclose(loss, expected)


def test_ssim_mse_alpha_one_skips_mse_term():
    target = torch.zeros(1, 1, 16, 16, 16, dtype=torch.float32)
    pred = torch.ones_like(target)
    valid = torch.ones_like(target)

    pure_ssim = SSIMMSELoss3D(alpha=1.0)(pred, target, valid_mask=valid)
    mixed = SSIMMSELoss3D(alpha=0.5)(pred, target, valid_mask=valid)

    assert float(pure_ssim) <= float(mixed)
