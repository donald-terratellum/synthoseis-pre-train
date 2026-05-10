import torch
import pytest

from synthoseis_pre_train.losses import SSIMMSELoss3D, CompositeClusterAwareLoss
import torch.nn as nn


def test_composite_without_valid_mask_equals_base_weight():
    """When no valid_mask is supplied the cluster selection is empty and
    the composite should reduce to base_weight * base_loss.
    """
    B, C, D, H, W = 2, 1, 4, 8, 8
    pred = torch.zeros((B, C, D, H, W), dtype=torch.float32)
    target = torch.ones((B, C, D, H, W), dtype=torch.float32)

    base = SSIMMSELoss3D(alpha=0.0)  # pure MSE
    comp = CompositeClusterAwareLoss(base_criterion=base, kernel_size=5, eps=1e-6)

    L_base = base(pred, target, valid_mask=None)
    L_comp = comp(pred, target, valid_mask=None)

    expected = comp.base_weight * L_base.detach()
    assert torch.isclose(L_comp, expected, atol=1e-6)


def test_composite_returns_scalar_and_dtype():
    """Ensure the wrapper returns a scalar tensor on CPU and has finite value."""
    B, C, D, H, W = 1, 1, 3, 6, 6
    pred = torch.randn((B, C, D, H, W), dtype=torch.float32)
    target = torch.randn((B, C, D, H, W), dtype=torch.float32)
    valid_mask = torch.ones_like(pred)

    base = SSIMMSELoss3D(alpha=0.2)
    comp = CompositeClusterAwareLoss(base_criterion=base, kernel_size=5, eps=1e-6)

    L = comp(pred, target, valid_mask=valid_mask)
    assert isinstance(L, torch.Tensor) and L.ndim == 0
    assert torch.isfinite(L)


def test_incompatible_base_criterion_raises_type_error():
    """If the wrapped base criterion does not accept a ``valid_mask`` kwarg
    the wrapper should propagate a TypeError when called.
    """
    B, C, D, H, W = 1, 1, 3, 4, 4
    pred = torch.zeros((B, C, D, H, W), dtype=torch.float32)
    target = torch.zeros((B, C, D, H, W), dtype=torch.float32)
    valid_mask = torch.ones_like(pred)

    base = nn.MSELoss()
    comp = CompositeClusterAwareLoss(base_criterion=base, kernel_size=5, eps=1e-6)

    with pytest.raises(TypeError):
        _ = comp(pred, target, valid_mask=valid_mask)
