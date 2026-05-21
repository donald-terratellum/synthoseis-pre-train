import torch

from synthoseis_pre_train.losses import SSIMMSELoss3D, CompositeClusterAwareLoss


def test_composite_cluster_aware_mse_equivalence():
    """When base criterion is pure MSE and predictions are constant,
    composite should match the base loss value.

    SSIMMSELoss3D internally rescales amplitudes, so a hard-coded numeric
    expectation is brittle across valid implementation changes.
    """
    B, C, D, H, W = 1, 1, 3, 8, 8
    pred = torch.zeros((B, C, D, H, W), dtype=torch.float32)
    target = torch.ones((B, C, D, H, W), dtype=torch.float32)

    # Build a valid_mask with two fully-masked traces (all depth positions zero)
    valid_mask = torch.ones_like(pred)
    # Zero out two traces across depth at positions (2,2) and (5,5)
    valid_mask[:, :, :, 2, 2] = 0.0
    valid_mask[:, :, :, 5, 5] = 0.0

    base = SSIMMSELoss3D(alpha=0.0)  # pure MSE behavior
    comp = CompositeClusterAwareLoss(base_criterion=base, kernel_size=5, eps=1e-6)

    L = comp(pred, target, valid_mask=valid_mask)
    expected = base(pred, target, valid_mask=valid_mask)

    # With constant per-voxel error, weighted base/cluster terms should reduce
    # to the same scalar as the base criterion under the same valid mask.
    assert torch.isclose(L, expected, atol=1e-6)
