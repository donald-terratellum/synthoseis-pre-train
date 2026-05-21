import torch
from torch import nn

from train import _compute_masked_loss


class _MaskAwareCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_valid_mask = None
        self.last_output_shape = None
        self.last_target_shape = None

    def forward(self, output, target, valid_mask=None):
        self.last_output_shape = tuple(output.shape)
        self.last_target_shape = tuple(target.shape)
        self.last_valid_mask = valid_mask
        return ((output - target) ** 2 * valid_mask).mean()


class _FlatCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_output_shape = None
        self.last_target_shape = None

    def forward(self, output, target):
        self.last_output_shape = tuple(output.shape)
        self.last_target_shape = tuple(target.shape)
        return ((output - target) ** 2).mean()


def test_routing_uses_valid_mask_when_supported():
    output = torch.zeros((1, 1, 2, 3, 3), dtype=torch.float32)
    target = torch.ones_like(output)

    # True=visible at dataloader boundary, so train inverts and supervises False entries.
    mask = torch.ones_like(output, dtype=torch.bool)
    mask[:, :, :, 0, 0] = False
    mask[:, :, :, 1, 1] = False

    criterion = _MaskAwareCriterion()
    loss = _compute_masked_loss(criterion, output, target, mask)

    assert torch.isfinite(loss)
    assert criterion.last_output_shape == tuple(output.shape)
    assert criterion.last_target_shape == tuple(target.shape)
    assert criterion.last_valid_mask is not None

    expected_valid_mask = (~mask).to(dtype=output.dtype)
    assert torch.equal(criterion.last_valid_mask, expected_valid_mask)


def test_routing_falls_back_to_boolean_indexing_for_pointwise_loss():
    output = torch.arange(18, dtype=torch.float32).view(1, 1, 2, 3, 3)
    target = torch.zeros_like(output)

    mask = torch.ones_like(output, dtype=torch.bool)
    mask[:, :, :, 0, 0] = False
    mask[:, :, :, 2, 2] = False

    criterion = _FlatCriterion()
    loss = _compute_masked_loss(criterion, output, target, mask)

    assert torch.isfinite(loss)
    # Fallback path uses output[~mask], target[~mask] -> flattened selected voxels.
    expected_count = int((~mask).sum().item())
    assert criterion.last_output_shape == (expected_count,)
    assert criterion.last_target_shape == (expected_count,)
