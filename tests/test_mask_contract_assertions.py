import pytest
import torch
from torch import nn

from train import _compute_masked_loss


def test_mask_contract_allows_valid_shapes_and_bool_mask():
    output = torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32)
    target = torch.ones_like(output)
    mask = torch.ones_like(output, dtype=torch.bool)
    mask[:, :, :, 0, 0] = False

    loss = _compute_masked_loss(nn.MSELoss(), output, target, mask)

    assert torch.isfinite(loss)


def test_mask_contract_rejects_output_target_shape_mismatch():
    output = torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32)
    target = torch.ones((1, 1, 2, 2, 3), dtype=torch.float32)
    mask = torch.ones_like(output, dtype=torch.bool)

    with pytest.raises(ValueError, match="output/target shape mismatch"):
        _compute_masked_loss(nn.MSELoss(), output, target, mask)


def test_mask_contract_rejects_mask_shape_mismatch():
    output = torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32)
    target = torch.ones_like(output)
    mask = torch.ones((1, 1, 2, 2, 3), dtype=torch.bool)

    with pytest.raises(ValueError, match="mask/output shape mismatch"):
        _compute_masked_loss(nn.MSELoss(), output, target, mask)


def test_mask_contract_rejects_non_bool_mask_dtype():
    output = torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32)
    target = torch.ones_like(output)
    mask = torch.ones_like(output, dtype=torch.float32)

    with pytest.raises(TypeError, match="mask must have dtype torch.bool"):
        _compute_masked_loss(nn.MSELoss(), output, target, mask)
