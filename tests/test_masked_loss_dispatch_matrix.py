import pytest
import torch
from torch import nn

from train import _compute_masked_loss


class _ValidMaskCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_valid_mask = None

    def forward(self, output, target, valid_mask=None):
        self.last_valid_mask = valid_mask
        return ((output - target) ** 2 * valid_mask).mean()


class _OptInFallbackCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.allow_mask_indexing = True
        self.last_output_shape = None

    def forward(self, output, target):
        self.last_output_shape = tuple(output.shape)
        return (output - target).abs().mean()


class _StructuralNoMaskCriterion(nn.Module):
    def forward(self, output, target):
        # No valid_mask argument and no explicit fallback opt-in.
        return ((output - target) ** 2).mean()


def _sample_tensors():
    output = torch.arange(18, dtype=torch.float32).view(1, 1, 2, 3, 3)
    target = torch.zeros_like(output)
    mask = torch.ones_like(output, dtype=torch.bool)
    mask[:, :, :, 0, 0] = False
    mask[:, :, :, 1, 1] = False
    mask[:, :, :, 2, 2] = False
    return output, target, mask


def test_dispatch_matrix_prefers_valid_mask_signature():
    output, target, mask = _sample_tensors()

    criterion = _ValidMaskCriterion()
    loss = _compute_masked_loss(criterion, output, target, mask)

    assert torch.isfinite(loss)
    assert criterion.last_valid_mask is not None
    assert torch.equal(criterion.last_valid_mask, (~mask).to(dtype=output.dtype))


@pytest.mark.parametrize(
    "criterion",
    [
        nn.MSELoss(),
        nn.L1Loss(),
        nn.HuberLoss(delta=1.0),
        nn.SmoothL1Loss(beta=1.0),
    ],
)
def test_dispatch_matrix_allows_pointwise_fallback(criterion):
    output, target, mask = _sample_tensors()

    got = _compute_masked_loss(criterion, output, target, mask)
    expected = criterion(output[~mask], target[~mask])

    assert torch.isfinite(got)
    assert torch.allclose(got, expected)


def test_dispatch_matrix_allows_explicit_opt_in_fallback():
    output, target, mask = _sample_tensors()

    criterion = _OptInFallbackCriterion()
    loss = _compute_masked_loss(criterion, output, target, mask)

    assert torch.isfinite(loss)
    assert criterion.last_output_shape == (int((~mask).sum().item()),)


def test_dispatch_matrix_rejects_unsupported_criterion():
    output, target, mask = _sample_tensors()

    criterion = _StructuralNoMaskCriterion()

    with pytest.raises(ValueError, match="does not accept valid_mask"):
        _compute_masked_loss(criterion, output, target, mask)
