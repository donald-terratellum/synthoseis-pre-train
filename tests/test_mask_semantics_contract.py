import numpy as np
import torch
from torch import nn

from train import _compute_masked_loss
from synthoseis_pre_train.masking import apply_mask_to_seismic


class _CaptureMaskCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.valid_mask = None

    def forward(self, output, target, valid_mask=None):
        self.valid_mask = valid_mask
        return ((output - target) ** 2 * valid_mask).mean()


def test_apply_mask_to_seismic_true_preserve_false_masked():
    seismic = np.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ], dtype=np.float32)
    mask = np.array([
        [[True, False], [True, False]],
        [[False, True], [True, True]],
    ], dtype=bool)

    masked, original, out_mask = apply_mask_to_seismic(seismic, mask, fill_method="zero")

    assert np.array_equal(original, seismic)
    assert np.array_equal(out_mask, mask)
    # False entries must be replaced by zeros in masked output.
    assert np.all(masked[~mask] == 0.0)
    # True entries must preserve original values.
    assert np.array_equal(masked[mask], seismic[mask])


def test_train_boundary_inverts_mask_to_valid_mask():
    output = torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32)
    target = torch.ones_like(output)

    # Contract used in train: mask=True means visible; supervised voxels are ~mask.
    mask = torch.ones_like(output, dtype=torch.bool)
    mask[:, :, :, 0, 0] = False

    criterion = _CaptureMaskCriterion()
    loss = _compute_masked_loss(criterion, output, target, mask)

    assert torch.isfinite(loss)
    expected_valid_mask = (~mask).to(dtype=output.dtype)
    assert criterion.valid_mask is not None
    assert torch.equal(criterion.valid_mask, expected_valid_mask)
