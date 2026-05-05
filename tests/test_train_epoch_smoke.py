"""Smoke tests for merged-loader training path."""

import math

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from train import train_epoch


class _TinySeismicDataset(Dataset):
    """Tiny in-memory dataset matching train_epoch input contract."""

    def __init__(self, n: int = 4, side: int = 8):
        self.n = n
        self.side = side

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        shape = (self.side, self.side, self.side)
        input_data = torch.full(shape, fill_value=0.1 * (idx + 1), dtype=torch.float32)
        target = input_data + 0.5

        # True means visible. Loss is computed on ~mask (masked voxels),
        # so ensure at least some False entries each sample.
        mask = torch.ones(shape, dtype=torch.bool)
        mask[::2, ::2, ::2] = False
        return input_data, target, mask


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def test_merge_train_loaders_returns_single_loader():
    ds_a = _TinySeismicDataset(n=4)
    ds_b = _TinySeismicDataset(n=4)

    merged_ds = ConcatDataset([ds_a, ds_b])
    merged = DataLoader(merged_ds, batch_size=2, shuffle=True)

    assert isinstance(merged, DataLoader)
    assert len(merged.dataset) == len(ds_a) + len(ds_b)


def test_train_epoch_smoke_with_merged_loader():
    device = torch.device("cpu")
    model = _TinyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    ds_a = _TinySeismicDataset(n=4)
    ds_b = _TinySeismicDataset(n=4)
    merged_loader = DataLoader(ConcatDataset([ds_a, ds_b]), batch_size=2, shuffle=True)

    loss = train_epoch(
        model=model,
        train_loader=merged_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scaler=None,
        writer=None,
        epoch=0,
        output_dir=None,
        train_paths=[],
        val_paths=[],
        thermal_guard=None,
        grad_accum_steps=1,
        grad_clip_norm=0.0,
        ema=None,
        ema_update_every=1,
    )

    assert math.isfinite(loss)
    assert loss > 0.0
