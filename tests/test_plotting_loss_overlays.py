import torch

from synthoseis_pre_train.losses import SSIMMSELoss3D, CompositeClusterAwareLoss
from synthoseis_pre_train.plotting import make_4panel_figure


def test_cluster_loss_diagnostic_weight_maps_follow_masks():
    pred = torch.zeros((1, 1, 3, 7, 7), dtype=torch.float32)
    valid_mask = torch.ones_like(pred)
    valid_mask[:, :, :, 3, 3] = 0.0

    criterion = CompositeClusterAwareLoss(
        base_criterion=SSIMMSELoss3D(alpha=0.0),
        kernel_size=3,
        eps=1e-6,
        base_weight=0.25,
        cluster_weight=0.75,
    )

    base_map, cluster_map = criterion.diagnostic_weight_maps(pred, valid_mask=valid_mask)

    assert base_map.shape == pred.shape
    assert cluster_map.shape == pred.shape
    assert torch.all(base_map[valid_mask == 0] == 0)
    assert torch.all(base_map[valid_mask > 0] == torch.tensor(0.25))
    assert cluster_map.max().item() == 0.75
    assert cluster_map[:, :, :, 3, 3].min().item() == 0.75


def test_make_4panel_figure_accepts_weight_overlays():
    inp = torch.zeros((1, 8, 8, 8), dtype=torch.float32)
    out = torch.ones((1, 8, 8, 8), dtype=torch.float32)
    lbl = torch.ones((1, 8, 8, 8), dtype=torch.float32)
    base_weight = torch.zeros((1, 8, 8, 8), dtype=torch.float32)
    cluster_weight = torch.zeros((1, 8, 8, 8), dtype=torch.float32)
    base_weight[:, :, 4, :] = 0.25
    cluster_weight[:, :, :, 4] = 0.75

    fig = make_4panel_figure(
        inp,
        out,
        lbl,
        "overlay smoke test",
        base_weight_vol=base_weight,
        cluster_weight_vol=cluster_weight,
    )

    assert fig is not None
    assert len(fig.axes) >= 6
