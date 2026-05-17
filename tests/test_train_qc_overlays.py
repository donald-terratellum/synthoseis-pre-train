import numpy as np

from train import _compute_bounds_based_overlays


def test_qc_overlays_use_support_mask_for_z_bounds():
    """Blue/yellow overlays must not spill into invalid deep-z region."""
    z, x, y = 16, 8, 8
    inp = np.ones((z, x, y), dtype=np.float32)

    # Simulate non-zero padded tail: amplitudes are non-zero at deep z,
    # but mask marks them invalid and overlays must ignore them.
    support = np.zeros((z, x, y), dtype=bool)
    support[:10, :, :] = True

    # Create one fully blank trace in valid support to seed cluster overlay.
    inp[:10, 3, 5] = 0.0

    base, cluster, bounds = _compute_bounds_based_overlays(inp, support_mask=support)

    # Blue should stop at z=9 (last supported index).
    assert np.all(base[10:, :, :] == 0.0)
    # Yellow should also stop at z=9.
    assert np.all(cluster[10:, :, :] == 0.0)


def test_qc_cluster_overlay_tracks_blank_trace_location():
    """Yellow overlay should include blank trace and immediate XY neighbors."""
    z, x, y = 12, 9, 9
    inp = np.ones((z, x, y), dtype=np.float32)
    support = np.ones((z, x, y), dtype=bool)

    # One blank trace at (x=4, y=6) across all z in support.
    inp[:, 4, 6] = 0.0

    _, cluster, _ = _compute_bounds_based_overlays(inp, support_mask=support)

    # Seed location must be highlighted.
    assert np.all(cluster[:, 4, 6] > 0.0)

    # 3x3 dilation neighborhood should be highlighted at center z.
    cz = z // 2
    for xx in range(3, 6):
        for yy in range(5, 8):
            assert cluster[cz, xx, yy] > 0.0

    # Far location should remain zero.
    assert np.all(cluster[:, 0, 0] == 0.0)
