"""Regression checks for kernel-size scheduling defaults."""

from synthoseis_pre_train.models import ResBlock3d, create_model


def test_default_model_keeps_legacy_3x3_residual_kernels():
    """Default constructor should match pre-change 3x3 residual kernels."""
    model = create_model(use_checkpoint=False)

    res_blocks = [m for m in model.modules() if isinstance(m, ResBlock3d)]
    assert res_blocks, "Expected ResBlock3d modules in the default model"

    for block in res_blocks:
        assert block.conv1.kernel_size == (3, 3, 3)
        assert block.conv2.kernel_size == (3, 3, 3)


def test_custom_schedule_changes_shallow_kernels_but_not_bottleneck():
    """Custom schedule should apply per-stage kernels while bottleneck remains 3."""
    model = create_model(
        hidden_dims=(16, 32, 64, 128),
        kernel_sizes=(7, 5, 3, 3),
        use_checkpoint=False,
    )

    assert model.encoder.stem.conv1.kernel_size == (7, 7, 7)
    assert model.encoder.stem.conv2.kernel_size == (7, 7, 7)

    assert model.encoder.enc_blocks[0].conv1.kernel_size == (5, 5, 5)
    assert model.encoder.enc_blocks[1].conv1.kernel_size == (3, 3, 3)
    assert model.encoder.enc_blocks[2].conv1.kernel_size == (3, 3, 3)

    assert model.encoder.bottleneck.conv1.kernel_size == (3, 3, 3)
    assert model.encoder.bottleneck.conv2.kernel_size == (3, 3, 3)

    # Decoder mirrors encoder scales from deep->shallow: (3, 5, 7)
    assert model.decoder.dec_blocks[0].conv1.kernel_size == (3, 3, 3)
    assert model.decoder.dec_blocks[1].conv1.kernel_size == (5, 5, 5)
    assert model.decoder.dec_blocks[2].conv1.kernel_size == (7, 7, 7)
