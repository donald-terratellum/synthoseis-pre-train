import torch

from synthoseis_pre_train.models import AnisotropicResBlock3d, ResBlock3d, create_model


def test_model_uses_resblock_by_default():
    model = create_model(hidden_dims=(8, 16, 32, 64), use_checkpoint=False)
    assert isinstance(model.encoder.stem, ResBlock3d)


def test_model_uses_anisotropic_block_when_requested():
    model = create_model(hidden_dims=(8, 16, 32, 64), block_type="anisotropic", use_checkpoint=False)
    assert isinstance(model.encoder.stem, AnisotropicResBlock3d)


@torch.no_grad()
def test_anisotropic_model_forward_shape_matches_input():
    model = create_model(hidden_dims=(8, 16, 32, 64), block_type="anisotropic", use_checkpoint=False)
    x = torch.randn(1, 1, 32, 32, 32)
    y = model(x)
    assert y.shape == x.shape
