import torch
import torch.nn as nn
import numpy as np
from src.synthoseis_pre_train.models import create_model


def test_no_conv_transpose_in_decoder():
    model = create_model(hidden_dims=(8, 16, 32, 64), use_checkpoint=False)
    for m in model.decoder.modules():
        assert not isinstance(m, nn.ConvTranspose3d), "ConvTranspose3d found in decoder"


def test_skip_alignment_shapes():
    model = create_model(hidden_dims=(8, 16, 32, 64), use_checkpoint=False)
    model.eval()
    # small spatial size to keep test fast
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        bottleneck, skips = model.encoder(x)
        # run the decoder upsample modules stepwise and assert spatial matches
        cur = bottleneck
        for up, skip in zip(model.decoder.upsamples, reversed(skips)):
            up_out = up(cur)
            assert up_out.shape[2:] == skip.shape[2:], f"Upsampled shape {up_out.shape} != skip {skip.shape}"
            # simulate concatenation for next block input
            cur = torch.cat([up_out, skip], dim=1)
            # pass through a ResBlock to ensure block accepts shape
            # use the decoder block corresponding to this stage
        # Run full decoder to ensure final forward pass succeeds
        out = model.decoder(bottleneck, skips)
        assert out.shape[2:] == x.shape[2:]


def test_decoder_on_random_tensors():
    model = create_model(hidden_dims=(8, 16, 32, 64), use_checkpoint=False)
    model.eval()
    bottleneck = torch.randn(1, 64, 4, 4, 4)
    # create matching skips for sizes [8@32,16@16,32@8]
    skips = [torch.randn(1, 8, 32, 32, 32), torch.randn(1, 16, 16, 16, 16), torch.randn(1, 32, 8, 8, 8)]
    with torch.no_grad():
        out = model.decoder(bottleneck, skips)
    assert torch.isfinite(out).all()
