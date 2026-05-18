"""Test that _print_keras_like_model_summary runs without error using a saved checkpoint."""

from pathlib import Path

import pytest
import torch

CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints_sliding_stats6a" / "checkpoint_epoch_0002.pt"


@pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(),
    reason=f"Checkpoint not found at {CHECKPOINT_PATH}",
)
def test_model_forward_pass(capsys):
    """Verify model loads from checkpoint and runs a forward pass before testing torchinfo."""
    from src.synthoseis_pre_train.models import create_model

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = create_model(use_checkpoint=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        # Use a small input to keep the test fast.
        x = torch.zeros(1, 1, 32, 32, 32)
        y = model(x)
    assert y.shape == x.shape, f"Unexpected output shape: {y.shape}"


@pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(),
    reason=f"Checkpoint not found at {CHECKPOINT_PATH}",
)
def test_model_summary_from_checkpoint(capsys):
    """Verify safe static model summary runs for checkpoint-loaded model."""
    from src.synthoseis_pre_train.models import create_model
    from train import _print_keras_like_model_summary

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = create_model()
    model.load_state_dict(ckpt["model"])
    model.eval()

    _print_keras_like_model_summary(
        model,
        sample_shape=(128, 128, 128),
        device=torch.device("cpu"),
    )

    captured = capsys.readouterr()
    assert "Model summary" in captured.out, f"stdout was:\n{captured.out}\nstderr:\n{captured.err}"
    assert "Total params" in captured.out or "Parameters" in captured.out or "params" in captured.out.lower(), \
        f"No param count found. stdout was:\n{captured.out}\nstderr:\n{captured.err}"


@pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(),
    reason=f"Checkpoint not found at {CHECKPOINT_PATH}",
)
def test_model_summary_full_from_checkpoint(capsys):
    """Verify full debug summary path is callable and emits warning banner."""
    from src.synthoseis_pre_train.models import create_model
    from train import _print_keras_like_model_summary_full

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = create_model()
    model.load_state_dict(ckpt["model"])
    model.eval()

    _print_keras_like_model_summary_full(
        model,
        sample_shape=(32, 32, 32),
        device=torch.device("cpu"),
        show_trainable=True,
    )

    captured = capsys.readouterr()
    assert "Model summary FULL" in captured.out, f"stdout was:\n{captured.out}\nstderr:\n{captured.err}"
    assert "WARNING:" in captured.out, f"stdout was:\n{captured.out}\nstderr:\n{captured.err}"
