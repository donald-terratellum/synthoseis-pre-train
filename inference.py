#!/usr/bin/env python3
"""Simple GPU-aware inference runner for the seismic autoencoder."""

import argparse
from pathlib import Path

import torch

from synthoseis_pre_train.gpu_utils import autocast_context, get_default_device, print_device_summary
from synthoseis_pre_train.models import Seismic3DMambaAutoencoder


def build_model(sample_shape, device):
    model = Seismic3DMambaAutoencoder(
        input_channels=1,
        hidden_dims=(32, 64, 128, 256),
        spatial_size=tuple(sample_shape),
    ).to(device)
    return model


def run_inference(model, input_shape, device, batch_size=1):
    model.eval()
    dummy_input = torch.randn((batch_size, 1, *input_shape), dtype=torch.float32, device=device)

    with torch.no_grad():
        with autocast_context(device):
            output = model(dummy_input)

    print(f"Inference completed on {device}. Output shape: {output.shape}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Run inference with the seismic autoencoder")
    parser.add_argument("--model_path", type=str, required=False,
                        help="Path to saved model weights")
    parser.add_argument("--sample_shape", type=int, nargs=3, default=[128, 128, 128],
                        help="Input sample shape for inference")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to run inference on: auto, cuda, mps, cpu")

    args = parser.parse_args()
    device = get_default_device(args.device)
    print_device_summary(args.device)

    model = build_model(args.sample_shape, device)
    if args.model_path:
        path = Path(args.model_path)
        if path.exists():
            state = torch.load(path, map_location=device)
            model.load_state_dict(state)
            print(f"Loaded weights from {path}")
        else:
            print(f"Model path not found: {path}")

    run_inference(model, args.sample_shape, device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
