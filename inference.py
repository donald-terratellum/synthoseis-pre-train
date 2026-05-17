#!/usr/bin/env python3
"""Simple GPU-aware inference runner for the seismic autoencoder."""

import argparse
from pathlib import Path
import re

import numpy as np

import torch

from synthoseis_pre_train.gpu_utils import autocast_context, get_default_device, print_device_summary
from synthoseis_pre_train.models import create_model
from synthoseis_pre_train.transforms import QuantileNormalConfig, load_quantile_normal_transform


def build_model(sample_shape, device):
    model = create_model(
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


def _dataset_prefix_from_zarr_path(zarr_path: str) -> str:
    path = Path(zarr_path)
    name = path.name
    if name in ("data", "data.zarr"):
        source = path.parent.name
    else:
        source = path.stem

    source = source.replace(".zarr", "")
    matches = re.findall(r"run_\d+", source)
    if matches:
        return matches[-1]

    parts = source.split("_")
    if len(parts) >= 2 and parts[-2] == "run" and parts[-1].isdigit():
        return f"run_{parts[-1]}"

    return source


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
    parser.add_argument("--zarr_path", type=str, default=None,
                        help="Optional zarr store path for loading inverse quantile transform")
    parser.add_argument("--array_key", type=str, default=None,
                        help="Array key whose transform should be used for inverse mapping")
    parser.add_argument("--apply_inverse_quantile_transform", action="store_true",
                        help="Apply inverse quantile-normal transform to model output")
    parser.add_argument("--quantile_symmetry_mode", type=str, default="strict_odd",
                        choices=["strict_odd", "independent"],
                        help="Symmetry mode used when loading quantile transform")
    parser.add_argument("--quantile_epsilon", type=float, default=1e-6,
                        help="Epsilon used by the persisted quantile transform")
    parser.add_argument("--transforms_group", type=str, default="transforms",
                        help="Transforms subgroup in zarr store")

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

    output = run_inference(model, args.sample_shape, device, batch_size=args.batch_size)

    if args.apply_inverse_quantile_transform:
        if not args.zarr_path or not args.array_key:
            raise ValueError("--apply_inverse_quantile_transform requires --zarr_path and --array_key")

        cfg = QuantileNormalConfig(
            epsilon=args.quantile_epsilon,
            symmetry_mode=args.quantile_symmetry_mode,
            transforms_group=args.transforms_group,
        )
        transform = load_quantile_normal_transform(
            data_path=args.zarr_path,
            array_key=args.array_key,
            config=cfg,
        )
        if transform is None:
            raise RuntimeError(
                f"No persisted quantile transform found for key '{args.array_key}' "
                f"in {args.zarr_path}"
            )

        out_np = output.detach().cpu().numpy().astype(np.float32)
        restored = transform.inverse(out_np)
        prefix = _dataset_prefix_from_zarr_path(args.zarr_path)
        display_key = f"{prefix}/{args.array_key}" if prefix else args.array_key
        print(f"     . Applied reverse quantile transform to {display_key}")
        print(
            "Inverse transform applied:",
            f"mean={float(np.mean(restored)):.6f}",
            f"std={float(np.std(restored)):.6f}",
        )


if __name__ == "__main__":
    main()
