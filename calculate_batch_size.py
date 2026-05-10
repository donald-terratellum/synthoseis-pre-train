#!/usr/bin/env python3
"""Estimate a safe batch size using the same model as train.py."""

import argparse
import torch
import numpy as np

from synthoseis_pre_train.gpu_utils import get_default_device, get_memory_info
from synthoseis_pre_train.models import create_model


def estimate_safe_batch_size(sample_shape: tuple[int, int, int], device: torch.device, memory_info: dict) -> dict:
    """Mirror train.py's memory estimate and safe batch-size clamp."""
    model = create_model(
        input_channels=1,
        hidden_dims=(32, 64, 128, 256),
        spatial_size=tuple(sample_shape),
    ).to(device)

    weights_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    grads_bytes = weights_bytes
    adam_bytes = 2 * weights_bytes
    fixed_bytes = weights_bytes + grads_bytes + adam_bytes

    hidden = (32, 64, 128, 256)

    def _fm(batch: int, channels: int, shape: tuple[int, int, int] | list[int]) -> int:
        return batch * channels * shape[0] * shape[1] * shape[2] * 4

    act_per_sample = 2 * (
        _fm(1, hidden[0], sample_shape)
        + _fm(1, hidden[1], [d // 2 for d in sample_shape])
        + _fm(1, hidden[2], [d // 4 for d in sample_shape])
        + _fm(1, hidden[3], [d // 8 for d in sample_shape])
        + _fm(1, hidden[2], [d // 4 for d in sample_shape])
        + _fm(1, hidden[1], [d // 2 for d in sample_shape])
        + _fm(1, hidden[0], sample_shape)
    )
    io_per_sample = 2 * int(np.prod(sample_shape)) * 4
    per_sample_var = act_per_sample + io_per_sample

    peak_factor = 2.5
    per_sample_peak = per_sample_var * peak_factor

    total_mem = memory_info["total_bytes"]
    mps_watermark = 1.172 if device.type == "mps" else 1.0
    ceiling = total_mem * mps_watermark
    other_allocs = 6 * 1024**3
    available = ceiling - other_allocs
    safe_limit = available * 0.85

    def _total(batch_size: int) -> int:
        return fixed_bytes + batch_size * per_sample_peak

    safe_max_bs = 1
    while _total(safe_max_bs + 1) < safe_limit:
        safe_max_bs += 1

    return {
        "safe_max_batch_size": safe_max_bs,
        "weights_bytes": weights_bytes,
        "grads_bytes": grads_bytes,
        "adam_bytes": adam_bytes,
        "per_sample_peak": per_sample_peak,
        "peak_factor": peak_factor,
        "ceiling": ceiling,
        "other_allocs": other_allocs,
        "available": available,
        "safe_limit": safe_limit,
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate optimal batch size for seismic training")
    parser.add_argument("--sample-shape", type=int, nargs=3, required=True,
                       help="Sample shape (depth height width)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto/cuda/mps/cpu)")
    parser.add_argument("--quiet", action="store_true",
                       help="Only output the batch size number")

    args = parser.parse_args()

    device = get_default_device(args.device)
    memory_info = get_memory_info(device)

    if not args.quiet:
        print("=== Batch Size Calculator ===")
        print(f"Device: {device}")
        print(f"Available memory: {memory_info['free_bytes'] / 1024**3:.2f} GB")
        print(f"Safety factor: {args.safety_factor}")
        print(f"Sample shape: {args.sample_shape}")
        print()

    estimate = estimate_safe_batch_size(tuple(args.sample_shape), device, memory_info)
    optimal_batch_size = int(estimate["safe_max_batch_size"])

    print(optimal_batch_size)

    if not args.quiet:
        print()
        print("Memory breakdown:")
        print(f"  Weights: {estimate['weights_bytes'] / 1024**2:.1f} MB")
        print(f"  Gradients: {estimate['grads_bytes'] / 1024**2:.1f} MB")
        print(f"  Adam states: {estimate['adam_bytes'] / 1024**2:.1f} MB")
        print(f"  Per-sample peak: {estimate['per_sample_peak'] / 1024**3:.2f} GB")
        print(f"  Peak factor: {estimate['peak_factor']:.2f}x")
        print(f"  MPS ceiling: {estimate['ceiling'] / 1024**3:.2f} GB")
        print(f"  Other allocations: {estimate['other_allocs'] / 1024**3:.2f} GB")
        print(f"  Available for model: {estimate['available'] / 1024**3:.2f} GB")
        print(f"  Safe limit: {estimate['safe_limit'] / 1024**3:.2f} GB")
        print(f"  Safe max batch size: {optimal_batch_size}")


if __name__ == "__main__":
    main()