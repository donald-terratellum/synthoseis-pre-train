#!/usr/bin/env python3
"""Query GPU resources and print a summary for the current machine."""

from synthoseis_pre_train.gpu_utils import get_default_device, get_memory_info, print_device_summary


def main() -> None:
    print_device_summary()
    device = get_default_device()
    info = get_memory_info(device)

    print("\n=== Device Details ===")
    print(f"Device type: {device}")
    print(f"Total memory: {info['total_bytes'] / 1024 ** 3:.2f} GB")
    if info['free_bytes'] is not None:
        print(f"Free memory: {info['free_bytes'] / 1024 ** 3:.2f} GB")

    if device.type == 'cuda':
        import torch
        print(f"CUDA device name: {torch.cuda.get_device_name(device.index or 0)}")


if __name__ == '__main__':
    main()
