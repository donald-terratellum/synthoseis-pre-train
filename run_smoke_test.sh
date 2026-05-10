#!/usr/bin/env bash

# One-epoch smoke test for seismic training using the GPU-selected device.
# Adjust DATA_PATH if needed.

set -euo pipefail

# Suppress noisy macOS allocator warnings inherited by Python subprocesses.
# Setting to "0" (not unset) is an explicit disable signal to libmalloc.
if [[ "${OSTYPE:-}" == darwin* ]]; then
  export MallocStackLogging=0
  export MallocStackLoggingNoCompact=0
fi

DATA_PATH="/Users/donaldpg/synthoseis/fake_data/seismic__2026.29456161__300ph7b1/model_data.zarr"
BATCH_SIZE=1
EPOCHS=1
SAMPLE_SHAPE="128 128 128"

echo "=== Running one-epoch GPU smoke test ==="

echo "Data path: ${DATA_PATH}"

echo "Using uv-managed environment..."
uv run python train.py \
  --data_path "${DATA_PATH}" \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --sample_shape ${SAMPLE_SHAPE} \
  --device auto

echo "=== Smoke test finished ==="
