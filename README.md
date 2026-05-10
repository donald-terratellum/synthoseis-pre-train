# Seismic 3D Pre-training

Repository for pre-training a 3-D masked autoencoder on synthetic seismic data.
The model learns to reconstruct full amplitude volumes from heavily masked inputs,
making the pre-trained encoder a strong initialisation for downstream fault and
horizon segmentation tasks.

## Overview

This repository provides:
- **Masking strategies** for seismic data (peak/trough preservation, trace cluster masking)
- **Data augmentation** (independent z/xy stretch, phase rotation, time-to-depth simulation)
- **3-D U-Net model** with residual blocks and skip connections
- **Dynamic multi-dataset training pipeline** with automatic batch-size calculation and thermal pausing

## Quick Start

```python
from synthoseis_pre_train.masking import create_mask_3d, apply_mask_to_seismic, normalize_seismic
from synthoseis_pre_train.augmentation import augment_pair_3d, extract_random_subvolume
from synthoseis_pre_train.dataloader import SeismicDataset, create_dataloader
from synthoseis_pre_train.models import create_model
```

## Masking Strategy

- **Peak/Trough Preservation**: Only retains ~20% of voxels (maximas and mins along z-axis)
- **Trace Masking**: ~7% of traces masked in 3×3 clusters with 80/20 probability
- **Zero-filling**: Masked voxels set to 0

## Augmentations

- **Stretch/Squeeze**: Random scaling per axis with independent ranges (`z: 0.667-1.5`, `xy: 0.8-1.25`)
- **Phase rotation**: Constant phase shift per sample (`-180°` to `+180°`) via FFT
- **Time-to-Depth**: Simulates velocity increase with depth
- **Edge Masking**: Masks edges when squeezing to teach model edge awareness
- **Normalization**: Standard deviation = 1.0

## Environment

This repository uses `uv` for dependency and Python environment management.

```bash
uv sync
```

If Python 3.11 is not installed, install it through `uv`:

```bash
uv python install 3.11
```

Run commands inside the managed environment:

```bash
uv run python train.py --data_paths /path/to/seismic.zarr --epochs 100 --batch_size 4
uv run python tests/test_gpu_resources.py
uv run python inference.py --sample_shape 128 128 128 --batch_size 1 --device auto
```

Post-`uv sync` note (macOS / Apple Silicon)
----------------------------------------

`uv sync` recreates the project venv and may not include platform-specific
packaging (for example the `pip` binary can be bootstrapped by `uv`). For
installing PyTorch on macOS arm64 (MPS/Metal) run the helper script below or
manually bootstrap pip and then install the metal wheels index.

Manual commands:

```bash
# bootstrap pip if needed
uv run python -m ensurepip --upgrade
uv run python -m pip install -U pip setuptools wheel

# install PyTorch (Apple Silicon / metal wheels)
uv run python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/metal.html
```

Helper script: `scripts/bootstrap_pytorch.sh` (see repository)


## One-epoch GPU smoke test

A quick smoke test is available via the root wrapper script:

```bash
./run_smoke_test.sh
```

This runs a one-epoch training loop on the GPU-selected device using the synthetic dataset at:

`/Users/donaldpg/synthoseis/fake_data/seismic__2026.29456161__300ph7b1/model_data.zarr`

## Training

### Multi-dataset training on the Mac mini (recommended)

Run from `/Users/donaldpg/synthoseis-pre-train`.

Prime sudo once, then keep it alive during long runs so thermal checks can call `powermetrics` non-interactively:

```bash
sudo -k
sudo -v
while true; do sudo -n true; sleep 60; done >/dev/null 2>&1 &
SUDO_KEEPALIVE_PID=$!
trap 'kill $SUDO_KEEPALIVE_PID >/dev/null 2>&1' EXIT INT TERM
```

Launch training:

```bash
./train_multi_datasets.sh \
  --resume checkpoints/checkpoint_epoch_0005.pt \
  --max-epochs 35 \
  --val-split-ratio 0.3 \
  --batch-size 4 \
  --train-batches-per-epoch 60 \
  --val-batches-per-epoch 25 \
  --refresh-every-batches 85 \
  --grad-accum-steps 4 \
  --grad-clip-norm 1.0 \
  --ema-decay 0.999 \
  --ema-update-every 1 \
  --lr-schedule poly \
  --lr-warmup-epochs 5 \
  --lr-warmup-start-factor 0.1 \
  --lr-poly-power 0.9 \
  --lr-min 1e-6 \
  --thermal-max-c 85 \
  --thermal-cooldown-sec 300 \
  --thermal-check-every-batches 10 \
  --thermal-pressure-trip-level serious \
   2>&1 | tee -a train_multi_datasets_run_02.log
```

`train_multi_datasets.sh` discovers datasets dynamically each epoch from `--data-folder`,
ignores in-progress datasets that still have a `temp_folder__*` companion, preserves
train/val side exclusivity over time, and computes train/val active counts from
`--val-split-ratio` over currently discovered datasets. In fixed-step mode,
`--train-batches-per-epoch`, `--val-batches-per-epoch`, and `--refresh-every-batches`
keep train/validation step counts and refresh cadence stable while incorporating newly
generated datasets mid-epoch. Progress is written to both stdout and log file via `tee`.

Key options:

| Option | Default | Description |
|---|---|---|
| `--resume PATH` | — | Checkpoint to resume from |
| `--max-epochs N` | 25 | Total epochs to train |
| `--batch-size N\|auto` | `auto` | Batch size, or `auto` to probe via `calculate_batch_size.py` |
| `--data-folder PATH` | `/Users/donaldpg/synthoseis/fake_data` | Root folder of zarr datasets |
| `--val-split-ratio R` | 0.2 | Fraction of discovered datasets reserved for validation |
| `--train-batches-per-epoch N` | 120 | Fixed number of train batches per epoch |
| `--val-batches-per-epoch N` | 30 | Fixed number of validation batches per epoch |
| `--refresh-every-batches N` | 10 | Dataset refresh cadence during fixed-step training |
| `--thermal-max-c C` | 85 | Pause when CPU temperature reaches this Celsius threshold (`<=0` disables temperature threshold) |
| `--thermal-cooldown-sec SEC` | 300 | Sleep duration after thermal pause |
| `--thermal-check-every-batches N` | 10 | Thermal check cadence in training batches |
| `--thermal-pressure-trip-level LVL` | `serious` | Pressure fallback threshold: `off`, `nominal`, `fair`, `serious`, `critical` |
| `--lr-schedule MODE` | `poly` | Learning-rate schedule: `poly`, `cosine`, `constant` |
| `--lr-poly-power N` | `0.9` | Polynomial decay exponent when using `poly` |
| `--lr-min N` | `1e-6` | Floor LR for `poly`/`cosine` |
| `--lr-warmup-epochs N` | `5` | Number of warmup epochs before decay |
| `--lr-warmup-start-factor N` | `0.1` | Warmup start LR as a fraction of base LR |
| `--grad-accum-steps N` | `1` | Gradient accumulation steps (effective batch multiplier) |
| `--grad-clip-norm N` | `1.0` | Global norm clipping (`<=0` disables) |
| `--ema-decay N` | `0.999` | EMA decay (`<=0` disables EMA) |
| `--ema-update-every N` | `1` | EMA update cadence in optimizer steps |

### Thermal Checks and Automatic Pausing

Thermal guard exists to keep long MPS training stable on the Mac mini when the system
heats up and starts throttling. The guard can pause mid-epoch, save `checkpoints/thermal_latest.pt`,
cool down, and continue in the same process.

Thermal signal priority:

1. CPU temperature (when parseable from `powermetrics`)
2. Thermal pressure fallback (`Nominal/Fair/Serious/Critical`) when CPU die temperature is unavailable

Pressure-based pausing is explicit and tunable with `--thermal-pressure-trip-level`.
Typical setting is `serious`.

At startup, `train.py` prints one of:

- `Thermal monitor: available (current CPU ... )`
- `Thermal monitor: available via thermal pressure only (...)`
- `Thermal monitor: unavailable ...`

During training batch logs, thermal status appears as either `CPU temp: ...C` or
`Thermal pressure: ...`.

Batch progress now also reports elapsed wall time as true DHM:

- `Elapsed DHM: DD:HH:MM (+SS.s)`
- Example: `Elapsed DHM: 00:00:00 (+48.8s)`

### Optimisation defaults (3D medical style)

`train.py` now defaults to a 3D-medical-oriented optimisation stack:

- **LR schedule**: polynomial decay (`--lr_schedule poly`)
- **Warmup**: enabled by default (`--lr_warmup_epochs 5`, `--lr_warmup_start_factor 0.1`)
- **Gradient clipping**: enabled by default (`--grad_clip_norm 1.0`)
- **EMA**: enabled by default (`--ema_decay 0.999`, `--ema_update_every 1`)
- **Gradient accumulation**: available, default off (`--grad_accum_steps 1`)

Notes:

- `train_multi_datasets.sh` now exposes these optimizer flags directly.
- CLI names in the wrapper use hyphens (for example `--grad-accum-steps`) and are mapped to
  the matching `train.py` underscore arguments internally.

Example direct launch with larger effective batch via accumulation:

```bash
uv run python train.py \
  --data_folder /Users/donaldpg/synthoseis/fake_data \
  --epochs 100 \
  --grad_accum_steps 4
```

Additional `train.py` options:

| Option | Default | Description |
|---|---|---|
| `--lr_schedule` | `poly` | Learning-rate schedule: `poly`, `cosine`, `constant` |
| `--lr_poly_power` | `0.9` | Polynomial decay exponent when using `poly` |
| `--lr_min` | `1e-6` | Floor LR for `poly`/`cosine` |
| `--lr_warmup_epochs` | `5` | Number of warmup epochs before decay |
| `--lr_warmup_start_factor` | `0.1` | Warmup start LR as a fraction of base LR |
| `--grad_accum_steps` | `1` | Gradient accumulation steps (effective batch multiplier) |
| `--grad_clip_norm` | `1.0` | Global norm clipping (`<=0` disables) |
| `--ema_decay` | `0.999` | EMA decay (`<=0` disables EMA) |
| `--ema_update_every` | `1` | EMA update cadence in optimizer steps |

### Recommended presets

Use these presets with `train_multi_datasets.sh` as a starting point, then tune from there.

1. Conservative thermals

```bash
./train_multi_datasets.sh \
  --max-epochs 35 \
  --val-split-ratio 0.2 \
  --train-batches-per-epoch 120 \
  --val-batches-per-epoch 30 \
  --refresh-every-batches 10 \
  --grad-accum-steps 2 \
  --grad-clip-norm 1.0 \
  --ema-decay 0.999 \
  --ema-update-every 1 \
  --lr-schedule poly \
  --lr-warmup-epochs 5 \
  --lr-warmup-start-factor 0.1 \
  --lr-poly-power 0.9 \
  --lr-min 1e-6 \
  --thermal-max-c 82 \
  --thermal-cooldown-sec 360 \
  --thermal-check-every-batches 5 \
  --thermal-pressure-trip-level fair
```

2. Faster convergence

```bash
./train_multi_datasets.sh \
  --max-epochs 35 \
  --val-split-ratio 0.2 \
  --train-batches-per-epoch 120 \
  --val-batches-per-epoch 30 \
  --refresh-every-batches 10 \
  --grad-accum-steps 4 \
  --grad-clip-norm 1.0 \
  --ema-decay 0.999 \
  --ema-update-every 1 \
  --lr-schedule poly \
  --lr-warmup-epochs 5 \
  --lr-warmup-start-factor 0.1 \
  --lr-poly-power 0.9 \
  --lr-min 1e-6 \
  --thermal-max-c 85 \
  --thermal-cooldown-sec 300 \
  --thermal-check-every-batches 10 \
  --thermal-pressure-trip-level serious
```

3. Maximum stability

```bash
./train_multi_datasets.sh \
  --max-epochs 35 \
  --val-split-ratio 0.2 \
  --train-batches-per-epoch 120 \
  --val-batches-per-epoch 30 \
  --refresh-every-batches 10 \
  --grad-accum-steps 6 \
  --grad-clip-norm 0.7 \
  --ema-decay 0.9995 \
  --ema-update-every 1 \
  --lr-schedule poly \
  --lr-warmup-epochs 8 \
  --lr-warmup-start-factor 0.05 \
  --lr-poly-power 0.9 \
  --lr-min 1e-6 \
  --thermal-max-c 80 \
  --thermal-cooldown-sec 420 \
  --thermal-check-every-batches 5 \
  --thermal-pressure-trip-level fair
```

4. Overnight / unattended

Use the `--overnight` flag — it applies the Maximum stability thermal and optimizer defaults in one flag. Individual overrides still work on top of it.

```bash
./train_multi_datasets.sh \
  --overnight \
  --max-epochs 100 \
  --resume checkpoints/checkpoint_epoch_0019.pt \
  2>&1 | tee -a overnight_run.log
```

What `--overnight` pre-sets (all individually overridable):

| Setting | Overnight default | Normal default |
|---|---|---|
| `--thermal-max-c` | 80 | 85 |
| `--thermal-cooldown-sec` | 420 | 300 |
| `--thermal-check-every-batches` | 5 | 10 |
| `--thermal-pressure-trip-level` | `fair` | `serious` |
| `--grad-accum-steps` | 6 | 1 |
| `--grad-clip-norm` | 0.7 | 1.0 |
| `--lr-warmup-epochs` | 8 | 5 |
| `--lr-warmup-start-factor` | 0.05 | 0.1 |
| `--ema-decay` | 0.9995 | 0.999 |

Preset guidance:

- Start with **Faster convergence** for routine runs.
- Switch to **Conservative thermals** if frequent thermal pauses occur.
- Use **Maximum stability** or **Overnight** for very long runs, noisy data phases, or when loss spikes appear.

### Continuous dataset generation (run in parallel with training)

Run from `/Users/donaldpg/synthoseis/synthoseis` in a separate terminal:

```bash
~/synthoseis-pre-train/generate_datasets.sh -n 75 \
  --synthoseis-dir ~/synthoseis/synthoseis \
  -d ~/synthoseis/fake_data \
  --start-index 8 \
  --check-log ~/synthoseis-pre-train/train_multi_datasets_run_01.log
```

`generate_datasets.sh` runs synthoseis in a loop (`-n 75` = 75 runs).  After each
run it identifies the oldest dataset in the zarr folder that is not currently being
used for training (detected via the last 25 lines of `--check-log`) and deletes it,
keeping the folder at a roughly constant size while supplying fresh data.

`--start-index 8` sets the four-digit run tag for the first new run
(`synthoseis_run_0008`); omit it to auto-detect from existing directories.

Key options:

| Option | Default | Description |
|---|---|---|
| `-n, --num-runs N` | 1 | Number of synthoseis runs to execute |
| `--synthoseis-dir PATH` | `.` | Directory containing synthoseis `main.py` |
| `-d, --synthoseis-zarr-folder PATH` | `/Users/donaldpg/synthoseis/fake_data` | Zarr output folder |
| `--check-log FILE` | — | Training log scanned to identify the active dataset (protected from deletion) |
| `--start-index N` | auto | First run index (zero-padded to 4 digits) |
| `-c, --config PATH` | `config/example_bigger_ex.json` | Synthoseis config file |

### Single-dataset training (low-level)

```bash
uv run python train.py --data_paths /path/to/seismic.zarr --epochs 100 --batch_size 4
```

## GPU Resource Test

Use this script to detect available GPU resources and print device/memory information:

```bash
uv run python tests/test_gpu_resources.py
```

## Inference

Run a quick inference pass on the selected device with a dummy input volume:

```bash
uv run python inference.py --sample_shape 128 128 128 --batch_size 1 --device auto
```

To load pre-trained weights:

```bash
uv run python inference.py --model_path ./checkpoints/final_model.pt --sample_shape 128 128 128 --batch_size 1
```

## References

- [U-Mamba](https://github.com/ModelTC/U-Mamba) — optional SSM encoder blocks (`--use_mamba`; requires CUDA + `mamba_ssm`)
