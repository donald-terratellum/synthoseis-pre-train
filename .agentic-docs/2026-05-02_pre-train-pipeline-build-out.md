# Session Summary: Pre-training Pipeline Build-out

**Date:** 2026-05-02
**Scope:** entire `synthoseis-pre-train` repository
**Outcome:** Full masked-autoencoder pre-training pipeline assembled and debugged on an
M4 Mac mini (MPS, 24 GB unified memory); augmentation axis-scramble and extrapolation
bugs fixed; geometry test suite created (9/9 passing); multi-dataset training shell
scripts written and hardened against concurrent zarr writes; stdout buffering resolved;
training resumed from epoch 14; README overhauled with accurate instructions and
stale Mamba framing removed.

---

## Repository Overview

`synthoseis-pre-train` is a standalone PyTorch project that pre-trains a 3-D
masked autoencoder on synthetic seismic cubes produced by the
[synthoseis](https://github.com/donaldpg/synthoseis) pipeline.  The model learns
to reconstruct full seismic amplitude volumes from heavily masked inputs, making the
pre-trained encoder a strong initialisation for downstream fault/horizon segmentation.

| File / folder | Purpose |
|---|---|
| `train.py` | Main training script — single-process, MPS-first |
| `train_multi_datasets.sh` | Discovers, validates, and feeds all zarr datasets to `train.py` |
| `generate_datasets.sh` | Runs synthoseis in a loop; replaces the oldest unused dataset after each run |
| `calculate_batch_size.py` | Auto-sizes batch to fit MPS memory |
| `src/synthoseis_pre_train/models.py` | 3-D U-Net with ResBlock3d; optional U-Mamba encoder |
| `src/synthoseis_pre_train/augmentation.py` | Geometric augmentations (stretch/squeeze, time-to-depth) |
| `src/synthoseis_pre_train/dataloader.py` | `SeismicDataset` / `create_dataloader` — zarr → PyTorch |
| `src/synthoseis_pre_train/masking.py` | Peak/trough preservation + trace cluster masking |
| `src/synthoseis_pre_train/plotting.py` | TensorBoard 4-panel and cross-section figures |
| `src/synthoseis_pre_train/gpu_utils.py` | Device selection, memory accounting, AMP helpers |
| `tests/test_augmentation_geometry.py` | 9 geometric correctness tests |
| `tests/test_gpu_resources.py` | Device / memory smoke test |

---

## Hardware and Runtime Context

- **Machine:** M4 Mac mini, 24 GB unified memory, macOS 26.4.1
- **PyTorch:** 2.11.0, backend `mps`
- **Python:** 3.11.15 via `uv` venv
- **`torch.set_float32_matmul_precision("high")`** set for MPS throughput
- **Gradient checkpointing:** enabled on every `ResBlock3d` to reduce activation memory
- **Memory formula:**  
  `safe_limit = (MPS_ceiling − OTHER_ALLOCS) × 0.85`  
  where `MPS_ceiling = total_RAM × 1.172`, `OTHER_ALLOCS ≈ 6.44 GB`
- **Auto batch size** at 128³: resolves to 5 on the 24 GB machine

---

## Zarr Data Format and Axis Convention

Seismic cubes are written by synthoseis as Zarr v3 stores at:

```
/Users/donaldpg/synthoseis/fake_data/seismic__<timestamp>__<tag>/model_data.zarr
```

On-disk axis order: **(x_zarr, y_zarr, z_zarr) = (300, 300, 1499)**

After `np.transpose(raw, (2, 0, 1))` in `__getitem__`, the training tensor is
**(z, x, y) = (128, 128, 128)**.  This convention is fundamental:

| Axis index | Semantic |
|---|---|
| 0 | z — time / depth (may not be flipped/swapped) |
| 1 | x_zarr — spatial (may be flipped; may be swapped with axis 2) |
| 2 | y_zarr — spatial (may be flipped; may be swapped with axis 1) |

### Active array keys

```python
DEFAULT_ARRAY_KEYS = [
    "seismicCubes_cumsum__17_degrees",
    "seismicCubes_cumsum__29_degrees",
    "seismicCubes_cumsum__5_degrees",
    "seismicCubes_cumsum_fullstack",
    "seismicCubes_cumsum_fullstack_noise_free",   # renamed from cumsum_fullstack_noise_free
]
```

Each `__getitem__` call picks one key at random so the model sees all angle
stacks within a single epoch.

### `Z_ARTIFACT_MARGIN = 25`

The last 25 z-indices of every zarr array contain known synthoseis output
artifacts.  Subvolume sampling is capped at `z < 1499 − 25 = 1474` via a
module-level constant in `dataloader.py`.

---

## Bug 1: Augmentation Axis Scramble in `stretch_squeeze_3d`

### Root cause

`stretch_squeeze_3d` documents its interface as `scale_factors = (sz, sx, sy)` to match
the data axis order (z, x, y).  Two independent mistakes broke this invariant:

1. **Inside the function** — `zoom_factors` was assembled as `(sx, sy, sz)`, so scipy
   `zoom` applied the x-scale to the z-axis, the y-scale to x, and the z-scale to y.

2. **At the call site** in `augment_pair_3d` — the function was called as
   `stretch_squeeze_3d(data, (sx, sy, sz))` instead of `(sz, sx, sy)`.

Both bugs happened to partially cancel for uniform scale factors, making them invisible
during manual inspection.  Under non-uniform scales (especially when squeezing z to
teach depth-boundary awareness), the squeeze was applied to x, leaving real z-boundary
artifacts unmasked.

### Consequence in TensorBoard

A horizontal band of 12σ voxels appeared at z ≈ 100-120 in label (y) cross-section
plots.  These were squeeze boundary artifacts that the edge mask failed to cover because
the mask was checking the wrong axis.

### Fix

```python
# stretch_squeeze_3d — inside the function
zoom_factors = (sz, sx, sy)   # was: (sx, sy, sz)

# augment_pair_3d — at the call site
clean, stretch_mask = stretch_squeeze_3d(data, (sz, sx, sy), mask_edges=True)
# was: stretch_squeeze_3d(data, (sx, sy, sz), ...)
```

### Regression test

`tests/test_augmentation_geometry.py::test_squeeze_applied_to_correct_axis`
(parametrised over all three axes).  The z case directly reproduces the pre-fix failure.

---

## Bug 2: Extrapolation Artifacts in `time_to_depth_simulation`

### Root cause

The original implementation used `fill_value='extrapolate'` on `scipy.interpolate.interp1d`.
At z-planes where the stretched sample index exceeded `[0, z_size-1]`, the extrapolator
produced runaway values rather than physically meaningful boundary values.  A
hard-coded 10% boundary exclusion margin did not match the actual extrapolation zone,
leaving some out-of-bounds planes in the label.

A second performance issue: `stretched_indices` (a z-only array) was recomputed inside
the `128 × 128 = 16 384` per-trace inner loop, wasting ~16k redundant NumPy operations
per sample.

### Fix

```python
# Compute once, outside the trace loop
stretched_indices = original_indices * stretch_factors
valid_z = (stretched_indices >= 0) & (stretched_indices <= z_size - 1)
stretched_indices_clipped = np.clip(stretched_indices, 0, z_size - 1)

# Inside the loop — use clipped indices; no extrapolation
interp_func = interp1d(original_indices, trace, kind='linear',
                       bounds_error=False,
                       fill_value=(trace[0], trace[-1]))
depth_data[:, i, j] = interp_func(stretched_indices_clipped)

# After the loop — mask the out-of-bounds z-planes
depth_mask[~valid_z, :, :] = False
```

---

## Bug 3: Spurious Extrema at Squeeze Boundaries (`augment_pair_3d`)

### Root cause

After `stretch_squeeze_3d` crops the zoomed array back to the original shape and the
edge mask marks boundary regions as invisible, the literal z=0 and z=127 planes of the
output array were non-zero (filled with boundary interpolation values by `zoom`).  The
peak/trough detector in `masking.py` fired on these planes, marking them as peaks,
producing a visible horizontal streak in the `y` label cross-section in TensorBoard.

### Fix

After `stretch_squeeze_3d` and `time_to_depth_simulation`, `augment_pair_3d` now
identifies the first and last z-planes that are outside the valid data range and zeros
them in both `clean` (label) and `combined_mask`:

```python
# Find boundary z-planes: entire x-y slice all-zero in the mask
valid_z_planes = combined_mask.any(axis=(1, 2))
z_min = int(valid_z_planes.argmax())
z_max = int(len(valid_z_planes) - valid_z_planes[::-1].argmax() - 1)
clean[:z_min, :, :] = 0.0
clean[z_max + 1:, :, :] = 0.0
combined_mask[:z_min, :, :] = False
combined_mask[z_max + 1:, :, :] = False
```

---

## Augmentation Pipeline Summary (post-fix)

```
raw (z, x, y) — 128³ subvolume extracted from zarr
  │
  ├─ stretch_squeeze_3d(data, (sz, sx, sy))  ← zoom_factors = (sz, sx, sy)
  │     → clean, stretch_mask
  │
  ├─ [60% chance] time_to_depth_simulation(clean)
  │     → clean, td_mask;  combined_mask = stretch_mask & td_mask
  │
  ├─ flip_x (axis 1), flip_y (axis 2), swap_xy — both clean + mask
  │
  ├─ boundary z-plane zeroing  ← prevents false peak detections
  │
  ├─ normalise (std → 1.0);  y = clean
  │
  └─ [optional] add noise to get x
```

---

## `train.py` Improvements

### Epoch timing and ETA

After each epoch, the script prints:

```
Epoch time: 1h 23m 11s  |  Elapsed: 3h 15m 44s  |  Remaining: ~7h 10m  |  ETA: 2026-05-03 01:22
```

Implementation: `_fmt_duration()` helper; `training_start = time.monotonic()` before
the epoch loop; `epoch_start = time.monotonic()` at the top of each epoch.

### Array keys printed per dataset

Each dataset prints its active array keys at the start of its batch loop:

```
  Dataset seismic__2026.26296054__300ph6d4, seismicCubes_cumsum__17_degrees, ... [1/10]
```

### `train_paths` / `val_paths` scope fix

`train_paths` and `val_paths` were only defined in `main()` but were referenced inside
`train_epoch()`, causing `NameError`.  Fixed by adding them as explicit parameters to
`train_epoch`.

### Dataset split deduplication

`all_paths = list(dict.fromkeys(args.data_paths))` deduplicates before the split logic.
`kept_train` and `kept_val` are also deduplicated when restoring from a checkpoint that
was saved with duplicate entries (artefact of a prior run where `VALID_PATHS` was
doubled by the shell script).

### Resilient dataloader creation

Each `create_dataloader` call is now wrapped in `try/except`.  A bad zarr path (wrong
keys, deleted between validation and use) prints a `WARNING: skipping <name>` line and
continues.  A guard after loading aborts with a clean message if no train loaders survived.

### Resilient batch iteration

```python
loader_iter = iter(loader)
for batch_idx in range(len(loader)):
    try:
        input_data, target, mask = next(loader_iter)
        ...
    except StopIteration:
        break
    except Exception as e:
        print(f"    WARNING: skipping batch {batch_idx} — {e}")
        continue
```

This handles the race condition where a zarr store is deleted by `generate_datasets.sh`
while a batch is being fetched mid-epoch.

### macOS `num_workers`

```python
import platform
_num_workers = 0 if platform.system() == "Darwin" else min(4, os.cpu_count() or 1)
```

`num_workers > 0` with zarr + MPS on macOS causes fork failures (exit code 255).

---

## `dataloader.py` Improvements

### Key-level retry in `__getitem__`

When a zarr key disappears during training, the item-fetcher now tries all other
available keys before raising:

```python
candidates = list(self.available_cubes)
random.shuffle(candidates)
cube = None
for cube_name in candidates:
    try:
        cube = self.zarr_data[cube_name][:]
        break
    except (KeyError, FileNotFoundError, OSError):
        continue
if cube is None:
    raise RuntimeError("All array keys unavailable ...")
```

### `Z_ARTIFACT_MARGIN` corrected

Was `Z_ARTIFACT_MARGIN = 2` (the original placeholder value).
Corrected to `Z_ARTIFACT_MARGIN = 25` matching the known synthoseis artifact zone.

---

## `train_multi_datasets.sh`

### Zarr data validation

The shape check was extended to verify that a zarr array has actually been written
(not just that the directory and metadata exist):

```python
signal.signal(signal.SIGALRM, _timeout)
signal.alarm(15)  # hard 15-second timeout
...
if arr.nchunks_initialized == 0:
    continue   # directory exists but no chunks written yet
```

The 15-second `SIGALRM` hard-cut prevents the script from hanging on a zarr store
that is actively being written by `generate_datasets.sh`.

### Stdout buffering fix

`uv run python train.py -u` was incorrect — `-u` was parsed as an argument to
`train.py`, not as Python's unbuffered-mode flag.  Fixed to:

```bash
uv run python -u train.py \
```

Without `-u`, Python uses 8 KB block buffering on the pipe to `tee`, so no output
appears for hours.

---

## `generate_datasets.sh`

### Purpose

Runs synthoseis in a rolling loop:

1. Execute one synthoseis run → new `seismic__<timestamp>__synthoseis_run_NNNN/`
2. Find the oldest `seismic__*` directory that is neither the new dataset nor the
   dataset currently being used for training (identified via the last 25 lines of the
   training log).
3. Delete it.

This keeps the zarr folder at a roughly constant size while continuously supplying
fresh training data.

### macOS compatibility fixes

Three GNU-ism bugs were identified and fixed:

| GNU (Linux) | BSD/macOS replacement |
|---|---|
| `find ... -printf '%T@ %f\n'` | `ls -1dtr .../seismic__*/` |
| `grep -oP '(?<=Dataset )seismic__...'` | `grep -oE 'Dataset seismic__...' \| sed 's/^Dataset //'` |
| `grep -oP '\d{4}$'` | `grep -oE '[0-9]{4}$'` |

Without these fixes `list_datasets_oldest_first()` produced no output, so `oldest`
was always empty and no dataset was ever deleted.

### Safety guard

```bash
if [[ ! "$oldest_path" =~ seismic__ ]]; then
    echo "ERROR: refusing to delete '$oldest_path'" >&2
    continue
fi
```

Never removes a directory that does not contain the pattern `seismic__`.

---

## Geometry Test Suite (`tests/test_augmentation_geometry.py`)

9 tests, all passing in ~4 s on the M4 Mac mini.

| Test | What it checks |
|---|---|
| `test_squeeze_applied_to_correct_axis[z]` | `sz < 1` shrinks only the z-extent of valid data |
| `test_squeeze_applied_to_correct_axis[x]` | `sx < 1` shrinks only the x-extent |
| `test_squeeze_applied_to_correct_axis[y]` | `sy < 1` shrinks only the y-extent |
| `test_edge_mask_covers_zero_padded_region_z` | All zero voxels are outside the mask |
| `test_no_spurious_extrema_in_y_label[seed=0]` | `|max(y)| < 8.0` after normalisation |
| `test_no_spurious_extrema_in_y_label[seed=1]` | " |
| `test_no_spurious_extrema_in_y_label[seed=7]` | " |
| `test_no_spurious_extrema_in_y_label[seed=42]` | " |
| `test_dipping_layers_topology_preserved` | Pearson r > 0.90 vs analytical dipping-layer ground truth |

Helper `make_uniform_layers(shape, n_layers, dip_x_deg, seed)` generates synthetic
constant-amplitude dipping layers for reproducible geometric testing.

---

## `seismicCubes_cumsum_fullstack_noise_free` Key Rename

The synthoseis pipeline was updated in a parallel session to rename the zarr array
key from `cumsum_fullstack_noise_free` to `seismicCubes_cumsum_fullstack_noise_free`
(consistent with all other seismic cube keys).  `DEFAULT_ARRAY_KEYS` in `train.py`
and the candidate key list in `train_multi_datasets.sh` were updated accordingly.

---

## `README.md` Overhaul

### Training and generation commands

Two new subsections were added to the Training section with the exact production
commands, working directories, and options tables:

**Multi-dataset training** (from `/Users/donaldpg/synthoseis-pre-train`):
```bash
./train_multi_datasets.sh --resume checkpoints/checkpoint_epoch_0014.pt \
    --max-samples 150 --max-epochs 35 2>&1 | tee train_multi_datasets_run_01.log
```

**Continuous dataset generation** (from `/Users/donaldpg/synthoseis/synthoseis`,
in a separate terminal running in parallel with training):
```bash
~/synthoseis-pre-train/generate_datasets.sh -n 75 \
  --synthoseis-dir ~/synthoseis/synthoseis \
  -d ~/synthoseis/fake_data \
  --start-index 8 \
  --check-log ~/synthoseis-pre-train/train_multi_datasets_run_01.log
```

### Stale Mamba framing removed

The original README described Mamba blocks as the primary architecture and included
a "Notes" section instructing readers to integrate "actual Mamba blocks" as a
future TODO.  Both were inaccurate:

- The default model is a pure 3-D U-Net with `ResBlock3d`; `MambaBlock3d` is an
  optional upgrade behind `--use_mamba` that requires CUDA + `mamba_ssm` and is
  never active on the Mac mini.
- The "placeholder" TODO was obsolete — Mamba blocks are already implemented.

Changes made:

| Location | Was | Now |
|---|---|---|
| Title | "Seismic 3D Mamba Pre-training" | "Seismic 3D Pre-training" |
| Overview bullet | "3D Mamba concepts" | "3-D U-Net with residual blocks" |
| Quick Start import | `Seismic3DMambaAutoencoder` | `create_model` |
| Quick Start import | `random_augmentation_3d` | `augment_pair_3d` |
| References | 3 Mamba repo links + "placeholder" note | Single U-Mamba footnote |

---

## Model Architecture

`create_model()` returns a `Seismic3DMambaAutoencoder` — a 3-D U-Net with four
encoder/decoder levels.  The class name is a historical artefact; the active blocks
are `ResBlock3d` unless `--use_mamba` is passed.

```
Encoder:  1 → 32 → 64 → 128 → 256  (MaxPool3d 2× between levels)
Bottleneck: 256
Decoder:  256 → 128 → 64 → 32 → 1  (ConvTranspose3d 2× between levels)
```

- **Blocks:** `ResBlock3d` (two Conv3d + InstanceNorm3d + GELU + residual projection)
- **Mamba option:** `--use_mamba` replaces encoder blocks with `MambaBlock3d`
  (depthwise CNN + Mamba SSM branches); silently falls back to `ResBlock3d` if
  `mamba_ssm` is not installed or CUDA is unavailable
- **Parameters:** 10,689,185
- **Gradient checkpointing** on every block reduces peak activation memory from ~22 GB
  to ~18 GB at batch=5 on 128³

---

## Training State (as of session end)

- Resumed from `checkpoints/checkpoint_epoch_0014.pt`
- Running epoch 15/35 across 10–11 train datasets, 4 val datasets
- 8 manually-created datasets + 7 generated by `generate_datasets.sh`
- Val datasets: `300ph7b`, `300ph7b1`, `synthoseis_run_0003`, `synthoseis_run_0001`
- Warnings during loading: `synthoseis_run_0007` skipped (only `depth_maps` key
  present — zarr was written before seismic processing completed)
