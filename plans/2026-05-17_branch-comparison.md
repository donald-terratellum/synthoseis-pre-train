# Branch Comparison Report — 2026-05-17

## Branches

| Branch | Last Commit SHA | Message | Date | Remote? |
|--------|----------------|---------|------|---------|
| `main` (local) | `aa4dc40` | chore: align masking/dataloader and training scripts; packaging and launcher fixes | ~2026-05 | Yes (1 commit ahead of origin) |
| `main` (origin) | `746b256` | chore: docs and fixes — session summaries, cluster-aware loss, and launcher/train updates | 2026-05-10 | Yes |
| `feat/unet-decoder-fix` ← **current** | `84e1127` | feat: study-aligned loss loop, per-example logging, SSIM impl fix, histeq defaults | 2026-05-17 | Yes (in sync with origin) |
| `feat/mixed-batch-concatdataset` | `5c9ac9d` | feat: finalize epoch-boundary dataset flow and train loop cleanup | — | **No (local only)** |

---

## Branch Summaries

### `main` (origin/main) — `746b256`

The initial foundation commit for this repo's core infrastructure. All changes in this commit are **additions** (1,156 lines added, 0 deleted).

**What was introduced:**
- `calculate_batch_size.py` — standalone script to estimate safe batch size from device memory
- `run_smoke_test.sh` — one-epoch GPU smoke test launcher
- `src/synthoseis_pre_train/losses.py` (initial version) — `SSIMMSELoss3D` (zero-mean, seismic-specialized 3D SSIM-MSE) and `CompositeClusterAwareLoss` (upweights traces near masked cluster regions)
- `diagnostics/diagnostic_plot_cluster_masking_methods.py` — 3-panel PNG comparing uniform/Mitchell/Poisson-disc cluster center methods
- `tests/test_losses.py` — unit tests for `SSIMMSELoss3D` (perfect match → near-zero loss, masking respected, alpha blending)
- `tests/test_loss_cluster_aware.py` / `test_loss_cluster_aware_extra.py` — tests for `CompositeClusterAwareLoss`
- `tests/test_masking_sampling.py` (initial 213 lines) — tests for `create_mask_3d` center selection methods

**State of `train.py`/core modules at this point:** Prior to the decoder-fix and study-alignment work.

---

### `main` (local) — `aa4dc40`

One commit ahead of `origin/main`. Not yet pushed to remote.

**What was introduced:**
- Alignment of masking/dataloader with training scripts
- Packaging and launcher fixes
- *(Full diff not available without running locally — this commit exists only on Mac mini)*

---

### `feat/unet-decoder-fix` (current) — `84e1127`

The most active branch. 2,814 total changes (2,376 additions, 438 deletions) across 15+ files.

**Key changes by file:**

#### `train.py` — major rewrite of the training loop
- Added `_compute_study_scaled_losses()` — returns `(mse_loss, huber_loss×10, ssim_loss×200)` matching the study script conventions
- Added `_select_loss_by_type()` — routes to the correct scalar for backprop based on `--loss_type`
- Explicit `mse_fn`, `huber_fn`, `ssim_fn` module instantiation; `--ssim-implementation` flag now respected for SSIM path
- Per-example stdout logging: `mse`, `huber`, `ssim(x200)`, input/output/target amplitude stats after every batch
- `_compute_mask_breakdown()` — decomposes masked voxels into empty-z / cluster-trace / inter-peak categories
- `_compute_bounds_based_overlays()` + `_dump_overlay_debug()` — bounds-based QC weight maps for TensorBoard
- `ProcessTreeCsvMonitor` integration in `gpu_utils` used via `train.py`
- Amplitude statistics accumulation (`_new_amplitude_stats`, `_accumulate_amplitude_stats`, `_finalize_amplitude_stats`) reported per epoch
- Timing breakdowns: data-wait vs compute time per example
- `validate()` and `train_epoch()` extended with `return_details=True` returning dicts instead of bare floats
- `--model-arch` flag (`unet` vs `dynunet`), `--block_type` flag, `--pre_head_mode`, `--amplitude_transform`, quantile-normal flags

#### `src/synthoseis_pre_train/losses.py` — SSIM rescaling and DSSIM
- `_rescale_zero_centered_to_unit()` — maps zero-centered seismic amplitudes to `[0,1]` before SSIM
- Both `SSIMMSELoss3D` and `MONAIStyleSSIMMSELoss3D` now rescale inputs internally
- SSIM component changed to DSSIM formula `0.5 * (1 - ssim_score)` for both implementations
- `MONAIStyleSSIMMSELoss3D` class added

#### `src/synthoseis_pre_train/models.py` — anisotropic blocks and decoder fix
- `AnisotropicResBlock3d` — separates vertical (z) and lateral (XY) context with factored convolutions
- `_build_residual_block()` factory selects `resblock` vs `anisotropic` based on `block_type`
- `pre_head_norm` applied before the reconstruction head (`self.head(self.pre_head_norm(x))`)
- `create_model()` / `create_static_model()` expose `block_type` parameter

#### `src/synthoseis_pre_train/masking.py` — prefilter-only extrema
- `_generate_triangular_kernel()` and `_prefilter_along_z()` — triangular prefilter before extrema search
- Prefilter-only extrema detection (no tie-break); configurable `extrema_prefilter_kernel_length=19`, `extrema_prefilter_power=1.6`

#### `src/synthoseis_pre_train/gpu_utils.py` — process monitoring
- `ProcessTreeCsvMonitor` — background thread that writes CPU/GPU/MPS/thermal metrics to CSV at configurable intervals
- `get_mps_allocated_gb()` — MPS memory (current + driver)
- `_collect_process_tree_usage()` — walks the process tree for accurate multi-process CPU+RSS stats
- Multiple `powermetrics` command variants tried for robust GPU percent on macOS

#### `src/synthoseis_pre_train/plotting.py` — overlay and amplitude range
- `make_4panel_figure()` extended with `base_weight_vol`, `cluster_weight_vol`, `fixed_amplitude_range` parameters
- `make_crosssection_figure()` extended with `fixed_amplitude_range`
- Overlay annotations (text labels on cross-section panels)

#### `src/synthoseis_pre_train/dataloader.py` — transform pipeline
- `QuantileNormalTransform` integration (`derive_quantile_normal_transform`, `load_or_derive_quantile_normal_transform`)
- `HistEqParams` loading/derivation with fallback to in-memory derive for read-only zarr stores
- `mask_fill_method` (`zero` vs `gaussian`) and `mask_noise_std` kwargs
- Dataset prefix logging (`run_XXXX` extraction from zarr path)
- `enable_cluster_mask_expansion` parameter

#### `generate_datasets.sh`
- `wait_for_dataset_capacity()` — pauses dataset generation when ≥14 seismic datasets exist in the folder (30-min polling loop)

#### `train_multi_datasets.sh`
- `set -Eeuo pipefail` + `DEBUG`/`ERR`/`EXIT` traps for explicit failure diagnostics in redirected logs
- `--ssim-implementation`, `--mask-fill-method`, `--mask-noise-std`, `--amplitude-transform`, quantile flags, `--block-type`, `--pre-head-mode`, `--model-arch`, `--arch-preset` all wired up
- `ENABLE_CLUSTER_LOSS_FLAG` variable to avoid `${var:+...}` brace expansion issues

#### `inference.py`
- `_dataset_prefix_from_zarr_path()` — extracts `run_####` prefix from zarr path for output labeling

#### `pyproject.toml`
- Added `google-crc32c>=1.5.0` dependency (zarr CRC32C support)

#### `tests/test_masking_sampling.py`
- Extended with 43 additional lines (new test cases for prefilter and extrema behavior)

---

### `feat/mixed-batch-concatdataset` — `5c9ac9d`

**Local only — not pushed to GitHub remote.**

Commit message: *"feat: finalize epoch-boundary dataset flow and train loop cleanup"*

This branch focused on multi-dataset batching using `ConcatDataset` and epoch-boundary dataset discovery/pruning. The full diff is not available without running `git diff main...feat/mixed-batch-concatdataset` locally, but the commit message suggests:
- Epoch-boundary dataset refresh (replacing the deprecated batch-level `--refresh-every-batches`)
- `ConcatDataset` integration for mixing multiple seismic zarr stores in one dataloader
- Training loop cleanup to handle variable dataset counts across epochs

---

## Switching Branches Safely

**Yes, you can switch branches without losing changes**, but with caveats:

| Your current state | Safe to `git checkout`? | Recommended action |
|---|---|---|
| Uncommitted changes that **don't conflict** with the target branch | ✅ Git will carry them over | Switch directly — changes stay in working tree |
| Uncommitted changes that **conflict** with the target branch | ❌ Git will refuse | Either commit or stash first |
| Clean working tree | ✅ Always safe | Switch freely |

**Safest approach for your current state** (15 modified files):

```bash
# Option A: Commit first (recommended)
git -C ~/synthoseis-pre-train add -u
git -C ~/synthoseis-pre-train commit -m "wip: save state before branch switch"

# Option B: Stash
git -C ~/synthoseis-pre-train stash push -m "working state 2026-05-17"
git -C ~/synthoseis-pre-train checkout feat/mixed-batch-concatdataset
# ... do your work ...
git -C ~/synthoseis-pre-train checkout feat/unet-decoder-fix
git -C ~/synthoseis-pre-train stash pop
```

---

## Divergence Summary

```
origin/main (746b256) ─── local main (aa4dc40) [+1 unpushed]
      │
      └── feat/unet-decoder-fix (84e1127) ← current [in sync with remote]

local only:
      └── feat/mixed-batch-concatdataset (5c9ac9d) [never pushed]
```

`feat/unet-decoder-fix` contains the most recent and complete work. `main` is behind by all the training loop, SSIM alignment, anisotropic block, and process monitoring changes.
