# Session Summary: Epoch-Boundary Dataset Management and Resilience Hardening

**Date:** 2026-05-05
**Scope:** training orchestration, dataset lifecycle, thermal monitoring, shell wrapper ergonomics
**Outcome:** Moved all dataset pruning and discovery to epoch boundaries (no mid-epoch zarr access), eliminated a silent training crash caused by a broken reload path, cut powermetrics subprocess overhead from every 10 batches to once-per-run, and cleaned up all now-redundant mid-epoch fallback branches.

---

## Highlights

- **Epoch-boundary dataset management**: pruning and discovery now happen only at the top of each epoch loop. Active train/val lists are fixed for the duration of each epoch — no mid-epoch zarr disappearance risk.
- **Fixed silent crash**: outer `while` loop was breaking silently when a batch exception left `reload_requested=False`; any batch failure now always sets `reload_requested=True` before breaking.
- **Powermetrics overhead eliminated**: added `_POWERMETRICS_UNAVAILABLE` module sentinel in `gpu_utils.py`; after the first subprocess failure, all subsequent `get_cpu_temperature_c()` and `get_thermal_pressure_level()` calls return `None` immediately.
- **MallocStackLogging suppression**: changed from `unset` to explicit `export VAR=0` in all shell scripts and in the Python Darwin guard; avoids warnings in subprocess forks.
- **Generator append-only by default**: `generate_datasets.sh` no longer deletes the oldest dataset after each run; `--replace-oldest` is an explicit opt-in.
- **Iterator startup timing**: printed to stdout so slow first-batch windows are immediately diagnosable.
- **Cleanup**: removed `_prune_unavailable_train_subdatasets()` inner function and all mid-epoch `should_reload_fn`/`refresh_every_batches` plumbing from `train_epoch`. Calling site no longer passes those kwargs.

---

## Files Changed

| File | Change Summary |
|---|---|
| `train.py` | `_prune_oldest_to_target()` helper; `split_target_train/val` fixed at run start; epoch loop runs prune+discovery at boundary; iterator startup timing printed; removed `_prune_unavailable_train_subdatasets`, `should_reload_fn`, `refresh_every_batches` params; Darwin guard sets `Malloc*=0` |
| `src/synthoseis_pre_train/gpu_utils.py` | `_POWERMETRICS_UNAVAILABLE` sentinel; `_CLEAN_ENV` sets `MallocStackLogging=0`; thermal calls short-circuit after first subprocess failure |
| `src/synthoseis_pre_train/dataloader.py` | `SeismicDataset.__len__` skips missing cube keys; prunes `available_cubes` |
| `train_multi_datasets.sh` | Darwin guard uses `export VAR=0`; `--refresh-every-batches` noted deprecated in help; passes new `--train/val_batches_per_epoch`, `--val_split_ratio` |
| `generate_datasets.sh` | Default is now append-only; `--replace-oldest` is opt-in |
| `run_smoke_test.sh` | Darwin guard added (set to 0 not unset) |
| `calculate_batch_size.py` | Unified estimator aligned with train.py |
| `README.md` | `--batch-size NUM\|auto` added to key options table |
| `.gitignore` | Added `*.pt` and `events.out.tfevents.*` |

---

## Epoch-Boundary Dataset Lifecycle

### Old behaviour

- `_update_split()` was called mid-epoch to pick up new datasets.
- `_prune_unavailable_train_subdatasets()` mutated the live `ConcatDataset` in-place on every batch exception.
- `reload_requested` was only set on some code paths, so the outer `while` loop silently broke on others.

### New behaviour

```
epoch start
  ├── _prune_oldest_to_target()   # delete oldest complete datasets on disk; keep newest N
  ├── _update_split()             # assign any newly-appeared datasets to train or val
  └── build active_train / active_val   # fixed for this epoch

epoch loop (train_epoch)
  └── batch exception → reload_requested=True, break
      └── outer while rebuilds loader from epoch-fixed list, retries remaining batches
```

`split_target_train` and `split_target_val` are fixed once at run start from `--num_train`/`--num_val` and never recalculated mid-run.

---

## Powermetrics Sentinel Design

- Module-level `_POWERMETRICS_UNAVAILABLE: bool = False` in `gpu_utils.py`.
- On first failed subprocess attempt across all sampler variants, flag is set to `True`.
- All subsequent calls to `get_cpu_temperature_c()` and `get_thermal_pressure_level()` return `None` immediately — zero subprocesses spawned.
- Cost: one round of subprocess attempts at training start (order of seconds), then no overhead for the remainder of the run.

---

## Why This Matters

- **17.9-minute first batch window** was caused by 12 subprocess fork+exec calls per 10-batch thermal check cycle, all failing.  With the sentinel, the cost is paid once.
- **Silent crash after zarr deletion** is eliminated; the outer `while` loop always rebuilds a fresh loader when a batch fails, and the epoch-boundary design means zarr folders should never disappear while they are in the active list.
