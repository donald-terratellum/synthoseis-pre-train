# Plan: Mixed-Batch Training via ConcatDataset

**Date:** 2026-05-04  
**Scope:** `dataloader.py`, `train.py`, `tests/`  
**Goal:** Replace the sequential per-dataset training loop with a single shuffled
`ConcatDataset` loader so that every training batch contains examples drawn randomly
from all available seismic datasets.

---

## Context and Motivation

### Current behaviour

`train_epoch()` iterates over a list of `(name, DataLoader)` tuples:

```
for ds_idx, (ds_name, loader) in enumerate(train_loaders):   # dataset loop
    for batch_idx in range(len(loader)):                      # batch loop
        …
```

This means all gradient steps in a window come from one seismic style before the
model ever sees the next dataset.  With 4 datasets each contributing N batches,
the model sees data in blocks: A A A … B B B … C C C … D D D …

### Desired behaviour

Every mini-batch draws samples uniformly at random from the union of all datasets:

```
merged_loader = DataLoader(ConcatDataset([ds_A, ds_B, ds_C, ds_D]), shuffle=True)
for batch_idx, (input_data, target, mask) in enumerate(merged_loader):
    …
```

Gradient diversity per step improves — each optimizer update reflects all four
seismic style distributions simultaneously.

### Trade-offs considered

| Factor | Impact |
|---|---|
| Zarr I/O seek pattern | Slightly less sequential than per-dataset loops; negligible on modern NVMe |
| `num_workers=0` (macOS) | Reads already synchronous; no multi-process coordination cost |
| Checkpoint `ds_idx` field | Goes away — the concept of "current dataset index" no longer exists mid-epoch |
| TensorBoard per-dataset figures | Move to end-of-epoch; one figure per dataset from the last batch that touched it |
| Code size | Net reduction; the inner dataset loop and its bookkeeping are removed |

---

## Phases

### Phase 0 — Session housekeeping before refactor work

**Goal:** Capture current state, checkpoint existing unrelated work, and isolate
Mixed-Batch changes on a dedicated branch.

**Steps:**

1. Summarize this session's work completed since the latest `.agentic-docs`
    session summary.
2. Run `git status` and review changed files.
3. Stage and commit current completed work (non-ConcatDataset changes) with a
    clear message.
4. Create and switch to a new branch for this refactor (example name:
    `feat/mixed-batch-concatdataset`).
5. Verify branch and clean working tree before starting Phase 1.

**Code review checklist before commit:**
- [ ] Commit contains only intentional pre-refactor work
- [ ] Commit message reflects what was finalized in this session checkpoint
- [ ] New branch is created from the correct base commit
- [ ] Working tree is clean after branch switch (or only expected new plan/test files)

**Iterate until:**
- `git status` is clean or intentionally scoped
- `git branch --show-current` confirms the dedicated feature branch

**After phase completes:** update the Summary table at the bottom of this document.

### Phase 1 — `dataloader.py`: add `create_merged_dataloader`

**Goal:** A single, well-tested function that accepts a list of zarr paths and
returns one `DataLoader` backed by `ConcatDataset`.

**Responsibilities of this function (SRP):**
- Accept the same per-dataset kwargs that `create_dataloader` accepts today.
- Build one `SeismicDataset` per path.
- Combine them with `torch.utils.data.ConcatDataset`.
- Return a single shuffled `DataLoader`.
- Keep `create_dataloader` unchanged (backward compatibility; still used in tests).

**Code to add to `dataloader.py`:**

```python
from torch.utils.data import ConcatDataset, DataLoader as TorchDataLoader

def create_merged_dataloader(
    data_paths: List[str],
    batch_size: int = 4,
    sample_shape: Tuple[int, int, int] = (128, 128, 128),
    num_workers: int = 0,
    pin_memory: bool = True,
    array_keys: Optional[List[str]] = None,
    **dataset_kwargs,
) -> TorchDataLoader:
    """Return a single DataLoader backed by the union of all datasets.

    Samples are drawn uniformly at random from the merged pool each epoch,
    ensuring every mini-batch contains examples from multiple seismic styles.

    Args:
        data_paths: List of zarr store paths — one entry per seismic dataset.
        batch_size: Mini-batch size passed to DataLoader.
        sample_shape: (x, y, z) subvolume shape for each sample.
        num_workers: DataLoader worker processes (0 required on macOS/MPS).
        pin_memory: Pin host memory for faster GPU transfer.
        array_keys: 3-D array keys to sample from within each zarr store.
            None means all available 3-D keys.
        **dataset_kwargs: Extra keyword arguments forwarded to SeismicDataset
            (e.g. augment=True, normalize=True, target_std=1.0).

    Returns:
        A shuffled DataLoader over the merged ConcatDataset.

    Raises:
        ValueError: If data_paths is empty or no dataset can be opened.
    """
    if not data_paths:
        raise ValueError("data_paths must contain at least one path.")

    datasets: List[SeismicDataset] = []
    failed: List[str] = []
    for path in data_paths:
        try:
            ds = SeismicDataset(
                data_path=path,
                sample_shape=sample_shape,
                array_keys=array_keys,
                **dataset_kwargs,
            )
            datasets.append(ds)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{path}: {exc}")

    if failed:
        import warnings
        warnings.warn(
            f"Skipped {len(failed)} dataset(s) that could not be opened:\n"
            + "\n".join(f"  {f}" for f in failed),
            stacklevel=2,
        )

    if not datasets:
        raise ValueError(
            "No datasets could be opened. Check that data_paths are valid zarr stores."
        )

    merged = ConcatDataset(datasets)
    return TorchDataLoader(
        merged,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
```

**Code review checklist before commit:**
- [ ] `ConcatDataset` imported from `torch.utils.data` (not re-exported from this module)
- [ ] `failed` paths warned, not silently dropped
- [ ] `ValueError` raised when *all* paths fail (don't return an empty loader)
- [ ] No duplication with `create_dataloader` body (both delegate to `SeismicDataset.__init__`)
- [ ] Type annotations complete; `List` from `typing` already imported

**Tests to add (`tests/test_merged_dataloader.py`):**

1. `test_merged_dataloader_returns_dataloader` — result is a `DataLoader`
2. `test_merged_len_is_sum_of_dataset_lens` — `len(merged.dataset)` equals sum of constituent `__len__` values
3. `test_merged_batch_shape` — a single batch from the loader has the expected shape `(batch_size, 128, 128, 128)`
4. `test_merged_three_tensors_per_batch` — each batch unpacks as `(input_data, target, mask)`
5. `test_merged_single_path_equivalent_to_create_dataloader` — with one path, merged loader length matches `create_dataloader` length
6. `test_merged_empty_paths_raises` — `ValueError` on empty list
7. `test_merged_all_bad_paths_raises` — `ValueError` when all paths are invalid
8. `test_merged_one_bad_path_warns_and_continues` — warning issued, loader still created for good paths
9. `test_merged_length_scales_with_dataset_count` — merged length equals sum of constituent dataset lengths

**Iterate until:** `uv run pytest tests/test_merged_dataloader.py -v` → all pass.

**After phase completes:** update the Summary table at the bottom of this document.

---

### Phase 2 — `train.py`: replace dataset loop with merged loader in `train_epoch`

**Goal:** `train_epoch` accepts a single `DataLoader` instead of
`list[tuple[str, DataLoader]]`.  The inner per-dataset loop disappears; one flat
batch loop replaces it.

**Signature change:**

```python
# Before
def train_epoch(
    model: nn.Module,
    train_loaders: list,          # list[tuple[str, DataLoader]]
    …

# After
def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,     # single merged DataLoader
    …
```

**Loop change (pseudocode):**

```python
# Before — nested loops
for ds_idx, (ds_name, loader) in enumerate(train_loaders):
    …
    for batch_idx in range(len(loader)):
        …
    # end-of-dataset checkpoint + per-dataset TensorBoard figure

# After — flat loop
for batch_idx, (input_data, target, mask) in enumerate(train_loader):
    …
# end-of-epoch checkpoint + per-dataset TensorBoard figures (see Phase 3)
```

**Checkpointing change:**

`partial_latest.pt` currently saves `ds_idx` (which dataset was last completed).
With a merged loader this has no meaning.  Replace `ds_idx` with `batch_idx` so a
resumed run can at minimum report how far through the epoch it had reached.
The `ds_idx` field in the checkpoint dict should remain (set to -1) for backward
compatibility with any checkpoint reader.

**Thermal guard change:**

`thermal_guard.maybe_pause(ds_idx=…)` — replace with `ds_idx=-1` or remove the
argument if the guard is refactored (deferred to Phase 4).

**Code review checklist before commit:**
- [ ] `train_epoch` signature uses `train_loader: DataLoader`, not a list
- [ ] All call sites in `main()` updated to pass the merged loader
- [ ] `ds_idx` removed from per-batch state; checkpoint backward compat preserved (`ds_idx=-1`)
- [ ] Progress logging still fires every 10 batches with window-elapsed format
- [ ] EMA update, grad accumulation, grad clip logic unchanged
- [ ] No duplication of loss-accounting logic from the old per-dataset scope

**Tests to add / update:**

- `tests/test_train_epoch_smoke.py` (new): instantiate a tiny model, a small merged
  loader (2 datasets × 4 samples each, batch_size=2), run one epoch, assert returned
  loss is a finite float > 0.
- Update any existing test that calls `train_epoch` with the old `list[tuple]` signature.

**Iterate until:** `uv run pytest tests/ -v` → all pass.

**After phase completes:** update the Summary table at the bottom of this document.

---

### Phase 3 — `train.py`: end-of-epoch per-dataset TensorBoard figures

**Goal:** Preserve the useful per-dataset TensorBoard cross-section figures even
though the per-dataset loop is gone.

**Approach:**

After the flat training loop completes, run one **non-gradient** pass of one sample
per dataset using each `SeismicDataset` directly (accessible as
`merged_loader.dataset.datasets[i]`).  Forward through the model in `eval()` mode,
then restore `train()`.  This keeps plotting logic entirely separate from the hot
training path (SRP) and requires no per-batch bookkeeping.

**Code to add (helper function):**

```python
def _log_per_dataset_figures(
    model: nn.Module,
    merged_loader: DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    epoch_loss: float,
) -> None:
    """Log one cross-section figure per source dataset to TensorBoard.

    Runs a single inference batch per sub-dataset in eval mode.  This is called
    once at the end of each training epoch, so cost is negligible relative to
    the epoch itself.
    """
    model.eval()
    with torch.no_grad():
        for ds in merged_loader.dataset.datasets:
            ds_name = Path(ds.data_path).parent.name
            sample = ds[0]   # deterministic index-0 sample
            inp, tgt, _ = sample
            inp_t = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).float().to(device)
            out_t = model(inp_t)
            fig = make_4panel_figure(
                inp_t[0].cpu(), out_t[0].cpu(),
                torch.from_numpy(tgt).unsqueeze(0).cpu(),
                title=f"{ds_name}  |  epoch {epoch + 1}  |  loss {epoch_loss:.4f}",
            )
            writer.add_figure(f"train/{ds_name}", fig, global_step=epoch + 1)
            plt.close(fig)
    model.train()
```

**Code review checklist before commit:**
- [ ] `model.eval()` / `model.train()` are always paired (use try/finally if needed)
- [ ] `merged_loader.dataset.datasets` only accessed if dataset is a `ConcatDataset`
      — guard with `isinstance` check and a helpful warning if not
- [ ] Figures closed with `plt.close(fig)` to prevent memory leak
- [ ] Helper is not called during validation (only training end-of-epoch)

**Tests:**
- No new tests needed for this phase; visual output is validated by inspection in
  TensorBoard.  Smoke test from Phase 2 implicitly exercises this code path when
  `writer` is not None.

**After phase completes:** update the Summary table at the bottom of this document.

---

### Phase 4 — `train.py`: update `_build_loaders` and call site in `main()`

**After phase completes:** update the Summary table at the bottom of this document.

**Goal:** Replace `_build_loaders` return value with a merged loader for train and
keep val as a list of per-dataset loaders (useful for per-dataset val loss tracking).

**`_build_loaders` refactor:**

```python
def _build_loaders(
    train_paths: List[str],
    val_paths: List[str],
    loader_kwargs: dict,
) -> tuple[DataLoader, list[tuple[str, DataLoader]]]:
    """Build one merged train DataLoader and per-dataset val DataLoaders.

    Returns:
        train_loader: Single ConcatDataset-backed DataLoader for training.
        val_loaders: List of (name, DataLoader) pairs for per-dataset val metrics.
    """
```

**`main()` call site:**

```python
# Before
train_loaders, val_loaders = _build_loaders(active_train, active_val, loader_kwargs)
…
train_epoch(model, train_loaders, …)

# After
train_loader, val_loaders = _build_loaders(active_train, active_val, loader_kwargs)
…
train_epoch(model, train_loader, …)
```

**Code review checklist before commit:**
- [ ] Return type annotation added to `_build_loaders`
- [ ] `n_train` still printed correctly (use `len(train_loader)`)
- [ ] Resuming from a `thermal_latest.pt` checkpoint still works: `ds_idx` in
      checkpoint is now always -1; resume logic must not try to skip-ahead by
      fast-forwarding through datasets (remove any such fast-forward logic)
- [ ] `train_paths` / `val_paths` still passed through for checkpoint path tracking

**Tests to add:**
- `test_build_loaders_returns_merged_train` — assert `train_loader` is a `DataLoader`
  whose `.dataset` is a `ConcatDataset`.
- `test_build_loaders_val_is_list_of_tuples` — assert val return is `list[tuple[str, DataLoader]]`.

**Iterate until:** `uv run pytest tests/ -v` → all pass.

---

### Phase 5 — Validate with smoke test and commit

**Steps:**

1. Run full test suite: `uv run pytest tests/ -v`
2. Run smoke training for 2 epochs with 2 datasets and confirm TensorBoard shows
   interleaved dataset figures and a single merged loss curve.
3. Confirm `checkpoints/partial_latest.pt` is written mid-epoch.
4. Review final diff for:
   - No dead code left from the old per-dataset loop
   - No commented-out code blocks
   - All `TODO` / `FIXME` markers resolved
5. Commit all changed files with message:

```
feat: replace per-dataset train loop with ConcatDataset merged loader

Training now draws mini-batches uniformly from all seismic datasets
simultaneously rather than processing one dataset at a time.  This
improves gradient diversity per optimizer step.

Changes:
- dataloader.py: add create_merged_dataloader() using ConcatDataset
- train.py: train_epoch() accepts a single DataLoader; per-dataset TensorBoard
  figures logged via a post-epoch helper; _build_loaders() returns merged train
  loader and per-dataset val loaders
- tests/test_merged_dataloader.py: 9 new tests covering create_merged_dataloader
- tests/test_train_epoch_smoke.py: smoke test for merged-loader training path
```

**After phase completes:** update the Summary table at the bottom of this document.

---

### Phase 6 — Human validation on Mac mini (approval gate)

**Goal:** Confirm real end-to-end behavior under the target hardware/runtime
conditions before finalizing.

**Steps:**

1. Run a human-observed training test on the Mac mini for at least 2 epochs
    (approximately 4 hours total runtime).
2. Verify expected behavior during the run:
    - Training progresses without crashes/hangs.
    - Mixed-batch logs and TensorBoard outputs look correct.
    - Checkpoints and partial checkpoints are written as expected.
3. Record observations and any anomalies.
4. Request explicit user approval that the feature appears to work properly.

**Exit gate:**
- User explicitly approves the 2-epoch Mac mini run results.

**After phase completes:** update the Summary table at the bottom of this document.

---

### Phase 7 — Post-approval wrap-up (summary + final commit)

**Goal:** Finalize the session after human approval.

**Steps:**

1. Create a new session summary in `.agentic-docs` describing:
    - Implemented ConcatDataset mixed-batch changes
    - Test results (automated + human 2-epoch run)
    - Deviations from the plan and rationale
2. Stage and commit final approved Mixed-Batch changes with a clear commit
    message.
3. Verify final `git status` and commit history are clean and understandable.

**Iterate until:**
- Session summary is added and accurate
- Final commit is present with intended files only

**After phase completes:** update the Summary table at the bottom of this document.

---

## Summary (to be filled in after implementation)

_This section will be updated once each phase is complete._

| Phase | Status | Notes |
|---|---|---|
| 0 — session housekeeping + branch setup | completed | checkpoint commit `489657d` created on `main`; switched to `feat/mixed-batch-concatdataset` |
| 1 — `create_merged_dataloader` | completed | implemented in `dataloader.py`; Mac mini run: `tests/test_merged_dataloader.py` = 9 passed |
| 2 — flat `train_epoch` loop | completed | `train_epoch` accepts single `DataLoader`; `_merge_train_loaders` adapter in `main()`; Mac mini: 11/11 tests passed |
| 3 — per-dataset TensorBoard figures | completed | `_log_per_dataset_figures()` added to `train.py`; called from `main()` after `train_epoch`; guarded with `isinstance(ConcatDataset)`; no new tests (visual TensorBoard check) |
| 4 — `_build_loaders` + `main()` wiring | completed | `_build_loaders` now returns `(DataLoader \| None, list[tuple])` directly; `_merge_train_loaders` removed; smoke test updated to use `ConcatDataset` inline |
| 5 — smoke test + git commit | not started | |
| 6 — human Mac mini validation + approval | not started | |
| 7 — post-approval summary + final commit | not started | |
