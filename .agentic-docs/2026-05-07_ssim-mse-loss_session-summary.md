# Session Summary - 2026-05-07 - SSIM-MSE Loss

## Completed in this session
- Created phased implementation plan with checklist structure:
  - plans/2026-05-07_ssim-mse-loss-implementation-plan.md
- Implemented memory-aware 3D SSIM-MSE loss module:
  - src/synthoseis_pre_train/losses.py
- Integrated SSIM-MSE into training flow:
  - train.py
- Exposed SSIM-MSE options in launcher and pass-through:
  - train_multi_datasets.sh
- Added unit tests for loss behavior and masking:
  - tests/test_losses.py

## Design choices
- SSIM variant uses zero-mean assumption for seismic data:
  - local means are not estimated/used
  - local second-order moments use x^2, y^2, x*y only
- 3D Gaussian window is separable (three 1D conv3d passes) to reduce kernel memory overhead.
- Mixed loss form keeps "lower is better" semantics:
  - total = mse + ssim_weight * ssim_component
  - ssim_component = 0.5 * (1 - ssim_score)
- Valid-mask aware SSIM pooling avoids invalid edge regions driving local statistics.

## New SSIM-MSE knobs
- Python args in train.py:
  - --loss_type ssim_mse
  - --ssim_window_size
  - --ssim_sigma
  - --ssim_data_range
  - --ssim_weight
  - --ssim_min_valid_ratio
- Shell args in train_multi_datasets.sh:
  - --loss-type ssim_mse
  - --ssim-window-size
  - --ssim-sigma
  - --ssim-data-range
  - --ssim-weight
  - --ssim-min-valid-ratio

## Validation status
- Editor diagnostics: no errors in changed files.
- Syntax checks passed:
  - train.py
  - src/synthoseis_pre_train/losses.py
  - train_multi_datasets.sh
- Unit test execution was not completed in terminal due environment path mismatch for the runtime Python executable.

## Remaining Phase-4 work
- Run end-to-end training smoke with loss_type=ssim_mse.
- Compare stability and memory against huber/mse on same data split.
- Tune defaults (especially ssim_weight, sigma, min_valid_ratio) from empirical runs.

## Updates since 2026-05-07 (short)
- Added composite cluster-aware loss and GPU-friendly smoothing: `src/synthoseis_pre_train/losses.py` (CompositeClusterAwareLoss).
- Added composite cluster-aware loss and GPU-friendly smoothing: `src/synthoseis_pre_train/losses.py` (CompositeClusterAwareLoss).
  - Behavior: re-derives a 2D trace-level mask from batch labels where a trace is marked when all depth voxels are masked; smooths with a 5x5 (kernel_size adjustable) avg filter on the GPU; selects traces where smoothed density > eps and computes `L_cluster` over those traces.
  - Composite loss: `loss = base_weight * L_base + cluster_weight * L_cluster` (defaults: base_weight=1/3, cluster_weight=2/3) so voxels near masked clusters are emphasized — this makes subregions of large samples (e.g., 182^3 patches) contribute proportionally more to the loss when they contain masked clusters.
  - Exposed via CLI `--enable-cluster-loss` and kernel/weight parameters in `train_multi_datasets.sh` / `train.py`.
- Masking API refactor: removed legacy `trace_mask_ratio` and switched to `target_masked_fraction` in `src/synthoseis_pre_train/masking.py` and `src/synthoseis_pre_train/dataloader.py`.
- Training wiring fixes: removed legacy CLI arg and updated `_compute_masked_loss` to call losses accepting `valid_mask` (`train.py`).
- Launcher/script updates: `train_multi_datasets.sh` cleaned of `TRACE_MASK_RATIO`, added explicit cluster-loss startup print; `train.py` startup prints updated.
- Packaging & env helpers: `pyproject.toml` adjusted for PyTorch/pip/wheel, and added `scripts/bootstrap_pytorch.sh` and README notes for post-`uv sync` bootstrapping.
- Diagnostics and tests: added diagnostics plotting script with density overlay; added tests `tests/test_loss_cluster_aware.py` and `tests/test_loss_cluster_aware_extra.py`.
- Documentation: added session summary and branch/commit instructions under `.agentic-docs/` and a concise revised plan under `plans/`.

## Post-update validation notes
- Applied patches have aligned call sites; one runtime issue (unbound TRACE_MASK_RATIO) was fixed in `train_multi_datasets.sh`.
- Some unit tests still require a bootstrapped environment (`uv sync` then `uv pip install torch...`) before running; see `.agentic-docs/branch_and_commit_instructions.txt` and README notes.
