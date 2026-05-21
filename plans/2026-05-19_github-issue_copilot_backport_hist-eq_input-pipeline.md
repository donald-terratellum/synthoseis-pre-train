# Backport Input Pipeline Only from feat/unet-decoder-fix (Histogram Equalization + 128^3 Augmentation)

## Summary
Implement a selective backport from branch feat/unet-decoder-fix into a new branch from main.

Scope is strictly limited to training-input pipeline behavior:
- Nonlinear amplitude transform support (including histogram equalization) derived from the full 3D zarr dataset and applied to 128x128x128 training samples.
- Input generation and augmentation updates required for robust 128x128x128 sample creation.
- Train entrypoint and launcher-script plumbing needed to pass/activate those input options.

Do not include model architecture or loss-function changes.

## Branch Requirement (Mandatory)
Copilot must create and use a new branch from main before editing:
- Suggested branch name: feat/input-histeq-backport
- Base: main

## Source and Target
- Source reference branch: feat/unet-decoder-fix
- Target integration baseline: main

## In Scope
- Add and integrate histogram equalization/input transforms for training inputs.
- Port only input-pipeline related updates in:
  - src/synthoseis_pre_train/histogram_equalizer.py
  - src/synthoseis_pre_train/dataloader.py
  - src/synthoseis_pre_train/augmentation.py
  - src/synthoseis_pre_train/masking.py
  - src/synthoseis_pre_train/transforms.py (only if required by integration)
  - train.py
  - train_multi_datasets.sh (partial, input-related only)
- Add/update tests for histogram transform behavior and 128x128x128 sample invariants.

## Out of Scope (Hard Exclusions)
- Any changes to model architecture code:
  - src/synthoseis_pre_train/models.py
  - src/synthoseis_pre_train/dyn_models.py
- Any loss-function changes:
  - src/synthoseis_pre_train/losses.py
- Any unrelated refactors.

## train_multi_datasets.sh Rule
Only port script changes that are directly relevant to input pipeline and robustness:
- Allowed examples:
  - Input-transform argument forwarding (histogram/quantile/stdev modes)
  - Masking and transform-persistence related flags
  - Safe script hardening (error traps, output-dir creation) if independent
- Not allowed examples:
  - Model-arch options
  - Loss-type expansions
  - SSIM tuning additions
  - Cluster-loss tuning additions

## Implementation Tasks
1. Create branch feat/input-histeq-backport from main.
2. Diff main...feat/unet-decoder-fix and isolate only input-pipeline hunks.
3. Add histogram_equalizer module and integrate into dataloader flow.
4. Port required argument plumbing in train.py.
5. Port only allowed partial changes in train_multi_datasets.sh.
6. Add/update tests for:
   - fit/load/apply transform behavior
   - 128x128x128 sample shape and axis-order invariants
   - mask alignment after augmentation/masking
7. Validate and ensure excluded files are untouched.

## Validation Checklist
- Run targeted tests:
  - pytest tests -k "histogram or quantile or dataloader or augmentation or masking"
- Check CLI wiring:
  - python train.py --help
  - bash train_multi_datasets.sh --help
- Optional smoke path:
  - bash run_smoke_test.sh
- Scope guard:
  - git diff --name-only main...HEAD
  - Confirm no edits in:
    - src/synthoseis_pre_train/models.py
    - src/synthoseis_pre_train/dyn_models.py
    - src/synthoseis_pre_train/losses.py

## Acceptance Criteria
- Backport exists on new branch feat/input-histeq-backport from main.
- Histogram equalization transform is available and derived from full zarr dataset scope.
- Training pipeline emits valid augmented 128x128x128 samples.
- train.py and train_multi_datasets.sh correctly expose/forward input-pipeline options.
- No model/loss files changed.
- Tests and checks pass.

## Deliverables
- Code changes limited to in-scope files.
- Focused tests covering transform + input pipeline behavior.
- Brief PR summary describing exactly what was included/excluded.
