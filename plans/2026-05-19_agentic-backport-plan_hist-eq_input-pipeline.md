# Issue: Backport Histogram Equalization + 128^3 Input Pipeline from feat/unet-decoder-fix to main

## Objective
Backport only data-input related improvements from branch feat/unet-decoder-fix into a new branch cut from main.

Do include:
- Nonlinear amplitude transform support approximating standard normal behavior, derived once per full 3D zarr dataset and applied to 128x128x128 training subsets.
- Input generation/augmentation changes required for stable 128x128x128 training samples.
- Train-script argument plumbing needed to activate and configure the above.
- Selective updates in train_multi_datasets.sh for input-pipeline flags and script robustness.

Do not include:
- Model architecture changes.
- Loss-function changes.

## Confidence and Constraints
- Confidence: High, provided strict file allowlist and validation gates are enforced.
- Execution mode: autonomous, non-interactive.
- Hard guardrail: fail if any staged diff touches model or loss files.

## Branch Strategy
1. Create new branch from main:
   - name suggestion: feat/input-histeq-backport
2. Keep changes surgical and file-scoped.
3. Commit in small, reviewable slices.

## Allowed / Blocked File Scope
Allowed (edit/add):
- src/synthoseis_pre_train/histogram_equalizer.py
- src/synthoseis_pre_train/dataloader.py
- src/synthoseis_pre_train/augmentation.py
- src/synthoseis_pre_train/masking.py
- src/synthoseis_pre_train/transforms.py (only if required)
- train.py
- train_multi_datasets.sh (hunk-level allowlist: input-transform/masking flags + reliability traps only)
- tests/* related to histogram equalization and input pipeline
- docs/README updates only if needed for new CLI options

Blocked (must remain unchanged):
- src/synthoseis_pre_train/models.py
- src/synthoseis_pre_train/dyn_models.py
- src/synthoseis_pre_train/losses.py

## Implementation Steps
1. Diff extraction
- Compare main...feat/unet-decoder-fix and isolate hunks touching allowed files.
- Build a mapping of symbol-level changes (new args, new helper calls, new modules).

2. Add histogram equalizer module
- Bring in histogram_equalizer.py and ensure imports resolve on main.
- Confirm transform derivation uses full dataset scope (entire zarr seismic volume), not single sampled subvolume.
- Confirm transform application is deterministic and safe for out-of-range values.

3. Integrate into dataloader/input pipeline
- Add/port dataloader hooks to select amplitude transform mode.
- Ensure 128x128x128 sample extraction, axis handling, and augmentation path remain consistent.
- Keep masking and augmentation behavior compatible with existing training loop expectations.

4. Wire train.py arguments
- Add only input-pipeline-related CLI args needed to trigger the new transform and options.
- Preserve existing defaults and backwards compatibility where possible.
- Do not modify model or loss selection semantics.

4b. Selectively port train_multi_datasets.sh
- Backport only input-pipeline CLI forwarding changes: amplitude transform modes, masking controls, transform persistence group.
- Backport reliability hardening (error traps, output-dir creation) when independent of model/loss options.
- Do not port model architecture options, loss expansions, SSIM tuning, or cluster-loss controls.

5. Add or port tests
- Histogram mapping behavior (fit/load/apply).
- 128x128x128 sample shape and mask alignment invariants.
- Basic smoke test that dataloader + transform path runs at least one batch.

6. Validation gates
- Run targeted tests first, then broader smoke tests.
- Verify blocked files are unchanged.
- Verify no import/runtime errors from train.py path.

## Required Validation Commands
- pytest tests -k "histogram or quantile or dataloader or augmentation or masking"
- python train.py --help
- bash train_multi_datasets.sh --help
- Existing smoke script if available (read-only verification first, then run): run_smoke_test.sh
- git diff --name-only main...HEAD

## Acceptance Criteria
- New branch from main contains only allowed-file changes.
- Histogram equalization transform is available and derived from full zarr dataset scope.
- Training input pipeline can emit augmented 128x128x128 samples with expected shape/order.
- train_multi_datasets.sh forwards the new input-pipeline options correctly without introducing model/loss option changes.
- No model/loss code changed.
- Tests/smoke checks pass.

## Out of Scope
- Any UNet decoder changes, upsampling changes, or model topology edits.
- Any loss redesign/refactor or metric logic changes.

## Rollback Plan
- If validation fails due to coupling with excluded model/loss changes:
  1. Revert last commit slice only.
  2. Replace with compatibility shim in allowed files.
  3. Re-run targeted tests.

## Deliverables
- Code changes in allowed files only.
- Test updates/additions for histogram/input pipeline behavior.
- Brief migration notes in docs/README if new CLI flags were introduced.
