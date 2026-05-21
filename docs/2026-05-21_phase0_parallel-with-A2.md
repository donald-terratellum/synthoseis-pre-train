# Phase 0 Parallel Execution Plan (A2 Running)

Date: 2026-05-21
Branch: feat/masking-loss-reevaluation
Mode: Option 2 (keep A2 ablation running, perform non-training Phase 0 tasks now)

## Decision Record

- A1 ablation completed and can serve as operational baseline evidence.
- A2 ablation remains in progress and will not be interrupted yet.
- Constraint: only one train.py instance can run on the Mac mini.
- Therefore, Phase 0 proceeds with non-training implementation tasks only.

## In Scope Right Now (No train.py launches)

1. Branch setup and workspace hygiene.
2. Baseline artifact inventory and traceability documentation.
3. Constructive critic review checklist preparation.
4. Test-plan definition and test stubs that do not require active training runs.
5. Phase 1 analysis scaffolding in documentation.

## Deferred Until A2 Stops

1. Any new baseline or frozen-data training command.
2. Any train.py smoke or baseline rerun.
3. Any training-side metric comparison that requires a new run.

## Baseline Artifact References

- scripts/run_ablation_A1_to_C2.sh
- train_ablation_A1_to_C2.log
- checkpoints/ablation_2026_05_20/A1_lr5e5
- checkpoints/ablation_2026_05_20/summary_A1_C2.csv

## Phase 0 Checklist (Parallel Track)

Status legend: [x] done, [ ] pending

- [x] Create implementation branch from local main.
- [x] Record decision to reuse A1 as baseline evidence (operational baseline).
- [x] Record constraint and deferred training tasks while A2 is active.
- [ ] Capture exact A1 final metrics from summary CSV in this file.
- [ ] Add explicit baseline caveat: dataset membership drift due to prune/split events.
- [x] Draft critic review notes focused on mask semantics and loss-routing safety.
- [x] Define targeted test additions for mask semantics and gradient behavior.
- [ ] Commit Phase 0 critic notes and test-plan artifacts.

## Constructive Critic Review Prompts

1. Are mask boolean semantics consistent at each boundary (creation, inversion, loss, diagnostics)?
2. Is loss routing explicit and fail-fast, or can unsupported criteria silently misbehave?
3. Are diagnostics metrics isolated from training-path backprop computations?
4. Is code readable and maintainable with minimal hidden side effects?
5. Are normalization and reduction choices robust under sparse valid-mask edge cases?

## Constructive Critic Review Notes (Initial)

1. Mask semantics appear split across modules and rely on inversion at the train boundary. Risk: accidental semantic drift if a new call path bypasses the inversion contract.
2. Loss routing currently supports multiple criteria styles, but unsupported signatures must fail fast instead of silently flattening spatial structure.
3. Diagnostics and QC calculations need strict separation from the hot training path to avoid hidden compute overhead and accidental gradient coupling.
4. Sparse-mask edge cases require explicit denominator and neighborhood-validity safeguards to avoid unstable optimization signals.
5. Cluster-aware weighting should remain transparent and test-backed so weighting does not inadvertently dominate primary reconstruction terms.

## Targeted Test Plan (No New train.py Run Required)

Run now (safe while A2 is running):

1. `pytest -q tests/test_losses.py`
2. `pytest -q tests/test_masking_sampling.py`
3. `pytest -q tests/test_loss_cluster_aware.py tests/test_loss_cluster_aware_extra.py`
4. `pytest -q tests/test_train_epoch_smoke.py -k masked`

Prepare for addition in Phase 1/2 implementation (unit-level):

1. `tests/test_compute_masked_loss_routing.py`
	- Verify criteria that accept `valid_mask` receive full-shape tensors.
	- Verify unsupported criteria path fails clearly and early.
2. `tests/test_mask_semantics_contract.py`
	- Assert consistent meaning for mask booleans at dataloader, train, and loss interfaces.
3. `tests/test_mask_gradient_flow.py`
	- Confirm masked-out regions contribute zero gradient where intended.
	- Confirm gradients remain finite for sparse valid regions.
4. `tests/test_diagnostics_no_grad_separation.py`
	- Validate diagnostics/statistics code runs under no-grad and is not used for backward.

Acceptance for this parallel Phase 0 track:

1. Existing targeted tests pass on current branch without launching new train.py runs.
2. New test specifications are documented with clear pass criteria.
3. A1 baseline extraction remains pending until summary CSV is finalized.

## Transition Trigger

When you stop A2 (or it completes), run training-dependent Phase 0 items immediately:

1. Confirm baseline extraction from A1 artifacts.
2. Run a minimal frozen-data reproducibility check if required.
3. Continue Phase 1 mask-path implementation.
