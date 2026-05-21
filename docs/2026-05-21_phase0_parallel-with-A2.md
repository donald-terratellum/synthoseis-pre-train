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
- [ ] Draft critic review notes focused on mask semantics and loss-routing safety.
- [ ] Define targeted test additions for mask semantics and gradient behavior.
- [ ] Commit non-training Phase 0 artifacts.

## Constructive Critic Review Prompts

1. Are mask boolean semantics consistent at each boundary (creation, inversion, loss, diagnostics)?
2. Is loss routing explicit and fail-fast, or can unsupported criteria silently misbehave?
3. Are diagnostics metrics isolated from training-path backprop computations?
4. Is code readable and maintainable with minimal hidden side effects?
5. Are normalization and reduction choices robust under sparse valid-mask edge cases?

## Transition Trigger

When you stop A2 (or it completes), run training-dependent Phase 0 items immediately:

1. Confirm baseline extraction from A1 artifacts.
2. Run a minimal frozen-data reproducibility check if required.
3. Continue Phase 1 mask-path implementation.
