# Masking Loss Re-evaluation Plan

Date: 2026-05-21
Scope: Training pipeline masking behavior review and remediation, with MONAI 1.0.0 API/pattern comparison.

## Objective

Re-evaluate whether non-standard masking and weighting in loss computation is the primary cause of weak training improvements, and isolate masking effects from architecture and data quality factors.

## Required Outcomes

- Analyze mask usage overall in the training pipeline.
- Analyze mask usage inside loss computation.
- Analyze mask impact on back-propagation and gradient flow.
- Analyze mask usage that is restricted to printed stats, diagnostics, and QC plots.
- Compare current pipeline behavior to MONAI 1.0.0 training/loss masking patterns.

## Constraints and Process Rules

- Create and work on a new branch from local main.
- Execute work in explicit phases.
- Each phase must include:
  - constructive critic code review,
  - tests written or updated for that phase,
  - tests passing before phase completion,
  - git commit before proceeding.
- Final phase must include:
  - end-to-end review,
  - end-to-end testing,
  - final commit,
  - session summary in .agentic-docs as markdown and html.

## Phase Plan

### Phase 0: Branch, Baseline, and Reproducibility Lock

Purpose: Establish a stable reference so all mask changes are measurable.

Tasks:
1. Create branch from local main: feat/masking-loss-reevaluation.
2. Freeze a baseline config (dataset slice, seeds, epochs, loss type, mask params).
3. Capture baseline metrics and representative QC artifacts.
4. Record exact commands used for reproducibility.

Constructive critic review checklist:
- Baseline config is minimal, readable, reproducible.
- No hidden defaults that can distort comparisons.
- Logging includes enough detail to diagnose mask behavior.

Tests and gate:
- Run targeted existing tests that touch masking and loss paths.
- Baseline smoke run must complete successfully.
- Commit baseline setup and reproducibility notes.

### Phase 1: Ground-truth Mask Usage Mapping

Purpose: Produce a complete, unambiguous mask map from data loading to reporting.

Tasks:
1. Map where masks are created, transformed, expanded, combined.
2. Map exact mask semantics at each boundary (True means what, False means what).
3. Map all call sites where masks enter training logic.
4. Separate training-path usage from diagnostics-only usage.

Deliverables:
- Markdown report in plans folder.
- HTML report in plans folder.

Constructive critic review checklist:
- No ambiguous mask semantics remain.
- Every mapping entry references concrete file/function/symbol.
- Training-path and diagnostics-path are clearly separated.

Tests and gate:
- Add targeted tests for mask inversion semantics and mask union behavior.
- All targeted tests pass.
- Commit phase output and tests.

### Phase 2: Loss and Gradient Path Validation

Purpose: Verify whether masking changes optimization behavior in unintended ways.

Tasks:
1. Trace all loss computation branches with mask inputs.
2. Validate reduction and normalization with variable valid voxel counts.
3. Validate gradient behavior on masked vs unmasked voxels.
4. Verify sparse/empty-mask edge-case handling.

Constructive critic review checklist:
- Loss-path logic is explicit and fail-fast.
- No silent fallback that can flatten spatial structure unexpectedly.
- Gradient behavior matches documented intent.

Tests and gate:
- Add targeted tests for:
  - masked loss routing,
  - fallback behavior correctness,
  - zero-gradient in masked-out regions,
  - sparse-mask numerical stability.
- All targeted tests pass.
- Commit phase output and tests.

### Phase 3: MONAI 1.0.0 API/Pattern Comparison

Purpose: Compare existing approach to MONAI-standard masked loss patterns and training separation practices.

MONAI comparison scope:
- Masked loss wrappers and contracts (pattern-level alignment).
- Reduction behavior (mean/sum/none semantics).
- Masked region contribution to gradient calculation.
- Separation of train loss vs evaluation/metric reporting.

Tasks:
1. Produce gap table: current behavior vs MONAI pattern.
2. Rank gaps by training-risk severity.
3. Propose minimal and safe remediation order.

Constructive critic review checklist:
- Recommendations are practical, not over-engineered.
- Human readability and maintainability are preserved.
- Changes are optimized but not opaque.

Tests and gate:
- Add or update targeted tests that enforce selected remediations.
- All targeted tests pass.
- Commit phase output and tests.

### Phase 4: Mask Remediation Implementation

Purpose: Implement only high-value mask-related fixes first.

Tasks:
1. Harden mask semantics and loss dispatch logic.
2. Isolate diagnostics-only masking from training hot path.
3. Remove redundant per-batch diagnostic loss recomputation from critical path when safe.
4. Keep behavior and APIs explicit and testable.

Constructive critic review checklist:
- Code is clear and not AI-slop.
- Naming and docs are coherent.
- Runtime overhead is reduced or unchanged.

Tests and gate:
- Run targeted tests for affected files.
- Run train epoch smoke test.
- Commit implementation and tests.

### Phase 5: End-to-End Validation and Session Closeout

Purpose: Prove final integrated behavior and preserve traceability.

Tasks:
1. Execute end-to-end smoke plus selected convergence checks.
2. Validate no regression relative to baseline on key metrics/trends.
3. Perform final constructive critic review of the complete change set.
4. Create final session summary in .agentic-docs:
   - markdown,
   - html,
   - what changed,
   - why,
   - residual risks,
   - recommended next steps.

Tests and gate:
- End-to-end tests pass.
- Final commit created.

## Proposed Targeted Test Matrix

- tests/test_losses.py
- tests/test_masking_sampling.py
- tests/test_loss_cluster_aware.py
- tests/test_train_epoch_smoke.py
- New tests to add during execution:
  - test_compute_masked_loss_routing.py
  - test_mask_gradient_flow.py
  - test_mask_semantics_contract.py
  - test_diagnostics_no_grad_separation.py

## Key Risks to Monitor

1. Inconsistent mask semantics across modules.
2. Loss routing fallback that changes tensor shape assumptions.
3. Sparse-mask numeric instability in local-window losses.
4. Diagnostic logic accidentally affecting training throughput or behavior.

## Confidence Strategy

- Keep phase deltas small and measurable.
- Require passing tests and commit at every phase.
- Prefer fail-fast behavior for unsupported mask/loss contracts.
- Defer non-mask architecture/data interventions unless mask analysis proves insufficient.

## Open Questions (Must Be Asked During Execution If Needed)

1. Baseline run definition: exact dataset subset and runtime budget?
2. Minimum acceptable regression threshold for moving between phases?
3. Whether to include optional full-suite validation at phase boundaries when risk is high?

## File Output Summary

- This markdown plan: plans/2026-05-21_masking-loss-reevaluation-plan.md
- Matching HTML plan: plans/2026-05-21_masking-loss-reevaluation-plan.html
