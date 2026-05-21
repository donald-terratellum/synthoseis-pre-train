# Phase 2 Mask Gap Triage

Date: 2026-05-21
Branch: feat/masking-loss-reevaluation
Precondition: Phase 1 boundary gate complete (42 passed, commit 7260095).
Operating constraint: Keep A2 training run undisturbed; prioritize non-training-safe changes first.

## Objective

Convert Phase 1 findings into an ordered remediation backlog with explicit acceptance criteria, test gates, and safe rollout sequencing.

## Inputs From Phase 1

1. Mask semantics are correct but inversion at train boundary is still implicit.
2. Routing safety improved via fail-fast behavior and pointwise fallback constraints.
3. QC overhead is now opt-in and interval-gated, but runtime budget should be measured under real settings.
4. MONAI comparison indicates value in explicit contracts over signature-driven dispatch.

## Ranked Gap List

### High Priority

1. Contract formalization at train boundary
   - Gap: mask meaning transition (preserve/masked vs valid_mask) remains implicit.
   - Risk: future refactors can silently flip semantics.

2. Dispatcher contract clarity
   - Gap: runtime signature inspection still needed for mixed criteria.
   - Risk: custom structural criteria may be misrouted unless valid_mask support is explicit.

### Medium Priority

1. QC observability and overhead accounting
   - Gap: opt-in QC exists but no standard runtime budget check.
   - Risk: accidental enablement in large runs can degrade throughput.

2. Error ergonomics for unsupported criteria
   - Gap: fail-fast error exists but troubleshooting path is not documented in-user flow.
   - Risk: users may patch around error incorrectly.

### Low Priority

1. MONAI-style alignment polish
   - Gap: full fixed loss contracts are not universal yet.
   - Risk: low immediate impact; mostly maintainability.

## Remediation Backlog With Acceptance Criteria

1. Add explicit mask contract section in training docs
   - Scope: define source mask semantics, inversion point, and valid_mask consumer rules.
   - Acceptance: one canonical section referenced by train and loss docs; no conflicting statements.

2. Add contract assertion helper around _compute_masked_loss entry
   - Scope: centralize checks for dtype/shape/meaning expectations before dispatch.
   - Acceptance: tests cover expected pass and failure cases for malformed masks.

3. Add dispatcher compatibility matrix to tests
   - Scope: enumerate supported criterion categories and expected routing path.
   - Acceptance: matrix test fails on unsupported criteria without valid_mask and passes with opt-in pointwise fallback.

4. Add QC budget check in smoke/integration path
   - Scope: when QC is enabled, record average per-batch overhead and assert within configurable budget.
   - Acceptance: deterministic test or benchmark harness with documented threshold.

5. Add actionable troubleshooting note for fail-fast criteria errors
   - Scope: map common errors to fixes (implement valid_mask, choose pointwise criterion, or explicit opt-in where safe).
   - Acceptance: docs referenced from README or training guide.

## Implementation Order

1. Contract docs and assertions (high priority, low blast radius).
2. Dispatcher compatibility matrix expansion.
3. QC overhead budget instrumentation and tests.
4. Documentation and ergonomics polish.

## Progress Update (2026-05-21)

Completed in code and tests:

1. Item 1 complete: mask contract assertion helper added at training loss boundary.
2. Item 2 complete: dispatcher compatibility matrix test coverage added.
3. Item 3 complete: QC overhead budget instrumentation added with deterministic helper tests.

Validation status:

- Focused Phase 2 suite passed: 23 tests.
- A2 training process remained undisturbed.

Remaining in this phase:

1. Run full required non-training gate including baseline suites and all new Phase 2 suites.

Item 5 status:

- Complete: actionable fail-fast masked-loss troubleshooting guidance added to README training docs.

## Test Gate For Phase 2 Completion

Required non-training suite:

- tests/test_losses.py
- tests/test_masking_sampling.py
- tests/test_loss_cluster_aware.py
- tests/test_loss_cluster_aware_extra.py
- tests/test_compute_masked_loss_routing.py
- tests/test_mask_semantics_contract.py
- tests/test_train_epoch_smoke.py

Additional Phase 2 tests to add:

1. tests/test_mask_contract_assertions.py
2. tests/test_masked_loss_dispatch_matrix.py
3. tests/test_qc_overhead_budget.py

Gate condition:

- All required and new Phase 2 tests pass.
- No training process interruption required.
- Docs updated in both md/html under plans.

## Deliverables

1. Code updates for contract assertions and dispatch matrix coverage.
2. QC budget instrumentation and test(s).
3. Updated docs with explicit troubleshooting and contract guidance. (complete)
4. Phase 2 completion report artifact.

## Out Of Scope

1. Major architecture rewrite of the training loop.
2. Replacing all custom criteria with MONAI-native equivalents.
3. Any change requiring A2 run interruption.

## Execution Notes

- Favor small commits by backlog item.
- Keep all changes reversible and test-gated.
- Preserve current default behavior: QC metrics disabled unless explicitly enabled.
