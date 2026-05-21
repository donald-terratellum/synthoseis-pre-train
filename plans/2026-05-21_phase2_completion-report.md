# Phase 2 Completion Report: Mask Gap Triage

Date: 2026-05-21
Branch: feat/masking-loss-reevaluation
Status: Complete (non-training validation gate passed)

## Executive Summary

Phase 2 is complete. The mask contract and dispatch boundary were hardened, dispatcher compatibility was expanded with matrix coverage, QC overhead budget instrumentation was added, and fail-fast troubleshooting guidance was added to user-facing docs. The final non-training gate passed with 57 tests.

## Scope Completed

1. Contract assertion helper at loss boundary.
2. Dispatcher compatibility matrix test coverage.
3. QC overhead budget instrumentation and telemetry coverage.
4. Documentation ergonomics for fail-fast criterion routing errors.

## Delivered Changes

### 1) Mask Contract Hardening

- Added strict contract checks at the training loss boundary in `train.py`:
  - output/target shape match
  - mask shape match
  - mask dtype is bool
  - output/target/mask on same device
- Added helpers:
  - `_assert_mask_contract`
  - `_compute_qc_overhead_ratio`
  - `_is_qc_overhead_within_budget`

Evidence commit:

- a46faf0 train+test: enforce mask contract at loss boundary

### 2) Dispatcher Compatibility Matrix

- Added `tests/test_masked_loss_dispatch_matrix.py` covering:
  - valid_mask-preferred routing
  - approved pointwise fallback routing
  - explicit opt-in fallback routing
  - fail-fast unsupported criterion rejection

Evidence commit:

- 4cbb5c9 test: add masked-loss dispatcher compatibility matrix coverage

### 3) QC Overhead Budget Instrumentation

- Added train-time QC overhead telemetry in `train.py`:
  - `qc_metrics_calls`
  - `qc_time_sec`
  - `total_batch_time_sec`
  - `qc_overhead_ratio`
  - `qc_overhead_budget_exceeded`
- Added new CLI parameter:
  - `--batch_qc_max_overhead_ratio`
- Added deterministic QC overhead helper tests:
  - `tests/test_qc_overhead_budget.py`
- Extended train smoke tests for QC overhead telemetry:
  - `tests/test_train_epoch_smoke.py`

Evidence commit:

- 4c75ac6 train+test: add QC overhead budget instrumentation and telemetry coverage

### 4) Documentation Ergonomics (Item 5)

- Added user-facing troubleshooting guidance in `README.md` for fail-fast masked-loss routing errors:
  - symptom
  - why guard exists
  - safe resolution paths
  - quick validation checklist and focused pytest command
- Updated phase tracking docs:
  - `plans/2026-05-21_phase2_mask-gap-triage.md`
  - `plans/2026-05-21_phase2_mask-gap-triage.html`

Evidence commit:

- 1cebb7b docs: add fail-fast masked-loss troubleshooting and mark phase2 item5 complete

## Validation Evidence

Final non-training Phase 2 gate command:

```bash
.venv/bin/python -m pytest -q \
tests/test_losses.py \
tests/test_masking_sampling.py \
tests/test_loss_cluster_aware.py \
tests/test_loss_cluster_aware_extra.py \
tests/test_compute_masked_loss_routing.py \
tests/test_mask_semantics_contract.py \
tests/test_train_epoch_smoke.py \
tests/test_mask_contract_assertions.py \
tests/test_masked_loss_dispatch_matrix.py \
tests/test_qc_overhead_budget.py
```

Result:

- 57 passed in 3.85s

Operational constraint status:

- A2 training run remained undisturbed during Phase 2 implementation and validation.

## Residual Risks

1. Runtime signature inspection still exists in dispatcher logic; while fail-fast coverage is stronger, implicit dynamic routing remains a maintenance risk.
2. QC overhead budget warning is based on observed timing and can vary by hardware/load; thresholds may require tuning per environment.
3. Structural custom losses still require explicit valid_mask integration by implementers; docs now guide this, but enforcement remains at runtime.

## Phase 3 Recommendations

1. Replace signature inspection with explicit criterion capability contracts where practical.
2. Add a lightweight criterion-registration matrix for supported routing modes.
3. Add environment-profiled QC overhead thresholds (CPU/MPS/CUDA presets).
4. Add a short contract reference section in any internal contributor guide to reduce semantic drift.

## Completion Decision

Phase 2 goals are met. Proceed to Phase 3 contract simplification and maintainability hardening with the current safety and test baseline preserved.
