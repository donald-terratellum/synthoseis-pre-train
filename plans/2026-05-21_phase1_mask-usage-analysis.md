# Phase 1 Mask Usage Analysis

Date: 2026-05-21
Branch: feat/masking-loss-reevaluation
Scope: Training-path mask behavior mapping while A2 ablation remains active.

## Executive Summary

The current training pipeline applies masking in a way that directly affects optimization, not only reporting. Mask generation and union occur in the dataloader, mask inversion occurs at the training loss boundary, and most custom losses consume full-shape valid masks. A fallback path exists that uses boolean indexing for criteria without a valid_mask argument. Diagnostics and QC code currently recompute several masked losses per batch and should remain separated from the hot training path.

## A. Mask Usage Overall

Primary flow:

1. Mask creation starts in src/synthoseis_pre_train/masking.py via create_mask_3d.
2. Semantics at source are explicit: True means preserve, False means masked in apply_mask_to_seismic.
3. Dataloader applies masking to model input only and keeps target unmasked.
4. Dataloader optional expansion path dilates blank-trace adjacency and reapplies masking.
5. Final dataloader mask is a conjunction of trace_mask and geom_mask.
6. Training receives mask tensor and passes it into _compute_masked_loss.

Key implementation points:

- src/synthoseis_pre_train/dataloader.py creates trace_mask using create_mask_3d and applies it with apply_mask_to_seismic.
- src/synthoseis_pre_train/dataloader.py optionally performs enable_cluster_mask_expansion logic and applies masking again.
- src/synthoseis_pre_train/dataloader.py combines trace_mask and geom_mask into final mask.
- train.py consumes this mask in train and validation loops.

Observed risk:

- A dataloader comment says mask=True means visible/included in loss, while training inverts mask before criterion call. This semantic boundary must be documented and enforced to avoid accidental misuse.

## B. Mask Usage In Loss Computation

Central dispatcher behavior:

- train.py::_compute_masked_loss defines valid_mask as inverse of incoming mask.
- If criterion forward includes valid_mask, full-shape tensors are passed.
- Otherwise, fallback path is now restricted to pointwise criteria only
   (MSE/L1/Huber/SmoothL1) or explicit opt-in via allow_mask_indexing=True.
- Unsupported criteria now fail fast with a clear ValueError instead of silent
   flattened fallback.

Loss modules with explicit valid_mask support:

1. SSIMMSELoss3D in src/synthoseis_pre_train/losses.py
   - Accepts valid_mask.
   - Validates shape and applies masked weighting in reductions.

2. MONAIStyleSSIMMSELoss3D in src/synthoseis_pre_train/losses.py
   - Accepts valid_mask.
   - Applies valid_mask in local-window computations.

3. SlidingWindowStatsLoss3D in src/synthoseis_pre_train/losses.py
   - Accepts valid_mask.
   - Uses masked pooling/statistics unless apply_to_all_voxels is enabled.

4. CompositeClusterAwareLoss in src/synthoseis_pre_train/losses.py
   - Accepts valid_mask.
   - Splits regions into base and cluster masks by masking density.
   - Calls base criterion twice with derived masks.

Boundary cleanup result:

- Fallback flattening remains available for safe pointwise criteria.
- Structural or custom criteria without valid_mask support now fail early.
- New routing tests cover valid_mask path, safe fallback path, and fail-fast path.

## C. Mask Usage In Gradient Backpropagation

Training path:

1. train.py computes masked loss.
2. train.py performs backward on scaled loss.
3. Gradients are clipped and optimizer step executes on accumulation boundary.

Impact:

- Because losses apply valid_mask-weighted reductions, masked-out regions do not contribute to gradient terms where valid_mask is zero.
- Composite cluster-aware logic redistributes weighting between base and cluster regions, changing gradient emphasis spatially.

Validation path:

- validate runs under no_grad and uses _compute_masked_loss for evaluation only.

Boundary cleanup result:

- Per-batch QC metrics are now disabled by default.
- QC metrics run only when --enable_batch_qc_metrics is set.
- QC cadence is controlled by --batch_qc_every.
- QC metrics are computed under no-grad and criteria are instantiated once per epoch.

## D. Mask Usage Restricted To Printed Statistics, Diagnostics, QC

Diagnostics-only usage currently observed:

1. train.py QC block prints masked metric values per batch using extra criterion instances.
2. train.py logs periodic figures and non-zero statistics.
3. src/synthoseis_pre_train/plotting.py consumes arrays/volumes only and does not consume mask tensors directly.

Boundary note:

- Diagnostics and plotting operate mostly outside backward via detach or no_grad contexts, but QC recomputation still occurs in the training loop and should be treated as hot-path overhead.

## E. MONAI 1.0.0 Pattern Comparison (API/Pattern Level)

Alignment:

1. The repository uses full-shape mask-aware loss signatures for major custom losses.
2. Reduction and masking are handled inside loss modules, similar to mask-aware wrapper style.

Divergence (updated):

1. Runtime signature inspection remains in use, but unsupported routing is now fail-fast and safer.
2. QC recomputation is now opt-in and interval-gated, reducing coupling to the hot path.
3. Semantic inversion at train boundary is implicit and should be formalized.

Phase 1 recommendations (remaining):

1. Keep a single explicit mask contract section in docs and tests.
2. Keep fail-fast routing tests in the required gate.
3. Monitor QC runtime overhead only when the opt-in flag is enabled.

## Immediate Phase 1 Deliverables Status

- Mask usage overall mapping: complete.
- Loss-path mapping: complete.
- Gradient/backprop mapping: complete.
- Diagnostics/QC-only mapping: complete.
- MONAI API/pattern comparison notes: complete.

## Phase 1 Implemented Changes

1. Added tests/test_compute_masked_loss_routing.py.
2. Added tests/test_mask_semantics_contract.py.
3. Hardened train.py::_compute_masked_loss with explicit fail-fast behavior for unsupported criteria.
4. Decoupled per-batch QC diagnostics in train.py by adding:
   - --enable_batch_qc_metrics (default off)
   - --batch_qc_every (interval control)
5. Added QC-enabled smoke coverage in tests/test_train_epoch_smoke.py.

## Phase 1/2 Boundary Checklist

- [x] Phase 1 mapping docs complete (md/html in plans).
- [x] Mask routing and semantics tests added.
- [x] Focused tests for routing/semantics/smoke passed.
- [x] Fail-fast routing behavior implemented.
- [x] Batch QC recomputation decoupled from default hot path.
- [x] Run full targeted non-training suite after latest train.py changes.
- [x] Commit boundary-cleanup documentation update.

## Next Actions (Phase 2 Entry)

1. Begin Phase 2 gap-triage updates from MONAI comparison findings.
2. Implement high-priority contract hardening items first.
3. Keep A2 training run undisturbed while landing non-training-safe fixes.
