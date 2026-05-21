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
- Otherwise, fallback path uses output[~mask], target[~mask] boolean indexing.

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

Observed risk:

- The fallback boolean-indexing path can flatten structure and may be unsafe for structural criteria that do not expose valid_mask.

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

Observed risk:

- Per-batch QC block in train.py recomputes extra masked losses for logging, increasing compute overhead inside training loop.

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

Divergence:

1. Runtime signature inspection plus fallback indexing in _compute_masked_loss is less explicit than MONAI-style fixed loss contracts.
2. QC metrics recomputation in train loop is more coupled than typical train-vs-metric separation patterns.
3. Semantic inversion at train boundary is implicit and should be formalized.

Phase 1 recommendations:

1. Keep a single explicit mask contract section in docs and tests.
2. Add fail-fast handling for unsupported criterion signatures.
3. Preserve diagnostics but move expensive QC recomputation out of batch hot path where feasible.

## Immediate Phase 1 Deliverables Status

- Mask usage overall mapping: complete.
- Loss-path mapping: complete.
- Gradient/backprop mapping: complete.
- Diagnostics/QC-only mapping: complete.
- MONAI API/pattern comparison notes: complete.

## Next Phase 1 Actions

1. Add tests/test_compute_masked_loss_routing.py for dispatcher behavior and fail-fast checks.
2. Add tests/test_mask_semantics_contract.py for end-to-end mask meaning consistency.
3. Run targeted non-training test gate.
4. Commit docs and new tests as Phase 1 checkpoint.
