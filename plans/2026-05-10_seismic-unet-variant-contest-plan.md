# Seismic U-Net Variant Contest Plan

Intended attachment prompt: follow the attached plan.

Date: 2026-05-10
Scope: src/synthoseis_pre_train/models.py, train.py, tests/, .agentic-docs/, plans/
Goal: evaluate three seismic-specific U-Net variants against the current implementation and merge only the clear winner that also passes explicit safety and quality gates.

## Context

Current model facts:
- The current model is a 3D U-Net with 2-conv residual blocks, InstanceNorm3d, GELU, residual projection when channels change, skip-connected decoder, and a linear reconstruction head.
- Decoder upsampling was already improved to trilinear upsample plus Conv3d to avoid known grid-like artifacts from ConvTranspose3d.
- Encoder downsampling was already improved from kernel-2 stride-2 to kernel-3 stride-2 padding-1 to reduce aliasing.
- A pre-head InstanceNorm3d plus GELU block was added before the reconstruction head to stabilize masked reconstruction.

Seismic-specific design guidance from the discussion:
- Seismic is not isotropic in the same way as many medical volumes. The vertical axis often encodes wavelet, stratigraphic, and time-depth structure that differs from the lateral axes.
- Early layers should therefore be allowed to treat vertical and lateral structure differently.
- Deeper layers still need true 3D context because dipping reflectors, faults, and other structures re-orient across axes.
- Published 3D seismic and transferable 2D-to-3D designs usually favor moderate residual blocks, multiscale encoders, anisotropic early processing, and deeper context at coarse scales rather than extremely long residual blocks.

Recommended ranking of optional changes:
1. Most impactful: anisotropic early encoder processing, then isotropic deeper processing.
2. Next: more capacity at the deepest encoder stage and bottleneck, not everywhere.
3. Next: lightweight channel attention at deeper stages.

This plan treats the current implementation as the control and evaluates three unique variants in separate branches.

## Baseline And Contest Rules

Control:
- The current implementation at the contest start commit is the control.
- Tag the control commit or create a short-lived baseline branch before any variant work begins.

Required training schedule for all candidates and the control:
- 15 epochs
- 200 train batches per epoch
- batch size 3
- same data paths
- same validation split and validation batch schedule
- same loss configuration
- same checkpoint and output naming rules, with separate output directories per run

Required fairness controls:
- Use the same training command for all runs except for branch name and output directory.
- Record the exact commit SHA for each run.
- If no fixed random seed exists yet, add one in Phase 0 and use the same seed for all runs.
- If per-run metric capture is incomplete, add it in Phase 0 before any branch contest work.

Metrics to capture for every run:
- best validation total loss
- final validation total loss
- validation SSIM metric or SSIM-derived reconstruction score
- validation MSE
- epoch time statistics
- peak memory or at least whether the run completed without OOM
- qualitative artifact review on a fixed set of saved slices or inference examples

## Candidate Branches

### Variant A: anisotropic early encoder, isotropic deeper network

Suggested branch: feat/unet-aniso-early

Intent:
- Reflect seismic vertical-vs-lateral asymmetry in the earliest stages.
- Recover full 3D context in deeper stages where dips and faults re-orient structure.

Implementation target:
- Make the stem and first encoder stage anisotropic.
- Prefer kernels such as 1x3x3 or 3x3x1 in the earliest stage rather than fully isotropic 3x3x3 everywhere.
- Consider stride 1x2x2 in the first downsample, then return to 2x2x2-equivalent behavior deeper in the encoder.
- Keep the decoder isotropic unless a concrete shape or artifact reason appears during review.

Reason for rank:
- This is the most seismic-specific architectural change with the strongest physical justification.

### Variant B: deeper deep-stage residual capacity

Suggested branch: feat/unet-deep-context

Intent:
- Increase representational power where the receptive field is already large and the memory cost is lower.

Implementation target:
- Add one extra residual block at the deepest encoder stage and one extra residual block at the bottleneck.
- If needed, mirror one extra residual block in the deepest decoder stage for symmetry.
- Do not deepen all stages uniformly.

Reason for rank:
- Published 3D segmentation and seismic systems often add depth at coarse scales rather than making every residual block longer.

### Variant C: channel attention at deeper stages

Suggested branch: feat/unet-se-attention

Intent:
- Reweight feature channels adaptively after local feature extraction without a large parameter increase.

Implementation target:
- Add a lightweight squeeze-excitation style block after the second norm in ResBlock3d.
- Apply it only to deeper stages by default, for example channels at or above 128, unless profiling shows the full-network version is still cheap.
- Keep the residual block otherwise close to the current implementation.

Reason for rank:
- This is a low-risk, low-parameter modification that often improves reconstruction or segmentation quality when feature relevance varies strongly by scale.

## Phases

Every phase below must end with all of the following:
- code written or updated
- code review completed
- relevant tests passing
- git commit created
- session summary file updated

Session summary update rule:
- Update .agentic-docs/session_summary_2026-05-10.md if it remains the active session summary.
- If a new date-specific summary is more appropriate, create a new .agentic-docs markdown summary for that date and reference this plan.

### Phase 0: lock the baseline and evaluation harness

Goal:
- Make the contest reproducible before variant work starts.

Required work:
- Lock the baseline commit on main.
- Add or verify fixed-seed support if it is missing.
- Add or verify comparable validation metric logging for SSIM and MSE if it is missing.
- Add or verify a stable report-friendly output format for metrics per run.
- Add or verify a fixed qualitative review set of slices or inference samples.

Required review points:
- No variant-specific logic is introduced in the control path.
- Metric names and file locations are consistent across all runs.
- The training command can be copied and reused across branches.

Required tests:
- Existing unit tests pass.
- Any new metric or seed tests pass.

Required git artifact:
- Commit the baseline-lock and evaluation-harness work before branching into variants.

### Phase 1: implement Variant A on its own branch

Goal:
- Produce a clean anisotropic-early candidate that is isolated from other optional changes.

Required work:
- Create branch feat/unet-aniso-early from the locked baseline.
- Implement only the anisotropic-early variant.
- Add or update tests for tensor shapes, skip alignment, and forward stability.
- Run a focused code review on axis ordering, kernel shapes, stride behavior, and compatibility with existing inference and checkpoint paths.

Required tests:
- Relevant model tests pass.
- Full targeted pytest selection passes.

Required git artifact:
- Commit the variant and test changes on feat/unet-aniso-early.

Required session summary update:
- Record exactly what changed, why it is isolated, and how it should be trained.

### Phase 2: implement Variant B on its own branch

Goal:
- Produce a clean deeper-deep-stage candidate that is isolated from other optional changes.

Required work:
- Create branch feat/unet-deep-context from the locked baseline.
- Implement only the deep-stage capacity variant.
- Add or update tests for parameter counts, tensor shapes, and forward stability.
- Run a focused code review on block placement, checkpoint compatibility, and memory implications.

Required tests:
- Relevant model tests pass.
- Full targeted pytest selection passes.

Required git artifact:
- Commit the variant and test changes on feat/unet-deep-context.

Required session summary update:
- Record exactly what changed, why it is isolated, and the expected tradeoff versus the control.

### Phase 3: implement Variant C on its own branch

Goal:
- Produce a clean attention candidate that is isolated from other optional changes.

Required work:
- Create branch feat/unet-se-attention from the locked baseline.
- Implement only the channel-attention variant.
- Add or update tests for forward stability, parameter changes, and block behavior.
- Run a focused code review on attention placement, activation ordering, and extra compute cost.

Required tests:
- Relevant model tests pass.
- Full targeted pytest selection passes.

Required git artifact:
- Commit the variant and test changes on feat/unet-se-attention.

Required session summary update:
- Record exactly what changed, why it is isolated, and the expected tradeoff versus the control.

### Phase 4: run the contest and generate the report

Goal:
- Train the control and all three variants under the same schedule and produce one comparable report.

Required work:
- Run the control and all three variant branches for 15 epochs, 200 train batches per epoch, batch size 3.
- Use separate output directories and preserve logs for every run.
- Capture the exact training command, commit SHA, wall-clock timing, and metrics for every run.
- Generate the report at plans/2026-05-10_seismic-unet-variant-contest-report.md.

Report requirements:
- State the control commit and all variant branch commits.
- Show the exact command used for each run.
- Show a table for best val loss, final val loss, val SSIM, val MSE, epoch time, and completion status.
- Include a short qualitative artifact review section using the same fixed slices or inference samples for every run.
- Name one provisional winner and explain why.
- State explicitly whether any candidate failed a merge gate.

Required review points:
- Confirm that the same data, schedule, and metric definitions were used everywhere.
- Confirm that no run used a stale checkpoint from another branch.
- Confirm that the report tables match the saved logs.

Required tests:
- Re-run the relevant test suite on the branch used to generate or finalize the report tooling.

Required git artifact:
- Commit the report and any supporting metric-capture code.

Required session summary update:
- Record all run outcomes, failures, and the provisional winner.

### Phase 5: pick the winner and prepare the merge candidate

Goal:
- Select the winner against the other candidates and against the current implementation, then prepare one final merge-ready branch.

Winner selection rule:
- A candidate must beat the control and the other two variants on the primary decision metric and pass every absolute gate below.
- If the primary metric is too close to call, break ties using SSIM, MSE, stability, and qualitative artifact review, in that order.
- If no candidate passes the gates, do not merge any variant.

Absolute merge gates:
- 15 of 15 epochs complete successfully.
- 200 train batches per epoch used for the full run.
- Batch size 3 used for the full run.
- No NaN or Inf losses.
- No OOM or unrecovered runtime failure.
- Relevant tests pass on the winning branch.
- Mean epoch time is not more than 20 percent slower than the control.
- Peak memory is not more than 15 percent above the control, or the run must still fit comfortably on the target hardware without degraded batch size.
- Qualitative artifact review is no worse than the control.

Performance gates relative to the control:
- Best validation total loss improves by at least 2 percent.
- Validation SSIM is not worse than the control by more than 0.005.
- Validation MSE improves by at least 2 percent.

Preferred winner profile:
- Best validation total loss among all candidates.
- Validation SSIM improvement of at least 0.005 over the control.
- Validation MSE improvement of at least 5 percent over the control.

Required work:
- Create or update a merge-candidate branch from the winning variant.
- Perform an end-to-end code review across architecture, tests, metrics, and report artifacts.
- Re-run the full relevant tests.
- Update the contest report with the final winner decision.

Required git artifact:
- Commit the final merge-candidate state.

Required session summary update:
- Record the final winner, the evidence for the decision, and any deferred follow-up work.

## Contest Run Instructions

Run location:
- Run training on the Mac mini in /Users/donaldpg/synthoseis-pre-train.

Run procedure:
1. Check out the control commit or branch.
2. Run the locked training command with epochs 15, train_batches_per_epoch 200, and batch_size 3.
3. Save logs and outputs to a control-specific output directory.
4. Repeat for feat/unet-aniso-early, feat/unet-deep-context, and feat/unet-se-attention.
5. Do not change any hyperparameter other than branch-specific code.
6. Copy the exact command lines and commit SHAs into the contest report.

Command requirements:
- Use uv run python train.py.
- Use the same data path list, loss settings, validation settings, and checkpoint cadence across all runs.
- If the current working command includes SSIM, masking, or cluster-aware flags, keep them unchanged for every contestant.

## What Was Missing From The Original Request

The original request was directionally clear, but two items needed to be made explicit for a high-confidence contest:
- a locked baseline plus reproducibility rules
- a minimal evaluation harness for comparable metrics and qualitative review

Without those, the branch contest could produce a winner that is not actually comparable to the current implementation.

## Confidence

This plan is high confidence for process and decision quality.

One caution remains:
- raw absolute performance thresholds such as SSIM greater than a fixed global number are not high confidence without a locked dataset distribution and known baseline scale.

For that reason, this plan uses:
- absolute operational gates where they are reliable
- relative performance gates against the current implementation where the metric scale is dataset-dependent

That is the safest merge policy for this repository.
