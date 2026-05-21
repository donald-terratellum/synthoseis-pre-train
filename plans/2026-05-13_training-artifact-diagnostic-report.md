# Training Quality Diagnostic Report
## Observed Symptoms
- Vertical stripe artifacts — fully masked traces are predicted as near-zero columns rather than being interpolated from neighboring traces.
- No inter-extrema waveform reconstruction — the region between preserved peaks and troughs is predicted as flat rather than as the wavelet shape connecting them.
- Generalizing: the model has learned to copy what it can see and output zero everywhere else. It has not learned to interpolate spatially or extrapolate waveform shape along Z.

## Root Cause Analysis
### Cause 1 — Loss is computed only on masked voxels
In train.py::_compute_masked_loss:
```python
valid_mask = (~mask).to(dtype=output.dtype)   # 1 = masked
```

The model receives a gradient signal only at the voxels it cannot see. Preserved peaks and troughs — the only anchor the model has — receive zero gradient. The model never learns the amplitude relationship between a known peak and the voxels beside it.

### Cause 2 — Huber/MSE is purely voxelwise
The default loss (huber) treats every voxel independently. There is no penalty for a predicted trace that has the correct peak value but is flat between peaks, nor any penalty for a column that should smoothly continue a neighbor's waveform. The SSIM option exists in the codebase but is not the default.

### Cause 3 — Zero-fill is indistinguishable from real zero-crossings
apply_mask_to_seismic fills masked positions with 0.0. After standardization, seismic zero-crossings also have value ~0. The model cannot distinguish "I masked this" from "the actual seismic is zero here." It learns the safe strategy: output zero in doubt.

### Cause 4 — Full-trace masking dominates the sparse-input signal
create_mask_3d masks entire vertical columns (all Z at an X,Y position). At 15% masked fraction with 3×3 clusters, isolated traces deep inside a cluster have no immediate lateral neighbors with signal. For a 128³ volume with ~192px effective lateral span, the model needs a receptive field of ~5–10 traces just to "see around" a cluster center. The current encoder downsamples aggressively (128→64→32→16), which should produce a large effective receptive field, but without a gradient signal that rewards lateral coherence, the encoder never learns to use it for infilling.

### Cause 5 — XY adjacency dilation expands the dead zone
The _dilate_binary_2d(blank_xy, radius=1) call in dataloader.py::__getitem__ expands the zero-trace footprint by one pixel in all directions. This was intended to match the QC overlay logic, but it increases the effective masked area and widens the zone in which the model has no input signal.

## Recommended Updates (in priority order)

### Priority 1 — Switch to SSIM+MSE loss with a higher alpha
Why: The SSIM window spans 16 voxels in each dimension, so it directly penalizes predictions that are locally incoherent with neighbors. Huber cannot do this.

How: Re-launch (or resume) with:

```python
--loss_type ssim_mse --ssim_alpha 0.5 --ssim_window_size 11 --ssim_sigma 1.8 --ssim_data_range 6.0
```

The smaller window (11 vs 16) and tighter sigma (1.8 vs 2.67) better match the lateral correlation length of a seismic wavelet (~3–8 traces). data_range=6.0 matches a standardized volume with std=1 and range ≈ ±3σ.

### Priority 2 — Add a small full-volume auxiliary MSE term
Why: The model must learn to be consistent at preserved peaks, not just at masked locations. A small loss weight on all valid (non-squeeze-edge) voxels provides a gradient signal at the anchor points.

How: In _compute_masked_loss, after the existing masked loss, add:

```python
# Small aux term over all geometry-valid voxels (not just masked)
geom_valid = (target.abs() > 0).to(dtype=output.dtype)  # excludes squeeze-edge zeros
aux = 0.05 * F.mse_loss((output * geom_valid).flatten(), (target * geom_valid).flatten())
return masked_loss + aux
```

A weight of 0.05 is small enough not to shift the training objective but large enough to anchor waveform continuity.

### Priority 3 — Replace zero-fill with random noise fill
Why: Noisy fill breaks the zero-fill ambiguity. The model cannot cheaply output zeros for unknown positions because it cannot tell from the fill value alone which voxels to trust. This is the approach used in BERT and MAE.

How: In masking.py::apply_mask_to_seismic, change the default fill:

```python
def apply_mask_to_seismic(seismic_data, mask, fill_value="noise", noise_scale=0.1):
    masked_data = seismic_data.copy()
    if fill_value == "noise":
        sig = float(np.std(seismic_data[mask])) if mask.any() else 1.0
        masked_data[~mask] = np.random.normal(0.0, noise_scale * sig, (~mask).sum())
    else:
        masked_data[~mask] = float(fill_value)
    ...
```

Add a --mask_fill noise CLI flag to train.py so zero-fill can still be used when desired.

Note: this noise is injected after the sample has already been normalized or quantile-mapped in the dataloader, so `noise_scale=0.1` means roughly 0.1 of the post-transform standard deviation. I did not find a seismic paper that prescribes a universal 10x or 100x smaller fill-noise rule; MAE uses mask tokens with high masking ratios, BERT uses a masking scheme with learned masked representations, and stacked denoising autoencoders treat corruption level as a task choice rather than a fixed universal ratio.

References: [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377), [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805), [Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion](https://www.jmlr.org/papers/v11/vincent10a.html).

### Priority 4 — Add a spectral (frequency-domain) loss along Z
Why: Seismic waveforms have a characteristic frequency band (~10–80 Hz). A loss in frequency space directly penalizes the model for producing a flat trace between peaks, even if the peak amplitude is correct.

How: Add a new SpectralLoss3D class to losses.py:

```python
class SpectralLoss1D(nn.Module):
    """L2 loss on the power spectrum along the depth (z) axis."""
    def forward(self, pred, target, valid_mask=None):
        # pred/target: [B, C, D, H, W]
        P = torch.fft.rfft(pred.float(), dim=2)
        T = torch.fft.rfft(target.float(), dim=2)
        loss = ((P.real - T.real) ** 2 + (P.imag - T.imag) ** 2).mean()
        return loss
```

Use it as a third term in a composite loss at weight ~0.1. This is the single change with the highest expected impact on the between-peak artifact.

### Priority 5 — Reduce or split the masking strategy
Why: Full-trace masking at the current cluster sizes creates large dead zones. Mixing trace masking with depth-slice (Z-plane) masking gives the model more inter-extrema reconstruction tasks during training.

### How: Add a --mask_strategy flag:

- trace — current behavior (whole vertical columns)
- depth — mask random Z-planes across all traces (forces waveform interpolation along Z)
- mixed — 50% trace + 50% depth masking per sample
depth masking specifically targets the between-peak waveform problem because it always leaves lateral neighbors intact while removing the Z-axis continuity.

## Alternative Approaches
### A — Context encoder with visible-only input (MAE-style)
Instead of zero-filling masked positions, feed the model only the visible voxels using a sparse input representation. The encoder processes only unmasked tokens; the decoder attends to both encoder output and mask tokens to reconstruct. This is the architecture of MAE (He et al. 2021) and is why MAE works better than BERT-style zero-fill at reconstruction tasks. Requires architectural changes: the current U-Net would need an attention-based decoder or a mask-token mechanism.

### B — Diffusion-based inpainting as the pre-training objective
Frame reconstruction as denoising diffusion. The model learns a score function that can infill any masked region conditioned on the visible context. Naturally handles lateral coherence because the denoising score at any voxel is conditioned on neighboring values. Substantial rearchitecture of the training loop but no architectural change to the U-Net itself (the U-Net becomes the noise prediction network).

### C — Phase-preserving wavelet loss
Replace the spatial SSIM window with a 1D continuous wavelet transform (CWT) along Z, computing loss on the wavelet scalogram. This is directly matched to the seismic problem: it penalizes phase errors and amplitude errors at each frequency band independently. The CWT along Z can be implemented efficiently as a bank of Gabor-shaped Conv1D kernels.

### D — Curriculum masking
Start training with a low masked fraction (5%) and gradually increase to 30% over the first 50 epochs. This gives the model time to learn waveform shape at easy masking levels before it must handle the difficult full-trace-cluster regime. No code change is required beyond a --mask_fraction_schedule parameter that linearly interpolates target_masked_fraction.

## Recommended Immediate Action
For the current training run (already at epoch 25), the lowest-risk high-impact change is to restart with --loss_type ssim_mse --ssim_alpha 0.5 --ssim_window_size 11 --ssim_sigma 1.8 --ssim_data_range 6.0 using the current checkpoint as a warm start. The SSIM loss requires no code changes and directly addresses both the lateral stripe and inter-extrema artifacts. Priority 2 (auxiliary full-volume loss) is a small code change to _compute_masked_loss that can be added at the same time.












# Training Artifact Diagnostic Report

**Date:** 2026-05-13
**Scope:** current augmentation, masking, loss, and training design for `synthoseis-pre-train`

## Summary
The model is learning to reconstruct the visible peak and trough anchors, but it is not learning to fill the interior waveform structure or laterally interpolate masked traces with enough fidelity. The current design gives strong supervision only at masked voxels, uses zero-fill for missing context, and relies on a voxelwise objective by default. Those choices favor local copy/zero behavior over coherent seismic infilling.

## Observed Symptoms
- Vertical stripe artifacts in the predicted seismic output.
- Missing interior waveform structure between preserved peaks and troughs.
- Thick layers that do not retain the vertical amplitude pattern seen in the label volume.
- Predictions that appear concentrated on peak/trough values instead of interpolated or extrapolated seismic structure.

## Most Likely Causes
1. The training loss is applied only where voxels are masked, so preserved traces do not contribute a corrective gradient.
2. The default objective is voxelwise Huber or MSE, which does not directly reward local structural continuity.
3. Masked inputs are zero-filled, which makes masked regions look similar to natural zero-crossings after normalization.
4. The masking scheme removes entire traces, so the network must infer both vertical waveform shape and lateral continuity from relatively sparse context.
5. The model and augmentations are not explicitly optimized for interpolation across masked traces or for reconstructing between extrema along the Z axis.

## Recommended Updates
### 1. Keep the masking concept, but change the learning target
Preserve the current seismic masking idea, but do not train solely on masked voxels. Add a small auxiliary loss over all valid voxels so the model also learns to preserve known amplitude relationships and waveform continuity.

### 2. Make the default objective structural, not only pointwise
Prefer a mixed SSIM plus MSE objective, or a similar structural loss, instead of pure voxelwise loss. This gives the model a direct incentive to reconstruct local continuity, not just amplitude values at isolated points.

### 3. Stop using plain zero-fill as the only masked-value signal
Replace deterministic zero-fill with a non-ambiguous fill strategy such as noise, learned mask tokens, or a separate mask channel. That lets the network distinguish true seismic zeros from unknown voxels.

### 4. Add an explicit waveform-oriented auxiliary objective
Introduce a frequency-domain or 1D trace-shape loss along Z so the model is penalized when it predicts flat interiors between peaks and troughs.

### 5. Reduce the difficulty gap in the masking curriculum
Start with easier masking ratios or mixed masking patterns, then gradually increase trace-cluster density. This gives the network a chance to learn waveform shape before handling the hardest holes.

## Alternative Approaches
### A. Mask-token or MAE-style architecture
Feed the model visible voxels plus explicit mask tokens instead of zero-filled inputs. This is the cleanest way to force the network to learn inpainting rather than copying zeros.

### B. Diffusion-based inpainting
Use a diffusion or denoising objective for reconstruction. This is a larger change, but it is often better at generating smooth, context-aware infill.

### C. Wavelet or spectral loss along Z
Use a frequency-sensitive auxiliary loss so the network is rewarded for preserving seismic character between extrema.

### D. Curriculum masking
Train first on sparse masking and later on clustered trace masking. This is a lower-risk change that can improve convergence without changing the model architecture.

## Recommended Plan
1. Keep the current masking workflow, but add a nonzero auxiliary loss over visible voxels.
2. Switch the default reconstruction loss from pure voxelwise loss to a structural mixed loss.
3. Replace zero-fill with a fill strategy that cannot be confused with a real seismic zero.
4. Add a small trace-shape or spectral auxiliary term if the artifact persists.
5. If the results still collapse toward peak/trough-only prediction, move to a mask-token architecture or diffusion-based inpainting.

## Risk Notes
- Increasing the loss too aggressively on visible voxels can reduce the model's incentive to infer missing regions, so the auxiliary term should stay small.
- Structural losses can improve visual continuity but may hide over-smoothing if used alone, so they should remain mixed with a pointwise term.
- Changing the input fill strategy can alter convergence behavior, so it should be introduced with a clear ablation test.

## Status
This report captures the earlier diagnosis and is suitable for future implementation planning or follow-up review.
