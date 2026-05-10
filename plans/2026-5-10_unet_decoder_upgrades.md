Seismic 3D U‑Net Reconstruction — Decoder Artifact Diagnosis & Fix Plan
1. Problem Summary
I am training a 3D U‑Net for seismic reconstruction using synthetic seismic volumes where the input contains only local Z‑axis extrema (peaks & troughs) and the network reconstructs the full seismic.

During inference on validation datasets, the output exhibits a fine‑scale, nearly regular mesh pattern — visually similar to viewing seismic through a screen door.
The artifact is most visible in smooth regions (e.g., inline 80–120, depth 80–90).

This artifact is not present in the training data and appears consistently across validation volumes.

I need the agent to analyze the architecture and training code and propose the minimal, correct fix.

2. Context
Model
The decoder uses:

python
nn.ConvTranspose3d(deep_ch, skip_ch, kernel_size=2, stride=2, bias=False)
This is the classic configuration that produces checkerboard / grid artifacts in 2D and 3D U‑Nets due to uneven overlap during transposed convolution upsampling.

Data
Input: sparse seismic (only extrema retained)

Target: full seismic

Loss: SSIM + MSE composite

Training: 3D volumes, no patch tiling during training

Observation
The artifact:

Is global, not patch‑boundary‑aligned

Has regular periodicity

Appears after upsampling stages

Is amplified by the sparse Z‑axis input representation

This strongly implicates the ConvTranspose3d upsampling.

3. Root Cause (What the agent should confirm)
✔ Primary cause: ConvTranspose3d checkerboard artifacts
ConvTranspose3d with kernel_size=2, stride=2 produces uneven overlap when reconstructing higher‑resolution feature maps.
This creates a periodic modulation pattern that appears as a mesh.

This is a well‑documented issue in U‑Nets for medical imaging and seismic.

✔ Why it’s amplified here
Sparse Z‑axis input → network must interpolate aggressively → any architectural aliasing becomes visible.

3a. Preliminary work
[ ] write a session summary for the current session in .agentic-docs
    - follow the format, style, content, sections, conciseness level of the previous 2 session summaries
[ ] commit code changes to git and push to github
[ ] tell the user when to start a new agentic session so the context window is cleared
[ ] create a new branch. change to the new branch for code modifications described below. use a meaningful name for the new branch
[ ] review the plan in this markdown file. write a revised version if appropriate to make it more concise, to raise confidence that the modifications will be successful, and/or to use fewer tokens.

4. Required Modification (What the agent should implement)
Replace ConvTranspose3d with:
Code
Upsample (trilinear) → Conv3d
Target decoder change
Replace:

python
nn.ConvTranspose3d(deep_ch, skip_ch, kernel_size=2, stride=2, bias=False)
With:

python
nn.Sequential(
    nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
    nn.Conv3d(deep_ch, skip_ch, kernel_size=3, padding=1, bias=False)
)
Why this works
Removes uneven overlap

Produces smooth, artifact‑free upsampling

Standard best practice in modern 3D U‑Nets

Does not require retraining from scratch (weights adapt quickly)

5. Additional Improvements (Optional but recommended)
A. Add light anti‑aliasing
After upsampling:

python
nn.AvgPool3d(kernel_size=3, stride=1, padding=1)
B. Add small smoothness regularizer
Total variation (TV) loss or Laplacian penalty:

Helps suppress residual grid noise

Does not blur structural features

C. Ensure consistent padding
All convs should use:

python
padding=1
to maintain alignment with skip connections.

6. Self‑Review Checklist for the Agent
The agent should verify:

Decoder architecture
[ ] No remaining ConvTranspose3d layers

[ ] All upsampling uses Upsample → Conv3d

[ ] Skip connection shapes match exactly

[ ] No asymmetric padding or cropping

Training pipeline
[ ] Loss computed on full‑resolution output

[ ] No unintended downsampling/upsampling in dataloader

[ ] No patch tiling during inference (unless blended)

Numerical stability
[ ] InstanceNorm3d is stable with batch size used

[ ] autocast + FP32 loss path is correct

7. Tests the Agent Should Write
Test 1 — Smooth Input Test
Feed a smooth 3D Gaussian blob through the model:

Before fix → grid artifact visible

After fix → no grid, smooth output

Test 2 — Identity Test
Train the model on identity mapping for a few batches:

Input = target

Output should converge to clean identity

Any grid pattern indicates architectural aliasing

Test 3 — Decoder‑Only Test
Run only the decoder on random tensors:

Visualize slices

Confirm no periodic modulation

Test 4 — Skip Alignment Test
Assert:

python
assert x.shape == skip.shape
after each upsample.

8. What I Want the Agent to Do
Analyze the decoder code and confirm the checkerboard cause.

Rewrite the decoder using Upsample → Conv3d.

Ensure skip connections remain correct.

Provide the updated decoder code.

Provide the tests listed above.

Explain how to integrate the fix into the existing training pipeline.