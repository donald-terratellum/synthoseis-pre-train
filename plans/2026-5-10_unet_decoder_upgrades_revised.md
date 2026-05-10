UNet Decoder Upgrade — concise plan

Goal
Replace transposed-convolution upsampling with artifact-free upsampling in the decoder.

Steps
1. Locate decoder module(s) that use `nn.ConvTranspose3d(..., kernel_size=2, stride=2)`.
2. Replace each with:
   nn.Sequential(
       nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
       nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
   )
3. Ensure skip-connection shapes remain identical after the change.
4. Add tests:
   - Smooth Input Test (3D gaussian) before/after
   - Identity Test (input==target)
   - Decoder-only test (random tensors)
   - Skip alignment assertions
5. Optional: add `nn.AvgPool3d(kernel_size=3, stride=1, padding=1)` after upsample conv and/or a small TV/Laplacian penalty.
6. Run unit tests and a short validation run; iterate if artifacts persist.

Branch name suggestion: feat/unet-decoder-fix

Rationale
- Upsample+Conv3d removes uneven overlap that causes checkerboard artifacts.
- Minimal code changes; model weights adapt without full re-train.

Acceptance
- No `ConvTranspose3d` left in the decoder.
- Decoder outputs on smooth inputs show no periodic mesh.
- Skip shapes match exactly.
