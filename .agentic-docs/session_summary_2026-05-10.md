Session summary — 2026-05-10

Overview
- Focus: diagnose and fix checkerboard/grid artifacts produced by `ConvTranspose3d` in the 3D U‑Net decoder; implement minimal, low-risk architectural changes and tests.

What I changed/confirmed so far
- Investigated masking API and cluster-aware loss integration; aligned `train.py` and `train_multi_datasets.sh` to the refactored masking API.
- Added `CompositeClusterAwareLoss` and ensured training-time calls pass full tensors to losses that accept a `valid_mask` kwarg.
- Added startup feedback to indicate when cluster-aware loss is enabled.

Next goals (immediate)
- Replace `ConvTranspose3d` upsampling in the decoder with `Upsample(scale_factor=2, mode="trilinear", align_corners=False) -> Conv3d(kernel_size=3, padding=1)`.
- Add decoder-only unit tests (smooth input, identity, decoder-only random input, skip alignment).
- Run the decoder tests locally; iterate until artifacts are eliminated.

What remains for the user to do locally
- Create and switch to a new branch and run the commit/push commands (instructions provided in `.agentic-docs/branch_and_commit_instructions.txt`).
- Start a fresh agentic session after pushing so the new branch and commits are in the clear context.

Notes
- The planned change is low-risk; weights adapt quickly and retraining from scratch is not required.
- Optional enhancements: add light anti-aliasing (AvgPool3d) and a small TV/Laplacian regularizer if residual grid remnants persist.

Status: beginning Step 3a (session summary + branch prep).

---

Session update — 2026-05-14

Overview
- Focus: add configurable residual block family (`resblock` vs `anisotropic`) and simple Gaussian mask infill while preserving current defaults.

Phase 1 — Model block selection (completed)
- Purpose: keep `ResBlock3d` as default while introducing `AnisotropicResBlock3d` for controlled experiments.
- Method:
	- Added `AnisotropicResBlock3d` in `src/synthoseis_pre_train/models.py`.
	- Added `_build_residual_block` factory and threaded `block_type` through encoder, decoder, `SeismicUNet3d`, and `create_model`.
	- Kept `resblock` as the default.
- Tests:
	- `tests/test_decoder_upsample.py`
	- `tests/test_model_block_types.py`
	- User-run result on Mac mini: 6 passed.

Phase 2 — Training CLI and launcher wiring (completed)
- Purpose: expose a parameter-driven block choice and mask infill controls from CLI/script without changing defaults.
- Method:
	- Added to `train.py`:
		- `--block_type/--block-type` (`resblock|anisotropic`, default `resblock`)
		- `--mask_fill_method/--mask-fill-method` (`zero|gaussian`, default `zero`)
		- `--mask_noise_std/--mask-noise-std` (default `1e-2`)
	- Added same controls to `train_multi_datasets.sh` and forwarded them to `train.py`.
	- Updated startup logging to print selected block family and masking infill settings.
	- Removed unintended duplicated trailing content from `train_multi_datasets.sh`.
- Tests:
	- Static/editor diagnostics: no syntax or lint errors in updated files.

Phase 3 — Gaussian mask infill implementation (completed)
- Purpose: add simplest optional infill strategy for masked voxels using zero-mean Gaussian noise with user-defined std.
- Method:
	- Extended `apply_mask_to_seismic` in `src/synthoseis_pre_train/masking.py` with:
		- `fill_method` (`zero` or `gaussian`)
		- `noise_std` (`>=0`)
	- Preserved legacy behavior: default remains zero-fill.
	- Threaded mask configuration through `SeismicDataset` in `src/synthoseis_pre_train/dataloader.py`.
	- Updated dataloader calls to `create_mask_3d` and `apply_mask_to_seismic` with explicit kwargs.
- Tests:
	- Added/updated masking and dataloader unit tests:
		- `tests/test_masking_sampling.py` (new infill behavior and validation tests)
		- `tests/test_quantile_transforms.py` (monkeypatch signature updates)
		- `tests/test_dataloader_non_cubic_shape.py` (monkeypatch signature updates)
	- Static/editor diagnostics: no syntax or lint errors in updated files.

Finalization status
- Code changes for requested features are implemented.
- Remaining: run full targeted pytest on Mac mini and perform final end-to-end review before commit.