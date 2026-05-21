# Copilot Ablation Plan (Formatted)

## Scope
This document contains a concrete run sheet and low-risk implementation plan for seismic reconstruction training on a Mac mini home lab.

## Assumptions
1. 24 GB unified memory on MPS.
2. 8 active zarr datasets, each approximately 300x300x1500.
3. Dynamic refresh policy: add 2 datasets and remove 2 datasets each epoch.
4. Training crops are 128x128x128.
5. Goal: reduce mean drift and recover label-level variance.
6. Phase A through Phase C are runnable with current [train.py](../train.py).
7. Phase D requires small code additions in [train.py](../train.py) and [src/synthoseis_pre_train/models.py](../src/synthoseis_pre_train/models.py).

## Common Setup
```bash
export DATA_FOLDER=/path/to/your/zarr/root
export RUN_ROOT=checkpoints/ablation_2026_05_20
mkdir -p "$RUN_ROOT"
```

### Recommended Base Knobs
1. `batch_size=2`
2. `grad_accum_steps=16`
3. Effective batch size `= 32`
4. `train_batches_per_epoch=64..80`
5. `val_batches_per_epoch=24..32`

Reason: improves optimization stability without exceeding memory, while sampling across rotating datasets.

## Phase A: Locked Baseline + LR Bracket
### Goals
1. Establish baseline under fixed normalization and range assumptions.
2. Compare `lr=5e-5` vs `lr=1e-4` with no other changes.

### A1 Baseline (`lr=5e-5`)
```bash
python train.py --data_folder "$DATA_FOLDER" --dataset_glob "seismic__*/model_data.zarr" --output_dir "$RUN_ROOT/A1_lr5e5" --device mps --sample_shape 128 128 128 --batch_size 2 --grad_accum_steps 16 --epochs 20 --val_split_ratio 0.25 --train_batches_per_epoch 64 --val_batches_per_epoch 24 --amplitude_transform histogram_equalization --transforms_group transforms_histeq_lock --loss_type ssim_mse --ssim_data_range 30.0 --ssim_window_size 16 --ssim_sigma 2.6667 --ssim_alpha 0.1667 --ssim_min_valid_ratio 0.5 --target_masked_fraction 0.15 --cluster_shape 3 --center_selection_method random_mixture --mask_fill_method zero --lr 5e-5 --lr_schedule poly --lr_poly_power 0.9 --lr_min 1e-6 --lr_warmup_epochs 3 --lr_warmup_start_factor 0.1 --ema_decay 0.995 --ema_update_every 1 --grad_clip_norm 1.0 --pre_head_mode identity --thermal_max_c 85 --thermal_pressure_trip_level serious --monitor_interval_sec 300
```

### A2 Baseline (`lr=1e-4`)
```bash
python train.py --data_folder "$DATA_FOLDER" --dataset_glob "seismic__*/model_data.zarr" --output_dir "$RUN_ROOT/A2_lr1e4" --device mps --sample_shape 128 128 128 --batch_size 2 --grad_accum_steps 16 --epochs 20 --val_split_ratio 0.25 --train_batches_per_epoch 64 --val_batches_per_epoch 24 --amplitude_transform histogram_equalization --transforms_group transforms_histeq_lock --loss_type ssim_mse --ssim_data_range 30.0 --ssim_window_size 16 --ssim_sigma 2.6667 --ssim_alpha 0.1667 --ssim_min_valid_ratio 0.5 --target_masked_fraction 0.15 --cluster_shape 3 --center_selection_method random_mixture --mask_fill_method zero --lr 1e-4 --lr_schedule poly --lr_poly_power 0.9 --lr_min 1e-6 --lr_warmup_epochs 3 --lr_warmup_start_factor 0.1 --ema_decay 0.995 --ema_update_every 1 --grad_clip_norm 1.0 --pre_head_mode identity --thermal_max_c 85 --thermal_pressure_trip_level serious --monitor_interval_sec 300
```

## Phase B: Drift-Control Loss Ablation
### Goals
1. Reduce low-frequency mean drift.
2. Preserve polarity and structure.

### B1 Conservative Sliding Stats
```bash
python train.py --data_folder "$DATA_FOLDER" --dataset_glob "seismic__*/model_data.zarr" --output_dir "$RUN_ROOT/B1_slide_conservative" --device mps --sample_shape 128 128 128 --batch_size 2 --grad_accum_steps 16 --epochs 20 --val_split_ratio 0.25 --train_batches_per_epoch 64 --val_batches_per_epoch 24 --amplitude_transform histogram_equalization --transforms_group transforms_histeq_lock --loss_type sliding_stats --sliding_stats_window 9 9 9 --sliding_stats_mean_weight 2.0 --sliding_stats_std_weight 1.0 --sliding_stats_min_weight 0.2 --sliding_stats_max_weight 0.2 --sliding_stats_mae_weight 0.3 --sliding_stats_mse_weight 0.3 --sliding_stats_eps 1e-6 --sliding_stats_std_ratio_clip 5.0 --target_masked_fraction 0.15 --cluster_shape 3 --center_selection_method random_mixture --mask_fill_method zero --lr 5e-5 --lr_schedule poly --lr_poly_power 0.9 --lr_min 1e-6 --lr_warmup_epochs 3 --ema_decay 0.995 --ema_update_every 1 --grad_clip_norm 1.0 --pre_head_mode identity --thermal_max_c 85 --thermal_pressure_trip_level serious --monitor_interval_sec 300
```

### B2 Stronger Mean Anchoring
```bash
python train.py --data_folder "$DATA_FOLDER" --dataset_glob "seismic__*/model_data.zarr" --output_dir "$RUN_ROOT/B2_slide_mean_strong" --device mps --sample_shape 128 128 128 --batch_size 2 --grad_accum_steps 16 --epochs 20 --val_split_ratio 0.25 --train_batches_per_epoch 64 --val_batches_per_epoch 24 --amplitude_transform histogram_equalization --transforms_group transforms_histeq_lock --loss_type sliding_stats --sliding_stats_window 9 9 9 --sliding_stats_mean_weight 3.0 --sliding_stats_std_weight 1.2 --sliding_stats_min_weight 0.2 --sliding_stats_max_weight 0.2 --sliding_stats_mae_weight 0.3 --sliding_stats_mse_weight 0.3 --sliding_stats_eps 1e-6 --sliding_stats_std_ratio_clip 5.0 --target_masked_fraction 0.15 --cluster_shape 3 --center_selection_method random_mixture --mask_fill_method zero --lr 5e-5 --lr_schedule poly --lr_poly_power 0.9 --lr_min 1e-6 --lr_warmup_epochs 3 --ema_decay 0.995 --ema_update_every 1 --grad_clip_norm 1.0 --pre_head_mode identity --thermal_max_c 85 --thermal_pressure_trip_level serious --monitor_interval_sec 300
```

## Phase C: Variance-Recovery Ablation
### Goals
1. Raise predicted std toward target std without ringing.
2. Tune std pressure and clipping.

### C1 Moderate Variance Push
```bash
python train.py --data_folder "$DATA_FOLDER" --dataset_glob "seismic__*/model_data.zarr" --output_dir "$RUN_ROOT/C1_var_moderate" --device mps --sample_shape 128 128 128 --batch_size 2 --grad_accum_steps 16 --epochs 20 --val_split_ratio 0.25 --train_batches_per_epoch 64 --val_batches_per_epoch 24 --amplitude_transform histogram_equalization --transforms_group transforms_histeq_lock --loss_type sliding_stats --sliding_stats_window 9 9 9 --sliding_stats_mean_weight 2.5 --sliding_stats_std_weight 1.5 --sliding_stats_min_weight 0.2 --sliding_stats_max_weight 0.2 --sliding_stats_mae_weight 0.3 --sliding_stats_mse_weight 0.3 --sliding_stats_eps 1e-6 --sliding_stats_std_ratio_clip 4.0 --target_masked_fraction 0.15 --cluster_shape 3 --center_selection_method random_mixture --mask_fill_method zero --lr 5e-5 --lr_schedule poly --lr_poly_power 0.9 --lr_min 1e-6 --lr_warmup_epochs 3 --ema_decay 0.995 --ema_update_every 1 --grad_clip_norm 1.0 --pre_head_mode identity --thermal_max_c 85 --thermal_pressure_trip_level serious --monitor_interval_sec 300
```

### C2 Strong Variance Push
```bash
python train.py --data_folder "$DATA_FOLDER" --dataset_glob "seismic__*/model_data.zarr" --output_dir "$RUN_ROOT/C2_var_strong" --device mps --sample_shape 128 128 128 --batch_size 2 --grad_accum_steps 16 --epochs 20 --val_split_ratio 0.25 --train_batches_per_epoch 64 --val_batches_per_epoch 24 --amplitude_transform histogram_equalization --transforms_group transforms_histeq_lock --loss_type sliding_stats --sliding_stats_window 9 9 9 --sliding_stats_mean_weight 2.5 --sliding_stats_std_weight 2.2 --sliding_stats_min_weight 0.2 --sliding_stats_max_weight 0.2 --sliding_stats_mae_weight 0.3 --sliding_stats_mse_weight 0.3 --sliding_stats_eps 1e-6 --sliding_stats_std_ratio_clip 3.0 --target_masked_fraction 0.15 --cluster_shape 3 --center_selection_method random_mixture --mask_fill_method zero --lr 5e-5 --lr_schedule poly --lr_poly_power 0.9 --lr_min 1e-6 --lr_warmup_epochs 3 --ema_decay 0.995 --ema_update_every 1 --grad_clip_norm 1.0 --pre_head_mode identity --thermal_max_c 85 --thermal_pressure_trip_level serious --monitor_interval_sec 300
```

## Phase D: Architecture Ablation for Sparse Inputs
### Current State
1. [src/synthoseis_pre_train/models.py](../src/synthoseis_pre_train/models.py) already uses large kernels in residual blocks.
2. Downsampling is already strided convolution.

### Recommendation
1. Do not switch all residual blocks to strided convolutions.
2. Use mixed kernel sizes by depth:
	 - shallow stages: larger kernels
	 - deeper stages: `3x3x3` or anisotropic blocks
3. Keep UNet depth unchanged initially; test kernel profile first.

### Required New CLI Flags
1. `--block_type`
2. `--kernel_profile`
3. `--hidden_dims`

### D1 Mixed Profile, Fixed Depth (after adding flags)
```bash
python train.py --data_folder "$DATA_FOLDER" --dataset_glob "seismic__*/model_data.zarr" --output_dir "$RUN_ROOT/D1_mixed_kernel" --device mps --sample_shape 128 128 128 --batch_size 2 --grad_accum_steps 16 --epochs 20 --val_split_ratio 0.25 --train_batches_per_epoch 64 --val_batches_per_epoch 24 --amplitude_transform histogram_equalization --transforms_group transforms_histeq_lock --loss_type sliding_stats --sliding_stats_window 9 9 9 --sliding_stats_mean_weight 2.5 --sliding_stats_std_weight 1.5 --sliding_stats_mae_weight 0.3 --sliding_stats_mse_weight 0.3 --sliding_stats_min_weight 0.2 --sliding_stats_max_weight 0.2 --sliding_stats_std_ratio_clip 4.0 --lr 5e-5 --lr_schedule poly --lr_poly_power 0.9 --lr_min 1e-6 --lr_warmup_epochs 3 --ema_decay 0.995 --ema_update_every 1 --grad_clip_norm 1.0 --pre_head_mode identity --block_type anisotropic --kernel_profile mixed_large_shallow --hidden_dims 32 64 128 256 --thermal_max_c 85 --thermal_pressure_trip_level serious
```

### D2 Lighter Channel Budget (after adding flags)
```bash
python train.py --data_folder "$DATA_FOLDER" --dataset_glob "seismic__*/model_data.zarr" --output_dir "$RUN_ROOT/D2_mixed_kernel_lighter" --device mps --sample_shape 128 128 128 --batch_size 2 --grad_accum_steps 16 --epochs 20 --val_split_ratio 0.25 --train_batches_per_epoch 64 --val_batches_per_epoch 24 --amplitude_transform histogram_equalization --transforms_group transforms_histeq_lock --loss_type sliding_stats --sliding_stats_window 9 9 9 --sliding_stats_mean_weight 2.5 --sliding_stats_std_weight 1.5 --sliding_stats_mae_weight 0.3 --sliding_stats_mse_weight 0.3 --sliding_stats_min_weight 0.2 --sliding_stats_max_weight 0.2 --sliding_stats_std_ratio_clip 4.0 --lr 5e-5 --lr_schedule poly --lr_poly_power 0.9 --lr_min 1e-6 --lr_warmup_epochs 3 --ema_decay 0.995 --ema_update_every 1 --grad_clip_norm 1.0 --pre_head_mode identity --block_type anisotropic --kernel_profile mixed_large_shallow --hidden_dims 24 48 96 192 --thermal_max_c 85 --thermal_pressure_trip_level serious
```

## Direct Answers to Architecture Questions
### Should strided convolutions be used instead?
1. Already yes for downsampling.
2. Keep that.
3. Do not add stride to all residual convs.

### Use different convolution sizes in different parts?
1. Yes.
2. For sparse-mask setup, use larger receptive fields early and smaller/decomposed kernels deeper.

### Change number of downsample/upsample steps?
1. Keep current depth first.
2. Change depth only after loss and kernel profile ablations stabilize mean/std drift.

## Recommended Improvements for Copilot Automation
### 1) Reproducibility Lock
- Context:
  1. With rotating datasets and thermal throttling on MPS, run-to-run variance can hide real improvements.
  2. Loss-ablation conclusions are unreliable unless randomness and split assignment are controlled.
- Goals:
  1. Make Phase A-C comparisons repeatable enough to trust ranking decisions.
  2. Ensure that any change in mean/std drift metrics is attributable to the experiment variable.
- Implementation plan:
  1. Add CLI args in [train.py](../train.py):
	  - `--seed` (int, default for example `1337`)
	  - `--deterministic` (flag)
  2. Early in startup, seed all RNGs in one helper:
	  - `random.seed(seed)`
	  - `numpy.random.seed(seed)`
	  - `torch.manual_seed(seed)`
	  - `torch.cuda.manual_seed_all(seed)` when CUDA is active
  3. Determinism toggles when `--deterministic` is set:
	  - `torch.use_deterministic_algorithms(True)`
	  - set backend knobs with explicit logging of any fallback
  4. Dataloader worker consistency:
	  - pass a generator with fixed seed into DataLoader
	  - add `worker_init_fn` to derive worker-specific seeds
  5. Persist and expose provenance:
	  - write seed + deterministic flag to checkpoint metadata
	  - log them to TensorBoard text/scalars at run start
	  - print them in startup stdout next to LR/loss config
  6. Resume safety:
	  - on `--resume`, verify seed compatibility and print warning if different
	  - keep split assignment deterministic when same seed + dataset list are used
  7. Validation criterion for this task:
	  - run A1 twice with same seed and confirm final val loss and drift metrics are within a tight tolerance band before proceeding with broader ablations

### 2) Fixed Validation Holdout
- Context: rotating datasets can confound validation trend.
- Goal: stable generalization signal.
- Implementation:
	1. Add `--val_paths_fixed` and `--train_paths_dynamic` in [train.py](../train.py).
	2. Never prune fixed validation paths.
	3. Apply dynamic add/drop only to train pool.
	4. Print active fixed validation list each epoch.

### 3) Drift Metrics Logging
- Context: failure mode is low-frequency mean drift and std suppression.
- Goal: optimize directly against drift.
- Implementation:
	1. In train and val loops in [train.py](../train.py), log:
		 - global mean pred/target
		 - global std pred/target
		 - pooled mean drift at kernel 8 and 16
		 - std ratio pred/target
	2. Write metrics to TensorBoard scalars.

### 4) Low-Frequency Mean Penalty
- Context: large low-frequency patches drift from zero.
- Goal: suppress local bias fields.
- Implementation:
	1. Extend [src/synthoseis_pre_train/losses.py](../src/synthoseis_pre_train/losses.py): add optional pooled-mean L1 term with configurable weight/pool size.
	2. Keep masked-only support.
	3. Expose args in [train.py](../train.py).

### 5) Phase D Architecture Flags
- Context: need reproducible architecture ablations.
- Goal: avoid hand edits per run.
- Implementation:
	1. Add args `--block_type`, `--kernel_profile`, `--hidden_dims` in [train.py](../train.py).
	2. Plumb flags into `create_model` call.
	3. Update [src/synthoseis_pre_train/models.py](../src/synthoseis_pre_train/models.py) block factory/profile selector.
	4. Print resolved per-stage kernels at startup.

### 6) Remove Training QC TODO Block
- Context: extra per-batch criterion creation confounds runtime and thermals.
- Goal: cleaner performance measurements.
- Implementation:
	1. Remove TODO QC block in [train.py](../train.py) around line region 909..953.
	2. Replace with optional lightweight scalar logging from already-computed loss.

## Tuning Guidance for Home Lab Constraints
1. Keep `batch_size=2` on MPS.
2. Use `grad_accum_steps=12..16`.
3. Start with effective batch `32`, reduce to `24` if wall time is too high.
4. Increase `train_batches_per_epoch` from 64 to 80 only if swap pressure remains low and thermals are stable.
5. Keep `val_batches_per_epoch=24..32`.
6. Keep `ema_decay=0.995` and polynomial LR schedule.
