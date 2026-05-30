#!/usr/bin/env bash

# Multi-dataset seismic training script
# Trains on all synthetic seismic datasets in a folder

set -euo pipefail

# Suppress noisy macOS allocator warnings inherited by Python subprocesses.
# Setting to "0" (not unset) is an explicit disable signal to libmalloc.
if [[ "${OSTYPE:-}" == darwin* ]]; then
  export MallocStackLogging=0
  export MallocStackLoggingNoCompact=0
fi

# ---------------------------------------------------------------------------
# Overnight mode: pre-set safer defaults BEFORE the normal defaults block so
# explicit CLI flags can still override any individual value afterwards.
# Activated by passing --overnight anywhere in the argument list.
# ---------------------------------------------------------------------------
OVERNIGHT=false
for _arg in "$@"; do
  if [[ "${_arg}" == "--overnight" ]]; then
    OVERNIGHT=true
    THERMAL_MAX_C=${THERMAL_MAX_C:-80}
    THERMAL_COOLDOWN_SEC=${THERMAL_COOLDOWN_SEC:-420}
    THERMAL_CHECK_EVERY_BATCHES=${THERMAL_CHECK_EVERY_BATCHES:-5}
    THERMAL_PRESSURE_TRIP_LEVEL=${THERMAL_PRESSURE_TRIP_LEVEL:-fair}
    GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-6}
    GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-0.7}
    LR_WARMUP_EPOCHS=${LR_WARMUP_EPOCHS:-8}
    LR_WARMUP_START_FACTOR=${LR_WARMUP_START_FACTOR:-0.05}
    EMA_DECAY=${EMA_DECAY:-0.9995}
    break
  fi
done

# Default parameters
MAX_EPOCHS=${MAX_EPOCHS:-25}
DATA_FOLDER=${DATA_FOLDER:-"/Users/donaldpg/synthoseis/fake_data"}
BATCH_SIZE=${BATCH_SIZE:-auto}
SAMPLE_SHAPE=${SAMPLE_SHAPE:-"128 128 128"}
DEVICE=${DEVICE:-"auto"}
VAL_SPLIT_RATIO=${VAL_SPLIT_RATIO:-0.2}
TRAIN_BATCHES_PER_EPOCH=${TRAIN_BATCHES_PER_EPOCH:-120}
VAL_BATCHES_PER_EPOCH=${VAL_BATCHES_PER_EPOCH:-30}
REFRESH_EVERY_BATCHES=${REFRESH_EVERY_BATCHES:-10}
THERMAL_MAX_C=${THERMAL_MAX_C:-85}
THERMAL_COOLDOWN_SEC=${THERMAL_COOLDOWN_SEC:-300}
THERMAL_CHECK_EVERY_BATCHES=${THERMAL_CHECK_EVERY_BATCHES:-10}
THERMAL_PRESSURE_TRIP_LEVEL=${THERMAL_PRESSURE_TRIP_LEVEL:-serious}
LR_SCHEDULE=${LR_SCHEDULE:-poly}
LR_POLY_POWER=${LR_POLY_POWER:-0.9}
LR_MIN=${LR_MIN:-1e-6}
LR_WARMUP_EPOCHS=${LR_WARMUP_EPOCHS:-5}
LR_WARMUP_START_FACTOR=${LR_WARMUP_START_FACTOR:-0.1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-1.0}
EMA_DECAY=${EMA_DECAY:-0.999}
EMA_UPDATE_EVERY=${EMA_UPDATE_EVERY:-1}
KERNEL_SIZES=${KERNEL_SIZES:-""}
HIDDEN_DIMS=${HIDDEN_DIMS:-"32 64 128 256"}
LOSS=${LOSS:-"huber"}
MAE_SMOOTH_KERNEL_WEIGHTS=${MAE_SMOOTH_KERNEL_WEIGHTS:-"1 2 1"}
HUBER_DELTA=${HUBER_DELTA:-1.0}
SSIM_WINDOW_SIZE=${SSIM_WINDOW_SIZE:-7}
SSIM_W1=${SSIM_W1:-1.0}
SSIM_W2=${SSIM_W2:-0.0}
SSIM_W3=${SSIM_W3:-0.0}
STATS_WINDOW_SIZE=${STATS_WINDOW_SIZE:-"9 9 9"}
STATS_MASK_MODE=${STATS_MASK_MODE:-"none"}
STATS_MEAN_WEIGHT=${STATS_MEAN_WEIGHT:-1.0}
STATS_STD_WEIGHT=${STATS_STD_WEIGHT:-1.0}
STATS_MIN_WEIGHT=${STATS_MIN_WEIGHT:-1.0}
STATS_MAX_WEIGHT=${STATS_MAX_WEIGHT:-1.0}
STATS_MAE_WEIGHT=${STATS_MAE_WEIGHT:-1.0}
STATS_MSE_WEIGHT=${STATS_MSE_WEIGHT:-1.0}
STATS_STD_RATIO_CLIP=${STATS_STD_RATIO_CLIP:-10.0}
RESUME=${RESUME:-""}

# Deep reconstruction head flag
DEEP_RECONSTRUCTION_HEAD=${DEEP_RECONSTRUCTION_HEAD:-0}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
      --deep-reconstruction-head)
        DEEP_RECONSTRUCTION_HEAD=1
        shift
        ;;
  case $1 in
    --smae)
      LOSS="smae"
      shift
      ;;
    --max-epochs)
      MAX_EPOCHS="$2"
      shift 2
      ;;
    --data-folder)
      DATA_FOLDER="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --sample-shape)
      SAMPLE_SHAPE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --val-split-ratio)
      VAL_SPLIT_RATIO="$2"
      shift 2
      ;;
    --train-batches-per-epoch)
      TRAIN_BATCHES_PER_EPOCH="$2"
      shift 2
      ;;
    --val-batches-per-epoch)
      VAL_BATCHES_PER_EPOCH="$2"
      shift 2
      ;;
    --refresh-every-batches)
      REFRESH_EVERY_BATCHES="$2"
      shift 2
      ;;
    --thermal-max-c)
      THERMAL_MAX_C="$2"
      shift 2
      ;;
    --thermal-cooldown-sec)
      THERMAL_COOLDOWN_SEC="$2"
      shift 2
      ;;
    --thermal-check-every-batches)
      THERMAL_CHECK_EVERY_BATCHES="$2"
      shift 2
      ;;
    --thermal-pressure-trip-level)
      THERMAL_PRESSURE_TRIP_LEVEL="$2"
      shift 2
      ;;
    --lr-schedule)
      LR_SCHEDULE="$2"
      shift 2
      ;;
    --lr-poly-power)
      LR_POLY_POWER="$2"
      shift 2
      ;;
    --lr-min)
      LR_MIN="$2"
      shift 2
      ;;
    --lr-warmup-epochs)
      LR_WARMUP_EPOCHS="$2"
      shift 2
      ;;
    --lr-warmup-start-factor)
      LR_WARMUP_START_FACTOR="$2"
      shift 2
      ;;
    --grad-accum-steps)
      GRAD_ACCUM_STEPS="$2"
      shift 2
      ;;
    --grad-clip-norm)
      GRAD_CLIP_NORM="$2"
      shift 2
      ;;
    --ema-decay)
      EMA_DECAY="$2"
      shift 2
      ;;
    --ema-update-every)
      EMA_UPDATE_EVERY="$2"
      shift 2
      ;;
    --kernel-sizes)
      KERNEL_SIZES="$2"
      shift 2
      ;;
    --hidden-dims)
      HIDDEN_DIMS="$2"
      shift 2
      ;;
    --loss-fn)
      LOSS="$2"
      shift 2
      ;;
    --huber-delta)
      HUBER_DELTA="$2"
      shift 2
      ;;
    --mae-smooth-kernel-weights)
      MAE_SMOOTH_KERNEL_WEIGHTS="$2"
      shift 2
      ;;
    --ssim-window-size)
      SSIM_WINDOW_SIZE="$2"
      shift 2
      ;;
    --ssim-w1)
      SSIM_W1="$2"
      shift 2
      ;;
    --ssim-w2)
      SSIM_W2="$2"
      shift 2
      ;;
    --ssim-w3)
      SSIM_W3="$2"
      shift 2
      ;;
    --stats-window-size)
      STATS_WINDOW_SIZE="$2"
      shift 2
      ;;
    --stats-mask-mode)
      STATS_MASK_MODE="$2"
      shift 2
      ;;
    --stats-mean-weight)
      STATS_MEAN_WEIGHT="$2"
      shift 2
      ;;
    --stats-std-weight)
      STATS_STD_WEIGHT="$2"
      shift 2
      ;;
    --stats-min-weight)
      STATS_MIN_WEIGHT="$2"
      shift 2
      ;;
    --stats-max-weight)
      STATS_MAX_WEIGHT="$2"
      shift 2
      ;;
    --stats-mae-weight)
      STATS_MAE_WEIGHT="$2"
      shift 2
      ;;
    --stats-mse-weight)
      STATS_MSE_WEIGHT="$2"
      shift 2
      ;;
    --stats-std-ratio-clip)
      STATS_STD_RATIO_CLIP="$2"
      shift 2
      ;;
    --overnight)
      # Already handled above; consume the flag so it isn't treated as unknown.
      shift
      ;;
    --resume)
      RESUME="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Train on multiple synthetic seismic datasets"
      echo ""
      echo "Options:"
      echo "  --deep-reconstruction-head  Use a deep reconstruction head (2 Conv3d layers with norm and activation)"
      echo "  --smae                    Use SMAE (Symmetric Mean Absolute Error) loss for regression (arXiv:2303.09935)"
      echo "  --max-epochs NUM      Maximum epochs for training (default: 25)"
      echo "  --data-folder PATH    Top-level folder containing datasets (default: /Users/donaldpg/synthoseis/fake_data)"
      echo "  --batch-size NUM|auto Batch size or 'auto' for automatic calculation (default: auto)"
      echo "  --sample-shape 'X Y Z' Sample shape (default: '128 128 128')"
      echo "  --device DEV          Device (auto/cuda/mps/cpu) (default: auto)"
      echo "  --val-split-ratio R   Validation split ratio over discovered datasets"
      echo "                       (default: 0.2)"
      echo "  --train-batches-per-epoch N"
      echo "                       Fixed number of train batches per epoch (default: 120)"
      echo "  --val-batches-per-epoch N"
      echo "                       Fixed number of val batches per epoch (default: 30)"
      echo "  --refresh-every-batches N"
      echo "                       Deprecated compatibility flag; dataset discovery/pruning"
      echo "                       now runs at epoch boundaries (default: 10)"
      echo "  --thermal-max-c NUM   Pause when CPU temperature reaches this Celsius value (default: 85)"
      echo "  --thermal-cooldown-sec NUM"
      echo "                       Cooldown pause in seconds after a thermal trip (default: 300)"
      echo "  --thermal-check-every-batches NUM"
      echo "                       Check CPU temperature every N training batches (default: 10)"
      echo "  --thermal-pressure-trip-level LVL"
      echo "                       Pause for thermal pressure at/above this level:"
      echo "                       off|nominal|fair|serious|critical (default: serious)"
      echo "  --lr-schedule MODE   LR schedule: poly|cosine|constant (default: poly)"
      echo "  --lr-poly-power NUM  Polynomial power for poly LR schedule (default: 0.9)"
      echo "  --lr-min NUM         Minimum LR floor for poly/cosine (default: 1e-6)"
      echo "  --lr-warmup-epochs N Warmup epochs before LR decay (default: 5)"
      echo "  --lr-warmup-start-factor NUM"
      echo "                       Warmup start as fraction of base LR (default: 0.1)"
      echo "  --grad-accum-steps N Gradient accumulation steps (default: 1)"
      echo "  --grad-clip-norm NUM Global gradient clipping max-norm (default: 1.0; <=0 disables)"
      echo "  --ema-decay NUM      EMA decay (default: 0.999; <=0 disables)"
      echo "  --ema-update-every N EMA update cadence in optimizer steps (default: 1)"
      echo "  --kernel-sizes 'K1 K2 ...'"
      echo "                       Optional odd per-stage kernel schedule forwarded to train.py"
      echo "                       (example: '7 5 3 3'; default keeps legacy 3x3 kernels)"
  echo "  --hidden-dims 'C1 C2 ...'"
      echo "                       Channel widths per encoder stage (default: '32 64 128 256')"
      echo "                       Length determines U-Net depth (e.g. '16 32 64 128' = shallower)"
  echo "  --loss-fn NAME       Loss function: mse | mae | mae_smooth | huber | ssim | sliding_stats | smae (default: huber)"
  echo "                       smae = Smooth MAE (e*tanh(e/2), arXiv:2303.09935)"
  echo "  --mae-smooth-kernel-weights 'W1 W2 ...'"
  echo "                       Odd-length 1D smoothing kernel for --loss-fn=mae_smooth (default: '1 2 1')"
      echo "  --huber-delta NUM    Delta for SmoothL1Loss when --loss-fn=huber (default: 1.0)"
      echo "  --ssim-window-size N Odd cubic SSIM window edge length when --loss-fn=ssim (default: 7)"
      echo "  --ssim-w1 NUM        Weight w1 for (1-SSIM) term when --loss-fn=ssim (default: 1.0)"
      echo "  --ssim-w2 NUM        Weight w2 for MSE term when --loss-fn=ssim (default: 0.0)"
      echo "  --ssim-w3 NUM        Weight w3 for L1 term when --loss-fn=ssim (default: 0.0)"
      echo "  --stats-window-size 'D H W'"
      echo "                       Sliding window size for --loss-fn=sliding_stats (default: '9 9 9')"
      echo "  --stats-mask-mode MODE"
      echo "                       Mask behavior for sliding_stats: none|valid (default: none)"
      echo "  --stats-mean-weight NUM"
      echo "  --stats-std-weight NUM"
      echo "  --stats-min-weight NUM"
      echo "  --stats-max-weight NUM"
      echo "  --stats-mae-weight NUM"
      echo "  --stats-mse-weight NUM"
      echo "                       Sliding_stats component weights (all default: 1.0)"
      echo "  --stats-std-ratio-clip NUM"
      echo "                       Std-ratio clipping bound for sliding_stats (default: 10.0)"
      echo "  --overnight           Enable overnight/unattended mode: applies safer thermal defaults"
      echo "                       (max-c 80, cooldown 420s, check every 5 batches, pressure=fair)"
      echo "                       and stability-first optimizer settings. Individual flags override."
      echo "  --resume PATH         Resume from checkpoint file (e.g. checkpoints/partial_latest.pt)"
      echo "  --help                Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

[[ "${OVERNIGHT}" == "true" ]] && echo "*** Overnight mode active — safer thermal and stability defaults applied ***"
echo "=== Multi-dataset Seismic Training ==="
echo "Data folder: ${DATA_FOLDER}"
echo "Max epochs: ${MAX_EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Sample shape: ${SAMPLE_SHAPE}"
echo "Device: ${DEVICE}"
echo "Val split ratio:    ${VAL_SPLIT_RATIO}"
echo "Train/val counts:   auto-resolved from discovered dataset count"
echo "Train batches/epoch: ${TRAIN_BATCHES_PER_EPOCH}"
echo "Val batches/epoch:   ${VAL_BATCHES_PER_EPOCH}"
echo "Refresh every:      ${REFRESH_EVERY_BATCHES} train batches (deprecated; epoch-boundary refresh is used)"
echo "Thermal max C:      ${THERMAL_MAX_C}"
echo "Thermal cooldown:   ${THERMAL_COOLDOWN_SEC}s"
echo "Thermal check rate: every ${THERMAL_CHECK_EVERY_BATCHES} batches"
echo "Thermal pressure trip level: ${THERMAL_PRESSURE_TRIP_LEVEL}"
echo "LR schedule:        ${LR_SCHEDULE}"
echo "Optimizer:          Adam (fixed in train.py)"
echo "Base LR:            train.py default (1e-4, unless passed directly to train.py --lr)"
echo "LR poly power:      ${LR_POLY_POWER}"
echo "LR min:             ${LR_MIN}"
echo "LR warmup:          ${LR_WARMUP_EPOCHS} epoch(s), start factor ${LR_WARMUP_START_FACTOR}"
echo "Grad accumulation:  ${GRAD_ACCUM_STEPS}"
echo "Grad clip norm:     ${GRAD_CLIP_NORM}"
echo "EMA decay:          ${EMA_DECAY}"
echo "EMA update every:   ${EMA_UPDATE_EVERY} step(s)"
if [[ -n "${KERNEL_SIZES}" ]]; then
echo "Kernel sizes:       ${KERNEL_SIZES}"
else
echo "Kernel sizes:       default (legacy 3x3 kernels)"
fi
echo "Hidden dims:        ${HIDDEN_DIMS}"
if [[ "${LOSS}" == "huber" ]]; then
echo "Loss function:       ${LOSS} (delta=${HUBER_DELTA})"
elif [[ "${LOSS}" == "mae_smooth" ]]; then
echo "Loss function:       mae_smooth (kernel_1d=${MAE_SMOOTH_KERNEL_WEIGHTS})"
elif [[ "${LOSS}" == "ssim" ]]; then
echo "Loss function:       ssim-hybrid (window=${SSIM_WINDOW_SIZE}, ssim_term=${SSIM_W1}, mse_term=${SSIM_W2}, mae_term=${SSIM_W3})"
elif [[ "${LOSS}" == "sliding_stats" ]]; then
echo "Loss function:       sliding_stats (window=${STATS_WINDOW_SIZE}, mask_mode=${STATS_MASK_MODE}, std_ratio_clip=${STATS_STD_RATIO_CLIP})"
echo "                     weights: mean=${STATS_MEAN_WEIGHT}, std=${STATS_STD_WEIGHT}, min=${STATS_MIN_WEIGHT}, max=${STATS_MAX_WEIGHT}, mae=${STATS_MAE_WEIGHT}, mse=${STATS_MSE_WEIGHT}"
else
echo "Loss function:       ${LOSS}"
fi
echo "Backprop config:     grad_accum_steps=${GRAD_ACCUM_STEPS}; grad_clip_norm=${GRAD_CLIP_NORM}; ema_decay=${EMA_DECAY}; ema_update_every=${EMA_UPDATE_EVERY}"
[[ -n "${RESUME}" ]] && echo "Resume from: ${RESUME}"
echo ""

# Calculate batch size if set to auto
if [[ "${BATCH_SIZE}" == "auto" ]]; then
    echo "Calculating optimal batch size..."
  if CALCULATED_BATCH_SIZE=$(uv run python calculate_batch_size.py \
    --sample-shape ${SAMPLE_SHAPE} \
    --device "${DEVICE}" \
    --quiet); then
        BATCH_SIZE="${CALCULATED_BATCH_SIZE}"
        echo "Using calculated batch size: ${BATCH_SIZE}"
    else
    echo "WARNING: Failed to calculate batch size automatically; using fallback batch size of 1"
        BATCH_SIZE=1
    fi
    echo ""
fi

# Verify data folder contains at least one seismic dataset folder
INITIAL_COUNT=$(find "${DATA_FOLDER}" -maxdepth 1 -type d -name "seismic__*" | wc -l | tr -d ' ')
if [[ "${INITIAL_COUNT}" -eq 0 ]]; then
    echo "ERROR: No seismic datasets found in ${DATA_FOLDER}"
    echo "Expected folders matching 'seismic__*'"
    exit 1
fi

echo "Found ${INITIAL_COUNT} dataset folder(s) in ${DATA_FOLDER} at startup"
echo ""

# Train — train.py re-scans DATA_FOLDER at the start of each epoch and
# incorporates new datasets automatically.  The set-difference split logic
# guarantees that no dataset ever appears in both train and val.
uv run python -u train.py \
    --data_folder "${DATA_FOLDER}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${MAX_EPOCHS}" \
    --sample_shape ${SAMPLE_SHAPE} \
    --device "${DEVICE}" \
    --output_dir "checkpoints" \
    --train_batches_per_epoch "${TRAIN_BATCHES_PER_EPOCH}" \
    --val_batches_per_epoch "${VAL_BATCHES_PER_EPOCH}" \
    --refresh_every_batches "${REFRESH_EVERY_BATCHES}" \
    --val_split_ratio "${VAL_SPLIT_RATIO}" \
    --thermal_max_c "${THERMAL_MAX_C}" \
    --thermal_cooldown_sec "${THERMAL_COOLDOWN_SEC}" \
    --thermal_check_every_batches "${THERMAL_CHECK_EVERY_BATCHES}" \
    --thermal_pressure_trip_level "${THERMAL_PRESSURE_TRIP_LEVEL}" \
    --lr_schedule "${LR_SCHEDULE}" \
    --lr_poly_power "${LR_POLY_POWER}" \
    --lr_min "${LR_MIN}" \
    --lr_warmup_epochs "${LR_WARMUP_EPOCHS}" \
    --lr_warmup_start_factor "${LR_WARMUP_START_FACTOR}" \
    --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
    --grad_clip_norm "${GRAD_CLIP_NORM}" \
    --ema_decay "${EMA_DECAY}" \
    --ema_update_every "${EMA_UPDATE_EVERY}" \
    ${KERNEL_SIZES:+--kernel_sizes ${KERNEL_SIZES}} \
    --hidden_dims ${HIDDEN_DIMS} \
    --loss "${LOSS}" \
    --mae_smooth_kernel_weights ${MAE_SMOOTH_KERNEL_WEIGHTS} \
    --huber_delta "${HUBER_DELTA}" \
    $( [[ "${DEEP_RECONSTRUCTION_HEAD}" == "1" ]] && echo "--deep-reconstruction-head" ) \
    --ssim_window_size "${SSIM_WINDOW_SIZE}" \
    --ssim_w1 "${SSIM_W1}" \
    --ssim_w2 "${SSIM_W2}" \
    --ssim_w3 "${SSIM_W3}" \
    --stats_window_size ${STATS_WINDOW_SIZE} \
    --stats_mask_mode "${STATS_MASK_MODE}" \
    --stats_mean_weight "${STATS_MEAN_WEIGHT}" \
    --stats_std_weight "${STATS_STD_WEIGHT}" \
    --stats_min_weight "${STATS_MIN_WEIGHT}" \
    --stats_max_weight "${STATS_MAX_WEIGHT}" \
    --stats_mae_weight "${STATS_MAE_WEIGHT}" \
    --stats_mse_weight "${STATS_MSE_WEIGHT}" \
    --stats_std_ratio_clip "${STATS_STD_RATIO_CLIP}" \
    ${RESUME:+--resume "${RESUME}"}

echo "=== Multi-dataset training complete ==="