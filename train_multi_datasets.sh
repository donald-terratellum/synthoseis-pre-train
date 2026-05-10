#!/usr/bin/env bash

# Multi-dataset seismic training script
# Trains on all synthetic seismic datasets in a folder

set -euo pipefail

# Suppress noisy macOS allocator warnings inherited by Python subprocesses.
# Setting to "0" (not unset) is an explicit disable signal to libmalloc.
if [[ "${OSTYPE:-}" == darwin* ]]; then
  # Some shells carry additional Malloc* variables that make tools like awk
  # print "MallocStackLogging ... No such file or directory". Normalize all
  # of them to 0 for this process tree.
  for _v in $(compgen -v); do
    if [[ "${_v}" == Malloc* ]]; then
      export "${_v}=0"
    fi
  done
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
LR=${LR:-1e-4}
LR_POLY_POWER=${LR_POLY_POWER:-0.9}
LR_MIN=${LR_MIN:-1e-6}
LR_WARMUP_EPOCHS=${LR_WARMUP_EPOCHS:-5}
LR_WARMUP_START_FACTOR=${LR_WARMUP_START_FACTOR:-0.1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-1.0}
EMA_DECAY=${EMA_DECAY:-0.999}
EMA_UPDATE_EVERY=${EMA_UPDATE_EVERY:-1}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints"}
TARGET_MASKED_FRACTION=${TARGET_MASKED_FRACTION:-0.15}
CLUSTER_SHAPE=${CLUSTER_SHAPE:-3}
CENTER_SELECTION_METHOD=${CENTER_SELECTION_METHOD:-random_mixture}
LOSS_TYPE=${LOSS_TYPE:-huber}
HUBER_DELTA=${HUBER_DELTA:-0.1}
SSIM_WINDOW_SIZE=${SSIM_WINDOW_SIZE:-16}
SSIM_SIGMA=${SSIM_SIGMA:-2.6666667}
SSIM_DATA_RANGE=${SSIM_DATA_RANGE:-30.0}
SSIM_ALPHA=${SSIM_ALPHA:-0.1666667}
SSIM_MIN_VALID_RATIO=${SSIM_MIN_VALID_RATIO:-0.5}
ENABLE_CLUSTER_LOSS=${ENABLE_CLUSTER_LOSS:-false}
CLUSTER_KERNEL_SIZE=${CLUSTER_KERNEL_SIZE:-5}
CLUSTER_EPS=${CLUSTER_EPS:-1e-6}
CLUSTER_BASE_WEIGHT=${CLUSTER_BASE_WEIGHT:-0.3333333}
CLUSTER_CLUSTER_WEIGHT=${CLUSTER_CLUSTER_WEIGHT:-0.6666667}
RESUME=${RESUME:-""}
MONITOR_DISABLED=${MONITOR_DISABLED:-false}
MONITOR_INTERVAL_SEC=${MONITOR_INTERVAL_SEC:-300}
MONITOR_CSV_PATH=${MONITOR_CSV_PATH:-""}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
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
    --lr)
      LR="$2"
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
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    
    --target-masked-fraction)
      TARGET_MASKED_FRACTION="$2"
      shift 2
      ;;
    --cluster-shape)
      CLUSTER_SHAPE="$2"
      shift 2
      ;;
    --center-selection-method)
      CENTER_SELECTION_METHOD="$2"
      shift 2
      ;;
    --loss-type)
      LOSS_TYPE="$2"
      shift 2
      ;;
    --huber-delta)
      HUBER_DELTA="$2"
      shift 2
      ;;
    --ssim-window-size)
      SSIM_WINDOW_SIZE="$2"
      shift 2
      ;;
    --ssim-sigma)
      SSIM_SIGMA="$2"
      shift 2
      ;;
    --ssim-data-range)
      SSIM_DATA_RANGE="$2"
      shift 2
      ;;
    --ssim-alpha)
      SSIM_ALPHA="$2"
      shift 2
      ;;
    --ssim-min-valid-ratio)
      SSIM_MIN_VALID_RATIO="$2"
      shift 2
      ;;
    --enable-cluster-loss)
      ENABLE_CLUSTER_LOSS="true"
      shift
      ;;
    --cluster-kernel-size)
      CLUSTER_KERNEL_SIZE="$2"
      shift 2
      ;;
    --cluster-eps)
      CLUSTER_EPS="$2"
      shift 2
      ;;
    --cluster-base-weight)
      CLUSTER_BASE_WEIGHT="$2"
      shift 2
      ;;
    --cluster-cluster-weight)
      CLUSTER_CLUSTER_WEIGHT="$2"
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
    --no-monitor)
      MONITOR_DISABLED=true
      shift
      ;;
    --monitor-interval-sec)
      MONITOR_INTERVAL_SEC="$2"
      shift 2
      ;;
    --monitor-csv-path)
      MONITOR_CSV_PATH="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Train on multiple synthetic seismic datasets"
      echo ""
      echo "Options:"
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
      echo "  --output-dir PATH    Checkpoint/output directory (default: checkpoints)"

      echo "  --target-masked-fraction NUM"
      echo "                       Target final masked fraction in [0,1] after cluster effects"
      echo "                       (default: 0.15)"
      echo "  --cluster-shape N    Odd cluster edge size (e.g. 3, 5, 7) (default: 3)"
      echo "  --center-selection-method METHOD"
      echo "                       random_mixture|mitchell_best_candidate|poisson_disc|uniform_random"
      echo "                       (default: random_mixture)"
      echo "  --loss-type TYPE     Loss function: mse|huber|ssim_mse (default: huber)"
      echo "  --huber-delta NUM    Delta parameter for Huber loss (default: 0.1)"
      echo "  --ssim-window-size N 3D SSIM Gaussian window size (default: 16)"
      echo "  --ssim-sigma NUM     3D SSIM Gaussian sigma (default: 2.6666667)"
      echo "  --ssim-data-range NUM"
      echo "                       Data range used by SSIM constants (default: 30.0)"
      echo "  --ssim-alpha NUM     Blend factor in [0,1] for SSIM+MSE; 0=MSE, 1=SSIM"
      echo "                       (default: 0.1666667, matching prior 0.2 weight ratio)"
      echo "  --ssim-min-valid-ratio NUM"
      echo "                       Minimum local valid support ratio in [0,1] (default: 0.5)"
      echo "  --enable-cluster-loss  Enable composite cluster-aware loss that upweights traces near masked clusters"
      echo "  --cluster-kernel-size N" \
      echo "                       Kernel size for 2D uniform filter applied to trace mask (odd int, default: 5)"
      echo "  --cluster-eps NUM      Epsilon threshold for smoothed cluster mask (default: 1e-6)"
      echo "  --cluster-base-weight NUM"
      echo "                       Weight for base loss in composite (default: 1/3)"
      echo "  --cluster-cluster-weight NUM"
      echo "                       Weight for cluster loss in composite (default: 2/3)"
      echo "  --overnight           Enable overnight/unattended mode: applies safer thermal defaults"
      echo "                       (max-c 80, cooldown 420s, check every 5 batches, pressure=fair)"
      echo "                       and stability-first optimizer settings. Individual flags override."
        echo "  --resume PATH         Resume from checkpoint file (e.g. checkpoints/partial_latest.pt)"
        echo "  --no-monitor          Disable background process-tree resource CSV monitor (enabled by default)"
        echo "  --monitor-interval-sec NUM"
        echo "                       Monitor sampling interval in seconds (default: 300)"
        echo "  --monitor-csv-path PATH"
        echo "                       CSV path for monitor rows (default: cpu_mem_stats_<pid>.csv)"
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

case "${LOSS_TYPE}" in
  mse|huber|ssim_mse)
    ;;
  *)
    echo "ERROR: --loss-type must be one of 'mse', 'huber', 'ssim_mse' (got: ${LOSS_TYPE})"
    exit 1
    ;;
esac

if ! awk "BEGIN { exit !(${HUBER_DELTA} > 0) }"; then
  echo "ERROR: --huber-delta must be > 0 (got: ${HUBER_DELTA})"
  exit 1
fi

if ! awk "BEGIN { exit !(${SSIM_WINDOW_SIZE} >= 3 && int(${SSIM_WINDOW_SIZE}) == ${SSIM_WINDOW_SIZE}) }"; then
  echo "ERROR: --ssim-window-size must be an integer >= 3 (got: ${SSIM_WINDOW_SIZE})"
  exit 1
fi

if ! awk "BEGIN { exit !(${SSIM_SIGMA} > 0) }"; then
  echo "ERROR: --ssim-sigma must be > 0 (got: ${SSIM_SIGMA})"
  exit 1
fi

if ! awk "BEGIN { exit !(${SSIM_DATA_RANGE} > 0) }"; then
  echo "ERROR: --ssim-data-range must be > 0 (got: ${SSIM_DATA_RANGE})"
  exit 1
fi

if ! awk "BEGIN { exit !(${SSIM_ALPHA} >= 0 && ${SSIM_ALPHA} <= 1) }"; then
  echo "ERROR: --ssim-alpha must be between 0 and 1 (got: ${SSIM_ALPHA})"
  exit 1
fi

if ! awk "BEGIN { exit !(${SSIM_MIN_VALID_RATIO} >= 0 && ${SSIM_MIN_VALID_RATIO} <= 1) }"; then
  echo "ERROR: --ssim-min-valid-ratio must be between 0 and 1 (got: ${SSIM_MIN_VALID_RATIO})"
  exit 1
fi



if ! awk "BEGIN { exit !(${TARGET_MASKED_FRACTION} >= 0 && ${TARGET_MASKED_FRACTION} <= 1) }"; then
  echo "ERROR: --target-masked-fraction must be between 0 and 1 (got: ${TARGET_MASKED_FRACTION})"
  exit 1
fi

if ! awk "BEGIN { exit !(${CLUSTER_SHAPE} >= 1 && int(${CLUSTER_SHAPE}) == ${CLUSTER_SHAPE} && int(${CLUSTER_SHAPE}) % 2 == 1) }"; then
  echo "ERROR: --cluster-shape must be a positive odd integer (got: ${CLUSTER_SHAPE})"
  exit 1
fi

case "${CENTER_SELECTION_METHOD}" in
  random_mixture|mitchell_best_candidate|poisson_disc|uniform_random)
    ;;
  *)
    echo "ERROR: --center-selection-method must be one of random_mixture|mitchell_best_candidate|poisson_disc|uniform_random (got: ${CENTER_SELECTION_METHOD})"
    exit 1
    ;;
esac

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
echo "LR poly power:      ${LR_POLY_POWER}"
echo "LR min:             ${LR_MIN}"
echo "LR warmup:          ${LR_WARMUP_EPOCHS} epoch(s), start factor ${LR_WARMUP_START_FACTOR}"
echo "Grad accumulation:  ${GRAD_ACCUM_STEPS}"
echo "Grad clip norm:     ${GRAD_CLIP_NORM}"
echo "EMA decay:          ${EMA_DECAY}"
echo "EMA update every:   ${EMA_UPDATE_EVERY} step(s)"
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

# Ensure checkpoint/output directory exists before launching training.
mkdir -p "${OUTPUT_DIR}"

# Train — run the local train.py explicitly so forwarded args match local parser.
# train.py re-scans DATA_FOLDER at the start of each epoch and incorporates new datasets.
# Using an explicit path avoids accidentally invoking a different installed script.
PYRUN=""
if command -v uv >/dev/null 2>&1; then
  PYRUN="uv run python -u"
elif command -v python3 >/dev/null 2>&1; then
  PYRUN="python3 -u"
elif command -v python >/dev/null 2>&1; then
  PYRUN="python -u"
else
  echo "ERROR: no suitable python interpreter found (tried: uv, python3, python)"
  exit 1
fi

$PYRUN ./train.py \
  --data_folder "${DATA_FOLDER}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${MAX_EPOCHS}" \
  --sample_shape ${SAMPLE_SHAPE} \
  --device "${DEVICE}" \
  --output_dir "${OUTPUT_DIR}" \
  --target_masked_fraction "${TARGET_MASKED_FRACTION}" \
  --cluster_shape "${CLUSTER_SHAPE}" \
  --center_selection_method "${CENTER_SELECTION_METHOD}" \
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
  --loss_type "${LOSS_TYPE}" \
  --huber_delta "${HUBER_DELTA}" \
  --ssim_window_size "${SSIM_WINDOW_SIZE}" \
  --ssim_sigma "${SSIM_SIGMA}" \
  --ssim_data_range "${SSIM_DATA_RANGE}" \
  --ssim_alpha "${SSIM_ALPHA}" \
  --ssim_min_valid_ratio "${SSIM_MIN_VALID_RATIO}" \
  ${ENABLE_CLUSTER_LOSS:+--enable_cluster_loss} \
  --cluster_kernel_size "${CLUSTER_KERNEL_SIZE}" \
  --cluster_eps "${CLUSTER_EPS}" \
  --cluster_base_weight "${CLUSTER_BASE_WEIGHT}" \
  --cluster_cluster_weight "${CLUSTER_CLUSTER_WEIGHT}" \
  ${RESUME:+--resume "${RESUME}"}

echo "=== Multi-dataset training complete ==="
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
RESUME=${RESUME:-""}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
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
echo "LR poly power:      ${LR_POLY_POWER}"
echo "LR min:             ${LR_MIN}"
echo "LR warmup:          ${LR_WARMUP_EPOCHS} epoch(s), start factor ${LR_WARMUP_START_FACTOR}"
echo "Grad accumulation:  ${GRAD_ACCUM_STEPS}"
echo "Grad clip norm:     ${GRAD_CLIP_NORM}"
echo "EMA decay:          ${EMA_DECAY}"
echo "EMA update every:   ${EMA_UPDATE_EVERY} step(s)"
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

# Training is invoked earlier via the $PYRUN block above which forwards
# all relevant loss/SSIM/cluster flags to the local ./train.py.  The older
# direct invocation using `uv run python -u train.py` was removed to avoid
# accidentally running a different `train.py` from PATH.