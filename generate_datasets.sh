#!/usr/bin/env bash
# generate_datasets.sh
# --------------------
# Runs synthoseis in a loop to produce new synthetic seismic datasets.
# Default mode is append-only. Optional legacy mode can replace the oldest
# dataset after each run.
#
# Usage:
#   ./generate_datasets.sh [options]
#
# Options:
#   -n, --num-runs N              Number of synthoseis runs to execute (default: 1)
#   -c, --config PATH             Synthoseis config JSON (default: config/example_bigger_ex.json)
#   --synthoseis-dir PATH     Directory containing the synthoseis main.py (default: .)
#   -d, --synthoseis-zarr-folder PATH Folder where seismic zarr datasets live
#                                 (default: /Users/donaldpg/synthoseis/fake_data)
#   --check-log FILE              Training log file to interrogate for the
#                                 currently-active dataset (last 25 lines scanned)
#   --start-index N               First run index (d4, default: 1; auto-detected from
#                                 existing _synthoseis_run_NNNN dirs if not set)
#   --no-replace                  Append-only mode (default): never delete datasets
#   --replace-oldest              Legacy mode: delete oldest replaceable dataset
#   -h, --help                    Show this help and exit

set -euo pipefail

# Suppress noisy macOS allocator warnings inherited by Python subprocesses.
# Setting to "0" (not unset) is an explicit disable signal to libmalloc.
if [[ "${OSTYPE:-}" == darwin* ]]; then
    export MallocStackLogging=0
    export MallocStackLoggingNoCompact=0
fi

# ── defaults ──────────────────────────────────────────────────────────────────
NUM_RUNS=1
CONFIG="config/example_bigger_ex.json"
SYNTHOSEIS_DIR="."
ZARR_FOLDER="/Users/donaldpg/synthoseis/fake_data"
CHECK_LOG=""
START_INDEX=""   # auto-detect if empty
NO_REPLACE=true

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--num-runs)          NUM_RUNS="$2";               shift 2 ;;
        -c|--config)            CONFIG="$2";                 shift 2 ;;
        --synthoseis-dir)       SYNTHOSEIS_DIR="$2";         shift 2 ;;
        -d|--synthoseis-zarr-folder) ZARR_FOLDER="$2";      shift 2 ;;
        --check-log)            CHECK_LOG="$2";              shift 2 ;;
        --start-index)          START_INDEX="$2";            shift 2 ;;
        --no-replace)           NO_REPLACE=true;              shift ;;
        --replace-oldest)       NO_REPLACE=false;             shift ;;
        -h|--help)
            awk '/^# Usage/{found=1} found && /^[^#]/{exit} found{sub(/^# ?/,""); print}' "$0"
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── helpers ───────────────────────────────────────────────────────────────────

# Return the dataset name (basename without path) currently being trained on,
# by scanning the last 25 lines of the training log.
# Looks for lines like: "Dataset seismic__2026.*__300ph..." 
active_dataset_from_log() {
    local log_file="$1"
    if [[ -z "$log_file" || ! -f "$log_file" ]]; then
        echo ""
        return
    fi
    # Grab the last "Dataset ..." line and extract the dataset directory name.
    # Training prints:  Dataset seismic__2026.XXXXXXXX__XXX, <array_keys> [N/M]
    tail -n 25 "$log_file" \
        | grep -oE 'Dataset seismic__[^, ]+' \
        | sed 's/^Dataset //' \
        | tail -n 1 || true
}

# List all seismic__* directories in ZARR_FOLDER sorted by modification time
# (oldest first).  Prints just the basename.
list_datasets_oldest_first() {
    # ls -dtr: oldest-first; -1 one entry per line; strip trailing slash
    ls -1dtr "$ZARR_FOLDER"/seismic__*/ 2>/dev/null \
        | while IFS= read -r d; do basename "${d%/}"; done
}

# Determine the run index to start from (auto-detect from existing run dirs).
detect_start_index() {
    local max=0
    # Look for _synthoseis_run_NNNN directories inside ZARR_FOLDER or cwd
    for dir in "$ZARR_FOLDER"/seismic__*_synthoseis_run_[0-9][0-9][0-9][0-9] \
                ./_synthoseis_run_[0-9][0-9][0-9][0-9]; do
        [[ -e "$dir" ]] || continue
        local n
        n=$(basename "$dir" | grep -oE '[0-9]{4}$' || true)
        [[ -n "$n" ]] && (( 10#$n > max )) && max=$((10#$n))
    done
    echo $(( max + 1 ))
}

# Parse "2h 15m 20s" / "15m 20s" / "20s" → total seconds.
parse_epoch_secs() {
    local timestr="$1"
    local h=0 m=0 s=0
    [[ "$timestr" =~ ([0-9]+)h ]] && h="${BASH_REMATCH[1]}"
    [[ "$timestr" =~ ([0-9]+)m ]] && m="${BASH_REMATCH[1]}"
    [[ "$timestr" =~ ([0-9]+)s ]] && s="${BASH_REMATCH[1]}"
    echo $(( h*3600 + m*60 + s ))
}

# Format seconds → "2h 15m 20s" / "15m 20s" / "20s".
fmt_duration() {
    local secs=$(( $1 ))
    local h=$(( secs / 3600 ))
    local m=$(( (secs % 3600) / 60 ))
    local s=$(( secs % 60 ))
    if (( h > 0 )); then
        printf "%dh %02dm %02ds" "$h" "$m" "$s"
    elif (( m > 0 )); then
        printf "%dm %02ds" "$m" "$s"
    else
        printf "%ds" "$s"
    fi
}

# Read the most recent "Epoch time: Xh Ym Zs | ..." line; return elapsed seconds (0 if absent).
read_epoch_secs_from_log() {
    local log_file="$1"
    [[ -z "$log_file" || ! -f "$log_file" ]] && echo 0 && return
    local raw
    raw=$(grep -oE 'Epoch time: [^|]+' "$log_file" 2>/dev/null | tail -n 1 \
        | sed 's/Epoch time: //' | xargs 2>/dev/null || true)
    [[ -z "$raw" ]] && echo 0 && return
    parse_epoch_secs "$raw"
}

# Read the most recent split line; return total dataset count (train + val), or 0.
# Parses "Restored split: N train, M val" or "Dataset split (N% val): N train, M val".
read_total_datasets_from_log() {
    local log_file="$1"
    [[ -z "$log_file" || ! -f "$log_file" ]] && echo 0 && return
    local line
    line=$(grep -E '(Restored split|Dataset split)' "$log_file" 2>/dev/null | tail -n 1 || true)
    [[ -z "$line" ]] && echo 0 && return
    local n_train=0 n_val=0
    [[ "$line" =~ ([0-9]+)\ train ]] && n_train="${BASH_REMATCH[1]}"
    [[ "$line" =~ ([0-9]+)\ val ]]   && n_val="${BASH_REMATCH[1]}"
    echo $(( n_train + n_val ))
}

# ── sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -d "$ZARR_FOLDER" ]]; then
    echo "ERROR: ZARR_FOLDER '$ZARR_FOLDER' does not exist." >&2
    exit 1
fi

if [[ ! -f "$SYNTHOSEIS_DIR/main.py" ]]; then
    echo "ERROR: '$SYNTHOSEIS_DIR/main.py' not found." >&2
    echo "       Set --synthoseis-dir to the directory containing synthoseis main.py" >&2
    exit 1
fi

if [[ ! -f "$CONFIG" && ! -f "$SYNTHOSEIS_DIR/$CONFIG" ]]; then
    echo "ERROR: config '$CONFIG' not found." >&2
    exit 1
fi

# Resolve config to an absolute path
[[ "$CONFIG" = /* ]] || CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"

# ── main loop ─────────────────────────────────────────────────────────────────
if [[ -z "$START_INDEX" ]]; then
    START_INDEX=$(detect_start_index)
fi

echo "=== Synthoseis Dataset Generation ==="
echo "  Runs requested : $NUM_RUNS"
echo "  Start index    : $(printf '%04d' "$START_INDEX")"
echo "  Config         : $CONFIG"
echo "  Zarr folder    : $ZARR_FOLDER"
[[ -n "$CHECK_LOG" ]] && echo "  Training log   : $CHECK_LOG"
if [[ "$NO_REPLACE" == "true" ]]; then
    echo "  Mode           : append-only (default)"
else
    echo "  Mode           : replace-oldest (--replace-oldest)"
fi
echo ""

# ── throttle state ────────────────────────────────────────────────────────────
# batch_start_time: wall-clock second when the current epoch-paced batch began.
# runs_in_batch:    successful replacements completed in this batch.
batch_start_time=$(date +%s)
runs_in_batch=0

for (( i=0; i<NUM_RUNS; i++ )); do
    run_idx=$(( START_INDEX + i ))
    run_tag="synthoseis_run_$(printf '%04d' "$run_idx")"
    log_file="_${run_tag}.log"

    echo "──────────────────────────────────────────"
    echo "Run $((i+1))/$NUM_RUNS  →  tag: $run_tag"
    echo "──────────────────────────────────────────"

    # ── 0. Throttle: ≤ ⌊total_datasets/3⌋ replacements per training epoch ──
    if [[ -n "$CHECK_LOG" ]]; then
        epoch_secs=$(read_epoch_secs_from_log "$CHECK_LOG")
        total_ds=$(read_total_datasets_from_log "$CHECK_LOG")
        if (( epoch_secs > 0 && total_ds > 0 )); then
            max_per_epoch=$(( total_ds / 3 ))
            (( max_per_epoch < 1 )) && max_per_epoch=1
            if (( runs_in_batch >= max_per_epoch )); then
                now=$(date +%s)
                elapsed=$(( now - batch_start_time ))
                remaining=$(( epoch_secs - elapsed ))
                if (( remaining > 0 )); then
                    echo "Throttle: ${runs_in_batch}/${max_per_epoch} replacements done" \
                         "in $(fmt_duration $elapsed) (epoch=$(fmt_duration $epoch_secs)," \
                         "total_ds=${total_ds})."
                    echo "  Sleeping $(fmt_duration $remaining) to pace with training epoch..."
                    sleep "$remaining"
                else
                    echo "Throttle: ${runs_in_batch}/${max_per_epoch} replacements done;" \
                         "epoch already elapsed — resetting batch."
                fi
                batch_start_time=$(date +%s)
                runs_in_batch=0
            else
                echo "Throttle: ${runs_in_batch}/${max_per_epoch} replacements this batch" \
                     "(epoch=$(fmt_duration $epoch_secs), total_ds=${total_ds})."
            fi
        else
            echo "Throttle: no epoch timing in log yet; running freely."
        fi
    fi

    # ── 1. Run synthoseis to produce a new dataset ──────────────────────────
    pushd "$SYNTHOSEIS_DIR" > /dev/null
    uv run python -u main.py \
        -n 1 \
        -r "_${run_tag}" \
        -c "$CONFIG" \
        --telemetry \
        --zarr-out essential \
        2>&1 | tee "$log_file"
    popd > /dev/null

    # ── 2. Find the zarr that was just created ───────────────────────────────
    # synthoseis names its output:  seismic__<timestamp>_<run_tag>/model_data.zarr
    # Find the newest seismic__ directory that contains the run_tag in its name.
    new_zarr_dir=$(find "$ZARR_FOLDER" -maxdepth 1 -type d \
        -name "*${run_tag}*" 2>/dev/null | sort | tail -n 1 || true)

    if [[ -z "$new_zarr_dir" ]]; then
        # Fallback: newest seismic__ dir overall
        new_zarr_dir=$(find "$ZARR_FOLDER" -maxdepth 1 -type d -name 'seismic__*' \
            -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | awk '{print $2}' || true)
    fi

    if [[ -z "$new_zarr_dir" ]]; then
        echo "WARNING: Could not locate newly created zarr directory; skipping replacement step." >&2
        continue
    fi

    echo "New dataset created: $(basename "$new_zarr_dir")"

    if [[ "$NO_REPLACE" == "true" ]]; then
        echo "Append-only mode enabled; skipping replacement/deletion step."
        echo ""
        continue
    fi

    # ── 3. Identify the active training dataset (must not be replaced) ───────
    active_ds=$(active_dataset_from_log "$CHECK_LOG")
    [[ -n "$active_ds" ]] && echo "Active training dataset (protected): $active_ds" \
                           || echo "No training log supplied; no dataset is protected."

    # ── 4. Find oldest dataset that is NOT the new one and NOT the active one ─
    oldest=""
    while IFS= read -r ds_basename; do
        # Skip the dataset we just created
        [[ "$ds_basename" == "$(basename "$new_zarr_dir")" ]] && continue
        # Skip the dataset currently in use by training
        [[ -n "$active_ds" && "$ds_basename" == "$active_ds" ]] && continue
        oldest="$ds_basename"
        break
    done < <(list_datasets_oldest_first)

    if [[ -z "$oldest" ]]; then
        echo "No replaceable dataset found; keeping all existing datasets."
        continue
    fi

    oldest_path="$ZARR_FOLDER/$oldest"
    echo "Replacing oldest unused dataset: $oldest"

    # Safety check: never remove a directory that doesn't look like a seismic dataset
    if [[ ! "$oldest_path" =~ seismic__ ]]; then
        echo "ERROR: '$oldest_path' does not look like a seismic dataset — refusing to delete." >&2
        continue
    fi

    rm -rf "$oldest_path"
    echo "Removed: $oldest_path"
    runs_in_batch=$(( runs_in_batch + 1 ))
    echo ""

done

echo "=== Generation complete ==="
