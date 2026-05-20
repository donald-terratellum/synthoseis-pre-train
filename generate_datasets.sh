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
#   --check-log FILE              Training log file to read epoch elapsed time
#                                 for generation pacing
#   --min-free-gb N               Minimum free disk space (GB) required before
#                                 starting a synthoseis run (default: 50)
#   --disk-recheck-sec N          Seconds to sleep before rechecking disk space
#                                 when below threshold (default: 3600)
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
TARGET_NEW_PER_EPOCH=2
MIN_FREE_GB=50
RECHECK_SLEEP_SEC=3600

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--num-runs)          NUM_RUNS="$2";               shift 2 ;;
        -c|--config)            CONFIG="$2";                 shift 2 ;;
        --synthoseis-dir)       SYNTHOSEIS_DIR="$2";         shift 2 ;;
        -d|--synthoseis-zarr-folder) ZARR_FOLDER="$2";      shift 2 ;;
        --check-log)            CHECK_LOG="$2";              shift 2 ;;
        --min-free-gb)          MIN_FREE_GB="$2";            shift 2 ;;
        --disk-recheck-sec)     RECHECK_SLEEP_SEC="$2";      shift 2 ;;
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

# Return free space in KiB for the filesystem containing the given path.
get_free_kib() {
    local path="$1"
    df -Pk "$path" 2>/dev/null | awk 'NR==2 {print $4}'
}

# Block until at least min_free_gb is available on the target filesystem.
wait_for_min_free_space() {
    local target_path="$1"
    local min_free_gb="$2"
    local sleep_sec="$3"
    local min_free_kib=$(( min_free_gb * 1024 * 1024 ))

    while true; do
        local free_kib
        free_kib=$(get_free_kib "$target_path")

        if [[ -z "$free_kib" || ! "$free_kib" =~ ^[0-9]+$ ]]; then
            echo "WARNING: Could not determine free space for '$target_path'; sleeping $(fmt_duration "$sleep_sec") before retry." >&2
            sleep "$sleep_sec"
            continue
        fi

        local free_gb
        free_gb=$(awk -v kib="$free_kib" 'BEGIN { printf "%.2f", kib/1024/1024 }')

        if (( free_kib >= min_free_kib )); then
            echo "Disk space check: ${free_gb} GB free (required: ${min_free_gb} GB) — proceeding."
            return 0
        fi

        echo "Disk space check: ${free_gb} GB free (required: ${min_free_gb} GB)."
        echo "  Not enough free disk for synthoseis staging (~25 GB/dataset + safety margin)."
        echo "  Sleeping $(fmt_duration "$sleep_sec") before rechecking."
        sleep "$sleep_sec"
    done
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

# ── sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -d "$ZARR_FOLDER" ]]; then
    echo "ERROR: ZARR_FOLDER '$ZARR_FOLDER' does not exist." >&2
    exit 1
fi

if [[ ! "$MIN_FREE_GB" =~ ^[0-9]+$ ]] || (( MIN_FREE_GB <= 0 )); then
    echo "ERROR: --min-free-gb must be a positive integer (got: '$MIN_FREE_GB')." >&2
    exit 1
fi

if [[ ! "$RECHECK_SLEEP_SEC" =~ ^[0-9]+$ ]] || (( RECHECK_SLEEP_SEC <= 0 )); then
    echo "ERROR: --disk-recheck-sec must be a positive integer (got: '$RECHECK_SLEEP_SEC')." >&2
    exit 1
fi

if [[ ! -f "$SYNTHOSEIS_DIR/main.py" ]]; then
    echo "ERROR: '$SYNTHOSEIS_DIR/main.py' not found." >&2
    echo "       Set --synthoseis-dir to the directory containing synthoseis main.py" >&2
    exit 1
fi

if [[ "$CONFIG" = /* ]]; then
    if [[ ! -f "$CONFIG" ]]; then
        echo "ERROR: config '$CONFIG' not found." >&2
        exit 1
    fi
else
    if [[ -f "$CONFIG" ]]; then
        CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
    elif [[ -f "$SYNTHOSEIS_DIR/$CONFIG" ]]; then
        CONFIG="$(cd "$SYNTHOSEIS_DIR/$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
    else
        echo "ERROR: config '$CONFIG' not found." >&2
        echo "       Looked in current directory and under --synthoseis-dir ('$SYNTHOSEIS_DIR')." >&2
        exit 1
    fi
fi

# ── main loop ─────────────────────────────────────────────────────────────────
if [[ -z "$START_INDEX" ]]; then
    START_INDEX=$(detect_start_index)
fi

echo "=== Synthoseis Dataset Generation ==="
echo "  Runs requested : $NUM_RUNS"
echo "  Start index    : $(printf '%04d' "$START_INDEX")"
echo "  Config         : $CONFIG"
echo "  Zarr folder    : $ZARR_FOLDER"
echo "  Min free space : ${MIN_FREE_GB} GB"
echo "  Recheck sleep  : $(fmt_duration "$RECHECK_SLEEP_SEC") (${RECHECK_SLEEP_SEC}s)"
[[ -n "$CHECK_LOG" ]] && echo "  Training log   : $CHECK_LOG"
if [[ "$NO_REPLACE" == "true" ]]; then
    echo "  Mode           : append-only (default)"
else
    echo "  Mode           : replace-oldest (--replace-oldest)"
fi
echo ""

for (( i=0; i<NUM_RUNS; i++ )); do
    run_idx=$(( START_INDEX + i ))
    run_tag="synthoseis_run_$(printf '%04d' "$run_idx")"
    log_file="_${run_tag}.log"

    echo "──────────────────────────────────────────"
    echo "Run $((i+1))/$NUM_RUNS  →  tag: $run_tag"
    echo "──────────────────────────────────────────"

    # Safeguard: synthoseis can require substantial temporary staging space
    # before it deletes non-essential zarr volumes.
    wait_for_min_free_space "$ZARR_FOLDER" "$MIN_FREE_GB" "$RECHECK_SLEEP_SEC"

    # ── 1. Run synthoseis to produce a new dataset ──────────────────────────
    run_start_secs=$(date +%s)
    pushd "$SYNTHOSEIS_DIR" > /dev/null
    uv run python -u main.py \
        -n 1 \
        -r "_${run_tag}" \
        -c "$CONFIG" \
        --telemetry \
        --zarr-out essential \
        2>&1 | tee "$log_file"
    popd > /dev/null
    run_elapsed_secs=$(( $(date +%s) - run_start_secs ))

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
        if [[ -n "$CHECK_LOG" ]]; then
            epoch_secs=$(read_epoch_secs_from_log "$CHECK_LOG")
            if (( epoch_secs > 0 )); then
                target_interval_secs=$(( epoch_secs / TARGET_NEW_PER_EPOCH ))
                (( target_interval_secs < 1 )) && target_interval_secs=1
                sleep_secs=$(( target_interval_secs - run_elapsed_secs ))
                if (( sleep_secs > 0 )); then
                    echo "Throttle: generation took $(fmt_duration $run_elapsed_secs);" \
                         "target cadence is ${TARGET_NEW_PER_EPOCH}/epoch" \
                         "(epoch=$(fmt_duration $epoch_secs))."
                    echo "  Sleeping $(fmt_duration $sleep_secs) to pace generation."
                    sleep "$sleep_secs"
                else
                    echo "Throttle: generation took $(fmt_duration $run_elapsed_secs)," \
                         "already at/above target interval $(fmt_duration $target_interval_secs)."
                fi
            else
                echo "Throttle: no epoch timing in log yet; running freely."
            fi
        fi
        echo ""
        continue
    fi

    # ── 3. Find oldest dataset that is NOT the new one ───────────────────────
    oldest=""
    while IFS= read -r ds_basename; do
        # Skip the dataset we just created
        [[ "$ds_basename" == "$(basename "$new_zarr_dir")" ]] && continue
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

    if [[ -n "$CHECK_LOG" ]]; then
        epoch_secs=$(read_epoch_secs_from_log "$CHECK_LOG")
        if (( epoch_secs > 0 )); then
            target_interval_secs=$(( epoch_secs / TARGET_NEW_PER_EPOCH ))
            (( target_interval_secs < 1 )) && target_interval_secs=1
            sleep_secs=$(( target_interval_secs - run_elapsed_secs ))
            if (( sleep_secs > 0 )); then
                echo "Throttle: generation took $(fmt_duration $run_elapsed_secs);" \
                     "target cadence is ${TARGET_NEW_PER_EPOCH}/epoch" \
                     "(epoch=$(fmt_duration $epoch_secs))."
                echo "  Sleeping $(fmt_duration $sleep_secs) to pace generation."
                sleep "$sleep_secs"
            else
                echo "Throttle: generation took $(fmt_duration $run_elapsed_secs)," \
                     "already at/above target interval $(fmt_duration $target_interval_secs)."
            fi
        else
            echo "Throttle: no epoch timing in log yet; running freely."
        fi
    fi

    echo ""

done

echo "=== Generation complete ==="
