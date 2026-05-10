# Session Summary: Thermal Guard and Training Ops Hardening

**Date:** 2026-05-04
**Scope:** training orchestration, thermal safety, dataset lifecycle handling, docs
**Outcome:** Dynamic train/val management stabilized for concurrent dataset generation,
validation progress logging added, in-progress datasets excluded from discovery,
thermal pausing implemented with checkpoint+cooldown, and macOS `powermetrics`
compatibility extended to pressure-based fallback with configurable trip threshold.

---

## Highlights

- Added explicit train/val dataset count controls (`num_train=4`, `num_val=2`) across:
  - `train.py`
  - `train_multi_datasets.sh`
- Enforced side exclusivity over time:
  - dataset once assigned to train never migrates to val (and vice-versa)
  - implemented via append-only historical assignment lists
- Excluded in-progress datasets:
  - discovery ignores `seismic__*` entries that still have a sibling `temp_folder__*`
- Added per-batch validation progress output matching train style:
  - dataset header + periodic `Val batch ...` logging
- Added thermal guard:
  - periodic checks every N batches
  - save `checkpoints/thermal_latest.pt` on thermal trip
  - pause for cooldown seconds, then resume in-process (no exit behavior)

---

## Files Changed

| File | Change Summary |
|---|---|
| `train.py` | Count-based split arguments (`--num_train`, `--num_val`), active-window loader selection, side-exclusivity logic, validation progress logging, thermal guard class, startup thermal monitor status, periodic train-batch thermal status, configurable pressure trip level |
| `train_multi_datasets.sh` | Added `--num-train`/`--num-val`; added thermal passthrough flags (`--thermal-max-c`, `--thermal-cooldown-sec`, `--thermal-check-every-batches`, `--thermal-pressure-trip-level`) |
| `src/synthoseis_pre_train/gpu_utils.py` | Added macOS thermal helpers with resilient `powermetrics` parsing and command fallback (`--once`, `-n 1 -i 1000`), plus thermal pressure extraction |
| `README.md` | Updated training invocation and operational guidance for sudo-backed thermal checks and pressure fallback |

---

## Thermal Guard Design (Final)

### Signals

1. **Primary:** CPU temperature in Celsius, when available
2. **Fallback:** thermal pressure level (`Nominal`, `Fair`, `Serious`, `Critical`)

### Pause Trigger

- Temperature trigger: `temp_c >= --thermal_max_c` (if temperature available)
- Pressure trigger: `level >= --thermal_pressure_trip_level`
  - default: `serious`
  - configurable: `off|nominal|fair|serious|critical`

### Action on Trip

- Save thermal checkpoint: `checkpoints/thermal_latest.pt`
- Sleep for `--thermal_cooldown_sec`
- Resume training loop in the same process

### Why pressure fallback was needed

On this macOS build, `powermetrics` does not expose a parseable CPU die temperature in
some modes, but it does report thermal pressure reliably. `osx-cpu-temp` was tested and
returned `0.0C`, so it was not used for control logic.

---

## Operational Notes (Mac mini)

- Thermal monitoring requires non-interactive sudo access for `powermetrics` during long runs.
- Recommended run pattern:
  - `sudo -v` before launch
  - optional sudo keepalive loop while training
- Startup status lines in `train.py` now make thermal mode explicit:
  - available via numeric CPU temp, or
  - available via pressure fallback, or
  - unavailable

---

## Known Risks / Follow-ups

- `tests/test_augmentation_geometry.py` remains in a collection-error state (`pytest` exit code 2) from prior work and was not resolved in this session.
- Thermal pressure fallback is robust for safety, but less granular than numeric temperature.
- If future macOS updates alter `powermetrics` output format again, only parser adjustments should be needed in `gpu_utils.py`.

---

## Suggested Next Session Start Checklist

1. Verify smoke run prints expected thermal mode at startup.
2. Reproduce and fix `tests/test_augmentation_geometry.py` collection error.
3. Decide whether to keep pressure trip default at `serious` or tune to `fair` for earlier pausing.
