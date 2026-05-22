# Session Summary: Geologic Score Study and Dataloader Integration (2026-05-22)

## Context and Goals
- Objective: design, implement, validate, and operationalize geologic-score-driven 3D crop-center selection for seismic pre-training.
- Scope:
  - build a study workflow to evaluate ranked `(x, y, z)` selection from zarr geologic score volumes.
  - add reproducible artifacts (HTML 3D visualization, JSON center lists, and rank-sampling diagnostics).
  - integrate the same center-selection strategy into training and validation dataloading.
- Key behavior targets:
  - enforce minimum score gating (`geologic_score >= 0.5`).
  - persist computed center lists so expensive score reads do not repeat each epoch.
  - train: new independent draw per example.
  - validation: one fixed draw per dataset, stable across epochs and stop/resume.

## What Was Done
- Created study planning and implementation artifacts in `studies/`.
- Implemented geologic-score study runner with:
  - candidate generation and ranked selection.
  - thresholded acceptance logic and distance backoff floor.
  - helper functions for JSON persistence near dataset zarr folders.
  - helper sampling by rank-weighted triangular probability.
  - Plotly HTML 3D output and matplotlib PNG rank-sampling histogram output.
- Tuned visualization:
  - depth axis orientation corrected (0 at top).
  - compressed z display aspect for laptop usability.
- Updated study defaults to:
  - `--target-count 1000`
  - `--candidate-counts 5000`
- Integrated geologic-score center selection into production loading path:
  - added centered extraction support in augmentation path.
  - added dataloader-level persisted ranked points and fixed validation center behavior.
  - added train.py CLI wiring for enabling/disabling and controlling geologic-score selection parameters.

## How It Was Done
- Study-first workflow:
  - wrote and refined implementation plan (`md` + `html`).
  - validated assumptions against real zarr schema in fake-data datasets.
- Implementation workflow:
  - built study script under `studies/` with persistence and diagnostics.
  - tested iteratively with bounded smoke runs.
  - added dataloader and augmentation integration after study behavior proved stable.
  - introduced train.py flags and loader kwargs to expose controls without hardcoding.
- Validation workflow:
  - compile checks for modified Python modules.
  - full suite run on Mac mini environment.

## When Was It Done and By Whom
- Date: 2026-05-22
- Human operator: Donald Griffith
- Agent: GitHub Copilot (GPT-5.3-Codex)
- Primary implementation commits (author: Donald Griffith):
  - `50ed273` (2026-05-22T18:04:15-05:00)
  - `15be197` (2026-05-22T18:04:26-05:00)

## Basic Info
### Relevant Commits
- `50ed273` Add geologic-score study tooling and planning artifacts
- `15be197` Integrate geologic-score center selection into training dataloaders

### Files Involved
- `pyproject.toml`
- `train.py`
- `src/synthoseis_pre_train/augmentation.py`
- `src/synthoseis_pre_train/dataloader.py`
- `studies/.gitignore`
- `studies/geological_score_selection_plan.md`
- `studies/geological_score_selection_plan.html`
- `studies/geological_score_selection_study.py`

### Validation Outcome
- Mac mini `uv run python -m pytest`: **29 passed, 0 failed**.
- Observed warnings were non-blocking (`pin_memory` on MPS, expected bad-path warning in tests).

## Next and Future Follow-Up Suggestions
1. Add explicit unit tests for geologic-score dataloader behaviors:
   - persisted points file reuse.
   - fixed validation center invariance across epochs.
   - minimum score enforcement.
2. Add lightweight telemetry to training logs:
   - selected rank distribution per epoch.
   - center reuse statistics for validation datasets.
3. Consider periodic refresh policy for ranked point JSON when geologic score volumes are regenerated.
4. Add optional deterministic seed controls specifically for triangular rank sampling during ablations.
