# Geological Score Selection Study Plan

## Goal
Create a study that selects 500 ordered 128x128x128 examples from any discovered seismic zarr dataset under `/Users/donaldpg/synthoseis/fake_data`, using `geological_score` to bias selection toward more complex and interesting geologic patterns while still preserving spatial diversity.

## Scope and Constraints
- Search root: `/Users/donaldpg/synthoseis/fake_data`
- Example size: `128x128x128`
- Boundary margin: `64` voxels on all axes
- Candidate pool: `1000` random 3D points per dataset
- Final selection: `500` accepted points per dataset, if available
- Output: HTML 3D plot that can be opened and rotated in a browser
- Existing study style reference: `scripts/losses_study.py`
- Plot style reference: Plotly 3D HTML output from the upstream synthoseis `plot_3D_faults_plot` example

## Proposed File Layout
- `studies/`
  - `geological_score_selection_plan.md` this plan
  - `geological_score_selection_plan.html` browser-viewable version of this plan
  - future implementation files for the study entrypoint, helper functions, and plotting output

## Phase 1: Requirements Lock and Dataset Discovery
### Tasks
1. Confirm the zarr array key name is exactly `geological_score`.
2. Define how any zarr dataset under the fake-data root is discovered and filtered.
3. Confirm the seismic array shape and axis order for reading score values and building 128-cube crop centers.
4. Define how the 64-voxel interior margin is enforced for all axes.

### Review Gate
- Check that the study can enumerate all candidate datasets without loading full volumes unnecessarily.
- Check that the score key exists and is readable before sampling begins.

### Validation
- Run a discovery-only smoke check against the fake-data root.
- Confirm at least one dataset exposes `geological_score` and a compatible seismic volume shape.

## Phase 2: Candidate Generation and Score Ranking
### Tasks
1. Generate 1000 interior candidate centers in 3D using a fast, spatially spread method inspired by Mitchell’s best candidate or Poisson-disc sampling.
2. Keep every candidate at least 64 voxels from each dataset boundary.
3. Read `geological_score` at each candidate center.
4. Sort candidates from highest score to lowest score before any greedy acceptance logic runs.
5. Run a sensitivity test across multiple discovered datasets and average the acceptance success rate for each candidate-count setting so the default can be increased if 1000 does not reliably produce 500 accepted outputs.

### Review Gate
- Check that the sampler is interior-safe and reproducible from a seed.
- Check that score ranking happens before distance-based rejection.
- Check that the sensitivity test is enough to justify the final default starting-point count.

### Validation
- Sample one dataset and inspect the top-ranked candidates.
- Confirm the candidate list contains exactly 1000 points unless the interior region is too small.
- Compare at least two larger candidate-count settings against the 1000-point baseline across multiple discovered datasets, and choose the smallest setting whose mean acceptance success rate reliably yields 500 accepted points.

## Phase 3: Greedy Selection with Adaptive Spacing
### Tasks
1. Accept the first point as the highest-scoring candidate.
2. For each remaining candidate, compute 3D distance to the previous 3 accepted points, or to all accepted points if fewer than 3 exist.
3. Accept a point only if it is farther than `dist_thresh` from all of the previous 3 accepted points, or from all accepted points if fewer than 3 exist.
4. If no unselected point passes, reduce `dist_thresh` by 8 and rescan the remaining candidates, but stop reducing once `dist_thresh` reaches 32.
5. Stop once 500 points are accepted or no valid candidates remain.

### Review Gate
- Check that the distance rule uses the intended reference set.
- Check that threshold backoff is monotonic and does not skip candidates incorrectly.
- Check that the 32 floor is enforced and cannot be reduced past the limit.

### Validation
- Run one deterministic selection pass.
- Verify the output ordering and acceptance count.
- Verify all selected points remain inside the allowed interior margin.
- Verify that the backoff stops at 32 even if additional candidates still fail the threshold.

## Phase 4: 3D HTML Plot Output
### Tasks
1. Create a Plotly 3D scatter plot of the selected 500 points.
2. Draw the plot inside a box matching the full zarr seismic dataset dimensions.
3. Include dataset path, selected count, seed, and threshold schedule in the title or subtitle.
4. Write the result to `studies/geological_score_selection_study.html` as a self-contained HTML file for browser viewing and rotation.

### Review Gate
- Check that the HTML output is self-contained and portable.
- Check that the plot visually shows the full dataset bounding box and the selected point cloud.

### Validation
- Open the generated HTML artifact in a browser.
- Confirm the axes and bounding box match the source dataset dimensions.

## Phase 5: End-to-End Review and Testing
### Tasks
1. Run the full study on one discovered dataset end to end.
2. Verify the selected point list, score ordering, and spacing behavior.
3. Verify the HTML artifact is written to the expected study output location.
4. Record any deviations from the initial assumptions.

### Final Review Gate
- Perform an end-to-end code review of the study files.
- Confirm the implementation is minimal, reproducible, and understandable.

### Final Validation
- Run a study smoke test on at least one dataset.
- Confirm the output count is 500 when enough valid interior candidates exist.
- Confirm the browser HTML plot is generated successfully.

## Session Summary
The final deliverable for this study should document:
1. The exact dataset discovery rule used under the fake-data root.
2. The exact `geological_score` sampling and ranking behavior.
3. The final distance threshold schedule.
4. The selected output path for the HTML plot.
5. Any assumptions made about zarr axis order or score resolution.
