#!/usr/bin/env python3
"""Study candidate-point selection driven by geologic score.

This script:
- discovers seismic zarr datasets under /Users/donaldpg/synthoseis/fake_data
- finds the geologic score array, accepting either spelling used in the data
- generates a spatially spread 3D candidate pool inside the valid interior
- ranks candidates by geologic score and greedily accepts diverse points
- writes a Plotly HTML artifact at studies/geological_score_selection_study.html
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_DATA_ROOT = Path("/Users/donaldpg/synthoseis/fake_data")
DEFAULT_OUTPUT_HTML = Path("studies/geological_score_selection_study.html")
DEFAULT_OUTPUT_RANK_HIST_PNG = Path("studies/geological_score_rank_sampling_hist.png")
DEFAULT_DATASET_GLOB = "seismic__*/model_data.zarr"
DEFAULT_SCORE_KEYS = ("geological_score", "geologic_score")
DEFAULT_SEISMIC_KEYS = ("seismicCubes_cumsum_fullstack",)
DEFAULT_CANDIDATE_COUNTS = (5000,)
DEFAULT_TARGET_COUNT = 1000
DEFAULT_SAMPLE_MARGIN = 64
DEFAULT_MIN_SCORE = 0.5
DEFAULT_DIST_THRESH_START = 96
DEFAULT_DIST_THRESH_FLOOR = 32
DEFAULT_CANDIDATE_PROBES = 24
DEFAULT_SENSITIVITY_SEED = 7
DEFAULT_RANK_HIST_BINS = 10
DEFAULT_RANK_HIST_REPEATS = 5
DEFAULT_RANK_HIST_DRAWS_PER_REPEAT = 50
DEFAULT_RANK_HIST_ALPHA = 0.8


@dataclass(frozen=True)
class DatasetInfo:
    label: str
    zarr_path: Path
    score_key: str
    seismic_key: str
    score_shape: tuple[int, int, int]
    seismic_shape: tuple[int, int, int]


@dataclass
class SelectionResult:
    selected_points: np.ndarray
    selected_scores: np.ndarray
    candidate_points: np.ndarray
    candidate_scores: np.ndarray
    accepted_count: int
    final_dist_thresh: int
    threshold_history: list[int]
    min_score: float
    filtered_candidate_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study geologic-score driven point selection.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dataset-glob", type=str, default=DEFAULT_DATASET_GLOB)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML)
    parser.add_argument("--output-rank-hist-png", type=Path, default=DEFAULT_OUTPUT_RANK_HIST_PNG)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
    parser.add_argument("--sample-margin", type=int, default=DEFAULT_SAMPLE_MARGIN)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--candidate-count", type=str, default="auto")
    parser.add_argument(
        "--candidate-counts",
        type=str,
        default=",".join(str(v) for v in DEFAULT_CANDIDATE_COUNTS),
        help="Comma-separated candidate-count values used when --candidate-count auto",
    )
    parser.add_argument(
        "--sensitivity-dataset-limit",
        type=int,
        default=0,
        help="Limit the number of discovered datasets used for sensitivity testing; 0 means all.",
    )
    parser.add_argument("--candidate-probes", type=int, default=DEFAULT_CANDIDATE_PROBES)
    parser.add_argument("--dist-thresh-start", type=int, default=DEFAULT_DIST_THRESH_START)
    parser.add_argument("--dist-thresh-floor", type=int, default=DEFAULT_DIST_THRESH_FLOOR)
    parser.add_argument("--score-keys", type=str, default=",".join(DEFAULT_SCORE_KEYS))
    parser.add_argument("--seismic-keys", type=str, default=",".join(DEFAULT_SEISMIC_KEYS))
    parser.add_argument("--exclude-regex", type=str, default=r"(?i)(in[._-]?progress|partial|tmp|staging)")
    parser.add_argument("--rank-hist-bins", type=int, default=DEFAULT_RANK_HIST_BINS)
    parser.add_argument("--rank-hist-repeats", type=int, default=DEFAULT_RANK_HIST_REPEATS)
    parser.add_argument("--rank-hist-draws", type=int, default=DEFAULT_RANK_HIST_DRAWS_PER_REPEAT)
    parser.add_argument("--rank-hist-alpha", type=float, default=DEFAULT_RANK_HIST_ALPHA)
    return parser.parse_args()


def _parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _int_tuple(shape: Iterable[int]) -> tuple[int, int, int]:
    values = tuple(int(v) for v in shape)
    if len(values) != 3:
        raise ValueError(f"expected 3D shape, got {values}")
    return values


def discover_datasets(
    data_root: Path,
    dataset_glob: str,
    exclude_regex: str | None,
) -> list[Path]:
    exclude_re = re.compile(exclude_regex) if exclude_regex else None
    out: list[Path] = []
    for path in sorted(data_root.glob(dataset_glob)):
        if exclude_re is not None and exclude_re.search(str(path)):
            continue
        if path.is_dir():
            out.append(path)
    return out


def choose_array_key(root: zarr.Group, preferred_keys: Iterable[str]) -> str:
    keys = list(root.array_keys())
    for key in preferred_keys:
        if key in keys:
            return key
    raise KeyError(f"none of the preferred keys were found: {list(preferred_keys)}")


def resolve_dataset_info(zarr_path: Path, score_keys: Iterable[str], seismic_keys: Iterable[str]) -> DatasetInfo:
    root = zarr.open(str(zarr_path), mode="r")
    score_key = choose_array_key(root, score_keys)

    seismic_key = None
    for key in seismic_keys:
        if key in root.array_keys():
            seismic_key = key
            break
    if seismic_key is None:
        for key in root.array_keys():
            if key != score_key and len(root[key].shape) == 3 and key.startswith("seismic"):
                seismic_key = key
                break
    if seismic_key is None:
        raise KeyError(f"no seismic volume key found in {zarr_path}")

    score_shape = _int_tuple(root[score_key].shape)
    seismic_shape = _int_tuple(root[seismic_key].shape)
    label = zarr_path.parent.name
    return DatasetInfo(
        label=label,
        zarr_path=zarr_path,
        score_key=score_key,
        seismic_key=seismic_key,
        score_shape=score_shape,
        seismic_shape=seismic_shape,
    )


def usable_shape(info: DatasetInfo) -> tuple[int, int, int]:
    return tuple(min(a, b) for a, b in zip(info.score_shape, info.seismic_shape))


def interior_bounds(shape: tuple[int, int, int], margin: int) -> tuple[np.ndarray, np.ndarray]:
    low = np.full(3, int(margin), dtype=np.int32)
    high = np.asarray(shape, dtype=np.int32) - int(margin)
    if np.any(high < low):
        raise ValueError(f"shape {shape} is too small for margin {margin}")
    return low, high


def random_point_within(
    low: np.ndarray,
    high: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    return np.array([rng.integers(int(lo), int(hi) + 1) for lo, hi in zip(low, high)], dtype=np.int32)


def min_distance_to_existing(point: np.ndarray, existing: np.ndarray) -> float:
    if existing.size == 0:
        return float("inf")
    diffs = existing.astype(np.float32, copy=False) - point.astype(np.float32, copy=False)
    return float(np.sqrt(np.sum(diffs * diffs, axis=1)).min())


def best_candidate_point(
    existing: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    rng: np.random.Generator,
    probe_count: int,
) -> np.ndarray:
    probes = np.stack([random_point_within(low, high, rng) for _ in range(probe_count)], axis=0)
    if existing.size == 0:
        return probes[0]
    diffs = probes[:, None, :].astype(np.float32, copy=False) - existing[None, :, :].astype(np.float32, copy=False)
    min_distances = np.sqrt(np.sum(diffs * diffs, axis=2)).min(axis=1)
    return probes[int(np.argmax(min_distances))]


def generate_spread_candidates(
    shape: tuple[int, int, int],
    count: int,
    margin: int,
    seed: int,
    probe_count: int,
) -> np.ndarray:
    low, high = interior_bounds(shape, margin)
    rng = np.random.default_rng(int(seed))
    points: list[np.ndarray] = []
    existing = np.empty((0, 3), dtype=np.int32)
    for _ in range(int(count)):
        candidate = best_candidate_point(existing, low, high, rng, probe_count)
        points.append(candidate)
        existing = np.asarray(points, dtype=np.int32)
    return existing


def read_scores(score_array: zarr.Array, points: np.ndarray) -> np.ndarray:
    return np.asarray(score_array[points[:, 0], points[:, 1], points[:, 2]], dtype=np.float32)


def write_selection_json_near_zarr(
    info: DatasetInfo,
    result: SelectionResult,
    candidate_count: int,
    target_count: int,
    seed: int,
    dist_thresh_start: int,
    dist_thresh_floor: int,
    json_name: str = "geologic_score_selected_points.json",
) -> Path:
    out_path = info.zarr_path.parent / json_name
    payload = {
        "dataset": str(info.zarr_path),
        "score_key": info.score_key,
        "seismic_key": info.seismic_key,
        "candidate_count": int(candidate_count),
        "filtered_candidate_count": int(result.filtered_candidate_count),
        "accepted_count": int(result.accepted_count),
        "target_count": int(target_count),
        "min_score": float(result.min_score),
        "seed": int(seed),
        "dist_thresh_start": int(dist_thresh_start),
        "dist_thresh_floor": int(dist_thresh_floor),
        "dist_thresh_final": int(result.final_dist_thresh),
        "threshold_history": [int(v) for v in result.threshold_history],
        "selected_points": result.selected_points.astype(int).tolist(),
        "selected_scores": result.selected_scores.astype(float).tolist(),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def triangular_rank_sample_index(num_points: int, rng: np.random.Generator) -> int:
    if num_points <= 0:
        raise ValueError("num_points must be > 0")
    # Peak probability at rank 0 with tail toward lower-ranked points.
    index = int(rng.triangular(0.0, 0.0, float(num_points)))
    return min(max(index, 0), num_points - 1)


def sample_point_by_rank_probability(
    rng: np.random.Generator,
    points_json_path: Path | None = None,
    ranked_points: np.ndarray | None = None,
    ranked_scores: np.ndarray | None = None,
) -> tuple[np.ndarray, float, int]:
    points: np.ndarray
    scores: np.ndarray

    if points_json_path is not None:
        payload = json.loads(points_json_path.read_text(encoding="utf-8"))
        points = np.asarray(payload.get("selected_points", []), dtype=np.int32)
        scores = np.asarray(payload.get("selected_scores", []), dtype=np.float32)
    elif ranked_points is not None and ranked_scores is not None:
        points = np.asarray(ranked_points, dtype=np.int32)
        scores = np.asarray(ranked_scores, dtype=np.float32)
    else:
        raise ValueError("provide either points_json_path or both ranked_points and ranked_scores")

    if len(points) == 0:
        raise ValueError("no points available to sample")

    idx = triangular_rank_sample_index(len(points), rng)
    return points[idx], float(scores[idx]), idx


def write_rank_sampling_histogram_png(
    points_json_path: Path,
    output_png: Path,
    seed: int,
    bins: int,
    repeats: int,
    draws_per_repeat: int,
    alpha: float,
) -> Path:
    payload = json.loads(points_json_path.read_text(encoding="utf-8"))
    selected_points = np.asarray(payload.get("selected_points", []), dtype=np.int32)
    if len(selected_points) == 0:
        raise ValueError("cannot build rank histogram with no selected points in JSON")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    index_max = len(selected_points) - 1

    fig, ax = plt.subplots(figsize=(10, 6))
    for repeat_idx in range(int(repeats)):
        rng = np.random.default_rng(int(seed) + 1009 * repeat_idx)
        sampled_ranks: list[int] = []
        for _ in range(int(draws_per_repeat)):
            _, _, rank_idx = sample_point_by_rank_probability(
                rng=rng,
                points_json_path=points_json_path,
            )
            sampled_ranks.append(int(rank_idx))
        ax.hist(
            sampled_ranks,
            bins=int(bins),
            range=(0, max(1, index_max)),
            alpha=float(alpha),
            label=f"repeat {repeat_idx + 1}",
        )

    ax.set_title("JSON index sampling distribution (rank-weighted)")
    ax.set_xlabel("Index in selected_points JSON list")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_png, dpi=140)
    plt.close(fig)
    return output_png


def select_points(
    score_array: zarr.Array,
    candidate_points: np.ndarray,
    target_count: int,
    dist_thresh_start: int,
    dist_thresh_floor: int,
    min_score: float,
) -> SelectionResult:
    candidate_scores = read_scores(score_array, candidate_points)
    keep_mask = candidate_scores >= float(min_score)
    filtered_points = candidate_points[keep_mask]
    filtered_scores = candidate_scores[keep_mask]
    order = np.argsort(filtered_scores)[::-1]
    ordered_points = filtered_points[order]
    ordered_scores = filtered_scores[order]

    accepted_points: list[np.ndarray] = []
    accepted_scores: list[float] = []
    remaining = list(range(len(ordered_points)))
    dist_thresh = int(dist_thresh_start)
    threshold_history: list[int] = []

    if len(ordered_points) == 0:
        return SelectionResult(
            selected_points=np.empty((0, 3), dtype=np.int32),
            selected_scores=np.empty((0,), dtype=np.float32),
            candidate_points=ordered_points,
            candidate_scores=ordered_scores,
            accepted_count=0,
            final_dist_thresh=dist_thresh,
            threshold_history=threshold_history,
            min_score=float(min_score),
            filtered_candidate_count=int(len(filtered_points)),
        )

    accepted_points.append(ordered_points[0])
    accepted_scores.append(float(ordered_scores[0]))
    remaining = remaining[1:]

    while remaining and len(accepted_points) < int(target_count):
        added_this_pass = False
        threshold_history.append(dist_thresh)
        still_remaining: list[int] = []
        for idx in remaining:
            point = ordered_points[idx]
            reference = np.asarray(accepted_points[-3:], dtype=np.int32)
            if len(accepted_points) < 3:
                reference = np.asarray(accepted_points, dtype=np.int32)
            distance = min_distance_to_existing(point, reference)
            if distance > dist_thresh:
                accepted_points.append(point)
                accepted_scores.append(float(ordered_scores[idx]))
                added_this_pass = True
                if len(accepted_points) >= int(target_count):
                    break
            else:
                still_remaining.append(idx)
        remaining = still_remaining
        if len(accepted_points) >= int(target_count):
            break
        if not added_this_pass:
            if dist_thresh <= int(dist_thresh_floor):
                break
            dist_thresh = max(int(dist_thresh_floor), dist_thresh - 8)

    return SelectionResult(
        selected_points=np.asarray(accepted_points, dtype=np.int32),
        selected_scores=np.asarray(accepted_scores, dtype=np.float32),
        candidate_points=ordered_points,
        candidate_scores=ordered_scores,
        accepted_count=len(accepted_points),
        final_dist_thresh=dist_thresh,
        threshold_history=threshold_history,
        min_score=float(min_score),
        filtered_candidate_count=int(len(filtered_points)),
    )


def run_pipeline_for_dataset(
    info: DatasetInfo,
    candidate_count: int,
    seed: int,
    target_count: int,
    margin: int,
    probe_count: int,
    dist_thresh_start: int,
    dist_thresh_floor: int,
    min_score: float,
) -> SelectionResult:
    root = zarr.open(str(info.zarr_path), mode="r")
    score_array = root[info.score_key]
    usable = usable_shape(info)
    candidate_points = generate_spread_candidates(
        shape=usable,
        count=int(candidate_count),
        margin=int(margin),
        seed=int(seed),
        probe_count=int(probe_count),
    )
    return select_points(
        score_array=score_array,
        candidate_points=candidate_points,
        target_count=int(target_count),
        dist_thresh_start=int(dist_thresh_start),
        dist_thresh_floor=int(dist_thresh_floor),
        min_score=float(min_score),
    )


def choose_candidate_count(
    dataset_infos: list[DatasetInfo],
    counts: list[int],
    seed: int,
    target_count: int,
    margin: int,
    probe_count: int,
    dist_thresh_start: int,
    dist_thresh_floor: int,
    min_score: float,
) -> tuple[int, list[dict[str, float]]]:
    table: list[dict[str, float]] = []
    best_count = counts[-1]
    for idx, candidate_count in enumerate(counts):
        per_dataset_rates: list[float] = []
        for ds_idx, info in enumerate(dataset_infos):
            result = run_pipeline_for_dataset(
                info=info,
                candidate_count=candidate_count,
                seed=seed + idx * 17 + ds_idx,
                target_count=target_count,
                margin=margin,
                probe_count=probe_count,
                dist_thresh_start=dist_thresh_start,
                dist_thresh_floor=dist_thresh_floor,
                min_score=min_score,
            )
            per_dataset_rates.append(min(result.accepted_count, target_count) / float(target_count))
        mean_rate = float(np.mean(per_dataset_rates)) if per_dataset_rates else 0.0
        table.append({"candidate_count": float(candidate_count), "mean_rate": mean_rate})
        if mean_rate >= 1.0 - 1e-9 and best_count == counts[-1]:
            best_count = candidate_count
            break
    else:
        for row in table:
            if row["mean_rate"] >= 1.0 - 1e-9:
                best_count = int(row["candidate_count"])
                break
    return best_count, table


def _box_edges(shape: tuple[int, int, int]) -> list[go.Scatter3d]:
    xmax = int(shape[0]) - 1
    ymax = int(shape[1]) - 1
    zmax = int(shape[2]) - 1
    edges = [
        ((0, 0, 0), (xmax, 0, 0)),
        ((0, ymax, 0), (xmax, ymax, 0)),
        ((0, 0, zmax), (xmax, 0, zmax)),
        ((0, ymax, zmax), (xmax, ymax, zmax)),
        ((0, 0, 0), (0, ymax, 0)),
        ((xmax, 0, 0), (xmax, ymax, 0)),
        ((0, 0, zmax), (0, ymax, zmax)),
        ((xmax, 0, zmax), (xmax, ymax, zmax)),
        ((0, 0, 0), (0, 0, zmax)),
        ((xmax, 0, 0), (xmax, 0, zmax)),
        ((0, ymax, 0), (0, ymax, zmax)),
        ((xmax, ymax, 0), (xmax, ymax, zmax)),
    ]
    traces: list[go.Scatter3d] = []
    for start, end in edges:
        traces.append(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines",
                line=dict(color="rgba(255,255,255,0.35)", width=3),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return traces


def build_figure(
    info: DatasetInfo,
    result: SelectionResult,
    candidate_count: int,
    seed: int,
    dist_thresh_start: int,
    dist_thresh_floor: int,
) -> go.Figure:
    dims = info.seismic_shape
    # Keep x/y proportional to dataset size, but compress z for a laptop-friendly view.
    xy_max = float(max(dims[0], dims[1], 1))
    z_scale = 0.285
    aspect_ratio = dict(
        x=float(dims[0]) / xy_max,
        y=float(dims[1]) / xy_max,
        z=max(0.1, (float(dims[2]) / xy_max) * z_scale),
    )
    selected = result.selected_points
    selected_scores = result.selected_scores
    fig = go.Figure()
    for trace in _box_edges(dims):
        fig.add_trace(trace)
    fig.add_trace(
        go.Scatter3d(
            x=selected[:, 0] if len(selected) else [],
            y=selected[:, 1] if len(selected) else [],
            z=selected[:, 2] if len(selected) else [],
            mode="markers",
            name="selected points",
            marker=dict(
                size=3,
                color=selected_scores if len(selected_scores) else [],
                colorscale="Viridis",
                opacity=0.9,
                colorbar=dict(title=info.score_key),
            ),
            hovertemplate="x=%{x}<br>y=%{y}<br>z=%{z}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title=(
            f"Geologic-score point selection for {info.label}<br>"
            f"{info.zarr_path} | score={info.score_key} | seismic={info.seismic_key}<br>"
            f"candidate_count={candidate_count} | seed={seed} | dist_thresh={dist_thresh_start}->{result.final_dist_thresh} floor={dist_thresh_floor}"
        ),
        margin=dict(l=0, r=0, t=80, b=0),
        legend=dict(itemsizing="constant"),
        scene=dict(
            xaxis=dict(title="X", range=[0, dims[0] - 1]),
            yaxis=dict(title="Y", range=[0, dims[1] - 1]),
            # Depth axis: show 0 at the top and increasing depth downward.
            zaxis=dict(title="Depth (Z)", autorange="reversed", range=[0, dims[2] - 1]),
            aspectmode="manual",
            aspectratio=aspect_ratio,
        ),
    )
    return fig


def print_sensitivity_table(table: list[dict[str, float]]) -> None:
    print("Sensitivity results:")
    for row in table:
        print(f"  candidate_count={int(row['candidate_count'])} mean_acceptance_rate={row['mean_rate']:.3f}")


def main() -> int:
    args = parse_args()
    data_root = args.data_root
    if not data_root.is_dir():
        print(f"ERROR: data root does not exist: {data_root}", file=sys.stderr)
        return 2

    score_keys = _parse_csv(args.score_keys)
    seismic_keys = _parse_csv(args.seismic_keys)
    dataset_paths = discover_datasets(data_root, args.dataset_glob, args.exclude_regex)
    if not dataset_paths:
        print("ERROR: no matching seismic datasets discovered", file=sys.stderr)
        return 1

    dataset_infos: list[DatasetInfo] = []
    for path in dataset_paths:
        try:
            info = resolve_dataset_info(path, score_keys=score_keys, seismic_keys=seismic_keys)
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping {path}: {exc}", file=sys.stderr)
            continue
        dataset_infos.append(info)

    if not dataset_infos:
        print("ERROR: no datasets had the required score and seismic arrays", file=sys.stderr)
        return 1

    print(f"Discovered {len(dataset_infos)} usable datasets:")
    for info in dataset_infos:
        print(
            f"  - {info.label}: score={info.score_key} {info.score_shape}, seismic={info.seismic_key} {info.seismic_shape}",
            file=sys.stderr,
        )

    candidate_counts = [int(v) for v in _parse_csv(args.candidate_counts)]
    if not candidate_counts:
        print("ERROR: --candidate-counts must contain at least one integer", file=sys.stderr)
        return 2
    candidate_counts = sorted(set(candidate_counts))

    if args.sensitivity_dataset_limit and args.sensitivity_dataset_limit > 0:
        sensitivity_infos = dataset_infos[: int(args.sensitivity_dataset_limit)]
    else:
        sensitivity_infos = dataset_infos

    chosen_candidate_count = int(args.candidate_count) if args.candidate_count != "auto" else None
    sensitivity_table: list[dict[str, float]] = []
    if chosen_candidate_count is None:
        chosen_candidate_count, sensitivity_table = choose_candidate_count(
            dataset_infos=sensitivity_infos,
            counts=candidate_counts,
            seed=int(args.seed),
            target_count=int(args.target_count),
            margin=int(args.sample_margin),
            probe_count=int(args.candidate_probes),
            dist_thresh_start=int(args.dist_thresh_start),
            dist_thresh_floor=int(args.dist_thresh_floor),
            min_score=float(args.min_score),
        )
        print_sensitivity_table(sensitivity_table)
        print(f"Chosen candidate count: {chosen_candidate_count}")

    if args.dataset_path is not None:
        selected_info = resolve_dataset_info(args.dataset_path, score_keys=score_keys, seismic_keys=seismic_keys)
    else:
        index = max(0, min(int(args.dataset_index), len(dataset_infos) - 1))
        selected_info = dataset_infos[index]

    result = run_pipeline_for_dataset(
        info=selected_info,
        candidate_count=int(chosen_candidate_count),
        seed=int(args.seed),
        target_count=int(args.target_count),
        margin=int(args.sample_margin),
        probe_count=int(args.candidate_probes),
        dist_thresh_start=int(args.dist_thresh_start),
        dist_thresh_floor=int(args.dist_thresh_floor),
        min_score=float(args.min_score),
    )

    print(
        f"Selected {result.accepted_count} points from {selected_info.label} using {chosen_candidate_count} candidates; "
        f"filtered_candidates={result.filtered_candidate_count}; "
        f"min_score={result.min_score:.3f}; "
        f"final_dist_thresh={result.final_dist_thresh}",
        file=sys.stderr,
    )

    json_output_path = write_selection_json_near_zarr(
        info=selected_info,
        result=result,
        candidate_count=int(chosen_candidate_count),
        target_count=int(args.target_count),
        seed=int(args.seed),
        dist_thresh_start=int(args.dist_thresh_start),
        dist_thresh_floor=int(args.dist_thresh_floor),
    )

    rng = np.random.default_rng(int(args.seed) + 991)
    sampled_point, sampled_score, sampled_rank = sample_point_by_rank_probability(
        rng=rng,
        ranked_points=result.selected_points,
        ranked_scores=result.selected_scores,
    )

    fig = build_figure(
        info=selected_info,
        result=result,
        candidate_count=int(chosen_candidate_count),
        seed=int(args.seed),
        dist_thresh_start=int(args.dist_thresh_start),
        dist_thresh_floor=int(args.dist_thresh_floor),
    )
    output_path = args.output_html
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True, auto_open=False)

    rank_hist_png_path = write_rank_sampling_histogram_png(
        points_json_path=json_output_path,
        output_png=args.output_rank_hist_png,
        seed=int(args.seed) + 2026,
        bins=int(args.rank_hist_bins),
        repeats=int(args.rank_hist_repeats),
        draws_per_repeat=int(args.rank_hist_draws),
        alpha=float(args.rank_hist_alpha),
    )

    report = {
        "selected_dataset": str(selected_info.zarr_path),
        "score_key": selected_info.score_key,
        "seismic_key": selected_info.seismic_key,
        "candidate_count": int(chosen_candidate_count),
        "accepted_count": int(result.accepted_count),
        "filtered_candidate_count": int(result.filtered_candidate_count),
        "target_count": int(args.target_count),
        "min_score": float(result.min_score),
        "final_dist_thresh": int(result.final_dist_thresh),
        "selected_points": result.selected_points.tolist(),
        "selection_json": str(json_output_path),
        "rank_sampling_histogram_png": str(rank_hist_png_path),
        "triangular_sample": {
            "rank_index": int(sampled_rank),
            "point_xyz": sampled_point.astype(int).tolist(),
            "score": float(sampled_score),
        },
    }
    print(json.dumps(report, indent=2))
    print(f"Wrote HTML: {output_path}")
    print(f"Wrote selected-point JSON: {json_output_path}")
    print(f"Wrote rank-sampling histogram PNG: {rank_hist_png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())