"""
Training Script for Seismic 3D Mamba Pre-training
===================================================
"""

import os
import random
import time
import math
import platform
from datetime import datetime, timedelta
import inspect
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

try:
    from torchinfo import summary as torchinfo_summary
except Exception:
    torchinfo_summary = None

from synthoseis_pre_train.dataloader import create_dataloader
from synthoseis_pre_train.gpu_utils import (
    get_default_device,
    get_memory_info,
    print_device_summary,
    autocast_context,
    create_grad_scaler,
    get_cpu_temperature_c,
    get_thermal_pressure_level,
    ProcessTreeCsvMonitor,
)
from synthoseis_pre_train.losses import (
    SSIMMSELoss3D,
    CompositeClusterAwareLoss,
    SlidingWindowStatsLoss3D,
)
from synthoseis_pre_train.models import create_model, _MAMBA_AVAILABLE
from synthoseis_pre_train.plotting import make_4panel_figure, make_crosssection_figure


# Defensive runtime scrub: set all Malloc* vars to "0" (explicit disable signal
# to libmalloc) rather than unsetting — absent vars may still trigger warnings
# on some macOS versions; "0" is the documented way to disable stack logging.
if platform.system() == "Darwin":
    for _k in list(os.environ.keys()):
        if _k.startswith("Malloc"):
            os.environ[_k] = "0"
    # Ensure the two key vars are present even if not already in env
    os.environ["MallocStackLogging"] = "0"
    os.environ["MallocStackLoggingNoCompact"] = "0"


def _save_checkpoint(path: Path, model, optimizer, scaler, epoch: int,
                     train_loss: float, val_loss: float,
                     train_paths: list = None, val_paths: list = None,
                     ds_idx: int = -1,
                     ema_state: dict | None = None) -> None:
    """Save a resumable checkpoint.  ds_idx=-1 means end-of-epoch."""
    torch.save({
        "epoch":       epoch,
        "ds_idx":      ds_idx,
        "model":       model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "scaler":      scaler.state_dict() if scaler is not None else None,
        "train_loss":  train_loss,
        "val_loss":    val_loss,
        "train_paths": train_paths,
        "val_paths":   val_paths,
        "ema_state":   ema_state,
    }, path)


def _format_elapsed_dhm(start_time: float) -> str:
    """Format elapsed wall time as DD:HH:MM.m (decimal minutes)."""
    elapsed = max(0.0, time.monotonic() - start_time)
    days = int(elapsed // 86400)
    hours = int((elapsed % 86400) // 3600)
    minutes_decimal = (elapsed % 3600) / 60.0
    return f"{days:02d}:{hours:02d}:{minutes_decimal:04.1f}"


def _build_lr_scheduler(optimizer: optim.Optimizer, args):
    """Create an epoch-level LR scheduler.

    The default "poly" schedule matches common 3D medical segmentation
    training practice (e.g., nnU-Net style polynomial decay).
    """
    schedule = args.lr_schedule.strip().lower()
    if schedule == "constant":
        return None

    if schedule == "poly":
        total_epochs = max(1, int(args.epochs))
        warmup_epochs = max(0, int(args.lr_warmup_epochs))
        warmup_start = max(0.0, min(1.0, float(args.lr_warmup_start_factor)))
        power = float(args.lr_poly_power)
        if args.lr <= 0:
            min_factor = 0.0
        else:
            min_factor = max(0.0, min(1.0, float(args.lr_min) / float(args.lr)))

        def _poly_lambda(epoch_idx: int) -> float:
            if warmup_epochs > 0 and epoch_idx < warmup_epochs:
                warmup_progress = (epoch_idx + 1) / warmup_epochs
                return warmup_start + (1.0 - warmup_start) * warmup_progress

            decay_steps = max(1, total_epochs - warmup_epochs - 1)
            progress = min(max((epoch_idx - warmup_epochs) / decay_steps, 0.0), 1.0)
            poly = (1.0 - progress) ** power
            return min_factor + (1.0 - min_factor) * poly

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_poly_lambda)

    if schedule == "cosine":
        total_epochs = max(1, int(args.epochs))
        warmup_epochs = max(0, int(args.lr_warmup_epochs))
        warmup_start = max(0.0, min(1.0, float(args.lr_warmup_start_factor)))
        if args.lr <= 0:
            min_factor = 0.0
        else:
            min_factor = max(0.0, min(1.0, float(args.lr_min) / float(args.lr)))

        def _cosine_lambda(epoch_idx: int) -> float:
            if warmup_epochs > 0 and epoch_idx < warmup_epochs:
                warmup_progress = (epoch_idx + 1) / warmup_epochs
                return warmup_start + (1.0 - warmup_start) * warmup_progress

            decay_steps = max(1, total_epochs - warmup_epochs - 1)
            progress = min(max((epoch_idx - warmup_epochs) / decay_steps, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_cosine_lambda)

    raise ValueError(f"Unknown lr schedule: {args.lr_schedule}")


def _compute_masked_loss(
    criterion: nn.Module,
    output: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute masked loss for either pointwise or 3D-structural criteria."""
    # Training uses ~mask voxels as supervised targets.
    valid_mask = (~mask).to(dtype=output.dtype)

    # If the criterion's forward accepts a 'valid_mask' kwarg, call it
    # with full-shape tensors so structural losses can operate on 5D inputs.
    try:
        sig = inspect.signature(criterion.forward)
        if "valid_mask" in sig.parameters:
            return criterion(output, target, valid_mask=valid_mask)
    except (ValueError, TypeError):
        # Fall back to positional-call inspection if signature extraction fails.
        pass

    # Otherwise assume the criterion expects flat 1D tensors of selected
    # voxels (pointwise losses like simple MSE). Use boolean indexing.
    return criterion(output[~mask], target[~mask])


def _print_keras_like_model_summary(
    model: nn.Module,
    sample_shape: tuple[int, int, int],
    device: torch.device,
    show_trainable: bool = True,
) -> None:
    """Print a safe Keras-like summary without running a model forward pass.

    This avoids hard crashes observed with torchinfo on some torch/macOS stacks.
    """
    del sample_shape  # kept for API compatibility

    model_device = device
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        pass

    print(f"Model summary (safe, no forward pass; device={model_device}):")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"  Total params:      {total_params:,}")
    if show_trainable:
        print(f"  Trainable params:  {trainable_params:,}")
        print(f"  Non-trainable:     {non_trainable_params:,}")

    # Compact per-module breakdown similar to a Keras-style layer table.
    header = f"{'Layer (type)':<54} {'Output Shape':<18} {'Param #':>12}"
    print(header)
    print("-" * len(header))

    shown = 0
    for name, module in model.named_modules():
        if name == "":
            continue
        own_params = sum(p.numel() for p in module.parameters(recurse=False))
        if own_params == 0 and len(list(module.children())) > 0:
            continue
        cls_name = module.__class__.__name__
        layer_name = f"{name} ({cls_name})"
        shape = "-"
        if hasattr(module, "out_channels"):
            shape = f"(N,{getattr(module, 'out_channels')},D,H,W)"
        elif hasattr(module, "num_features"):
            shape = f"(N,{getattr(module, 'num_features')},D,H,W)"
        print(f"{layer_name[:54]:<54} {shape:<18} {own_params:>12,}")
        shown += 1
        if shown >= 80:
            print("... (truncated)")
            break

    if torchinfo_summary is not None:
        print("  Note: torchinfo is installed, but forward-pass summary is disabled for stability.")


def _print_keras_like_model_summary_full(
    model: nn.Module,
    sample_shape: tuple[int, int, int],
    device: torch.device,
    show_trainable: bool = True,
) -> None:
    """Print full torchinfo summary with real forward pass (debug-only mode)."""
    print("Model summary FULL (debug mode; runs a real forward pass):")
    print("  WARNING: This can be slow and may increase memory use or fail on some environments.")

    if torchinfo_summary is None:
        print("  torchinfo is not installed; skipping full summary.")
        print("  Install with: pip install torchinfo")
        return

    # Keep the summary forward pass on CPU to avoid perturbing active accelerator state.
    target_device = torch.device("cpu")
    try:
        original_device = next(model.parameters()).device
    except StopIteration:
        original_device = device

    moved_to_cpu = original_device.type != "cpu"
    if moved_to_cpu:
        model.to(target_device)

    ckpt_modules = [m for m in model.modules() if hasattr(m, "use_checkpoint")]
    original_ckpt_flags = [getattr(m, "use_checkpoint") for m in ckpt_modules]
    for m in ckpt_modules:
        m.use_checkpoint = False

    col_names = ("input_size", "output_size", "num_params")
    if show_trainable:
        col_names = ("input_size", "output_size", "num_params", "trainable")

    input_size = (1, 1, int(sample_shape[0]), int(sample_shape[1]), int(sample_shape[2]))

    try:
        with torch.no_grad():
            torchinfo_summary(
                model,
                input_size=input_size,
                depth=8,
                col_names=col_names,
                row_settings=("depth", "var_names"),
                device=str(target_device),
                verbose=1,
            )
    except Exception as exc:
        print(f"  WARNING: full model summary failed ({exc})")
    finally:
        for m, old in zip(ckpt_modules, original_ckpt_flags):
            m.use_checkpoint = old
        if moved_to_cpu:
            model.to(original_device)


class ThermalGuard:
    def __init__(self, max_c: float, cooldown_sec: int,
                 check_every_batches: int, output_dir: Path,
                 pressure_trip_level: str = "serious"):
        self.max_c = max_c
        self.cooldown_sec = max(0, cooldown_sec)
        self.check_every_batches = max(1, check_every_batches)
        self.output_dir = output_dir
        self.pressure_trip_level = (pressure_trip_level or "off").strip().lower()
        self._pressure_order = {
            "nominal": 0,
            "fair": 1,
            "serious": 2,
            "critical": 3,
        }
        self._pressure_trip_idx = (
            None if self.pressure_trip_level == "off"
            else self._pressure_order[self.pressure_trip_level]
        )
        self.last_temp_c = None
        self.last_pressure_level = None

    def sample_temperature(self, batch_idx: int):
        """Sample CPU temperature at the configured periodic interval."""
        if self.max_c <= 0 and self._pressure_trip_idx is None:
            return None
        if batch_idx % self.check_every_batches != 0:
            return self.last_temp_c
        self.last_temp_c = get_cpu_temperature_c()
        self.last_pressure_level = get_thermal_pressure_level()
        return self.last_temp_c

    def maybe_pause(self, epoch: int, ds_idx: int, batch_idx: int,
                    model, optimizer, scaler,
                    train_paths: list, val_paths: list,
                    temp_c: float | None = None,
                    ema_state: dict | None = None) -> bool:
        """Checkpoint and pause training when CPU temperature is too high."""
        if self.max_c <= 0 and self._pressure_trip_idx is None:
            return False
        # Use cached last_temp_c/last_pressure_level set by sample_temperature()
        # rather than re-invoking it (avoids duplicate subprocess calls).
        if temp_c is None:
            temp_c = self.last_temp_c

        pressure_trip = False
        if self._pressure_trip_idx is not None and self.last_pressure_level is not None:
            pressure_idx = self._pressure_order.get(self.last_pressure_level.strip().lower())
            pressure_trip = pressure_idx is not None and pressure_idx >= self._pressure_trip_idx

        if temp_c is not None and temp_c >= self.max_c:
            trip_reason = f"CPU {temp_c:.1f}C >= {self.max_c:.1f}C"
        elif pressure_trip:
            trip_reason = f"thermal pressure {self.last_pressure_level}"
        else:
            return False

        ckpt_path = self.output_dir / "thermal_latest.pt"
        print(
            f"\nThermal pause: {trip_reason} "
            f"(epoch {epoch + 1}, dataset {ds_idx + 1}, batch {batch_idx})"
        )
        _save_checkpoint(
            ckpt_path,
            model,
            optimizer,
            scaler,
            epoch,
            train_loss=float("nan"),
            val_loss=float("nan"),
            train_paths=train_paths,
            val_paths=val_paths,
            ds_idx=ds_idx,
            ema_state=ema_state,
        )
        print(f"  Saved thermal checkpoint: {ckpt_path}")
        if self.cooldown_sec > 0:
            print(f"  Cooling down for {self.cooldown_sec} seconds...")
            time.sleep(self.cooldown_sec)
            print("  Resuming training after cooldown.")
        return True


class ModelEMA:
    """Exponential moving average of model weights."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        self.backup = None

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, tensor in model.state_dict().items():
                shadow_tensor = self.shadow[name]
                if torch.is_floating_point(shadow_tensor):
                    shadow_tensor.mul_(self.decay).add_(tensor.detach(), alpha=1.0 - self.decay)
                else:
                    shadow_tensor.copy_(tensor)

    def store(self, model: nn.Module) -> None:
        self.backup = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: nn.Module) -> None:
        if self.backup is None:
            return
        model.load_state_dict(self.backup, strict=True)
        self.backup = None

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state.get("decay", self.decay))
        shadow = state.get("shadow", {})
        for name, tensor in self.shadow.items():
            if name in shadow:
                self.shadow[name].copy_(shadow[name].to(device=tensor.device, dtype=tensor.dtype))


def _print_thermal_monitor_status(max_c: float, pressure_trip_level: str) -> None:
    """Print whether CPU thermal monitoring is available for this run."""
    pressure_trip_level = (pressure_trip_level or "off").strip().lower()
    if max_c <= 0 and pressure_trip_level == "off":
        print("Thermal monitor: disabled")
        return

    temp_c = get_cpu_temperature_c()
    pressure = get_thermal_pressure_level()
    if temp_c is None and pressure is None:
        print("Thermal monitor: unavailable (powermetrics output could not be parsed)")
        print("  Hint: run 'sudo -v' before starting training to enable automatic thermal pausing.")
        return

    if pressure_trip_level == "off":
        pressure_msg = "off"
    else:
        pressure_msg = pressure_trip_level.capitalize()

    if temp_c is not None:
        print(f"Thermal monitor: available (current CPU {temp_c:.1f}C, threshold {max_c:.1f}C)")
        if pressure is not None:
            print(f"  Thermal pressure: {pressure}")
        print(f"  Pressure trip level: {pressure_msg}")
    else:
        print(f"Thermal monitor: available via thermal pressure only ({pressure})")
        if pressure_trip_level == "off":
            print("  Pressure-based pausing is disabled; only CPU temperature can trigger a pause.")
        else:
            print(f"  Pause trigger uses pressure levels >= {pressure_msg} when CPU temperature is unavailable.")


# ---------------------------------------------------------------------------
# Dynamic dataset helpers
# ---------------------------------------------------------------------------

def _discover_zarr_paths(data_folder: str, dataset_glob: str) -> list:
    """Return zarr paths matching dataset_glob under data_folder, sorted oldest-first
    by the parent dataset folder mtime (consistent with generate_datasets.sh ls -1dtr).

    Datasets whose companion temp_folder__ sibling exists are in-progress and excluded.
    """
    paths = list(Path(data_folder).glob(dataset_glob))
    complete = []
    for p in paths:
        ds_folder = p.parent
        temp_companion = ds_folder.parent / ds_folder.name.replace("seismic__", "temp_folder__", 1)
        if temp_companion.exists():
            continue
        complete.append(p)
    complete.sort(key=lambda p: p.parent.stat().st_mtime)
    return [str(p) for p in complete]


def _prune_oldest_to_target(
    data_folder: str,
    dataset_glob: str,
    discovered: list,
    keep_total: int,
) -> list:
    """Prune oldest complete datasets on disk so only newest keep_total remain.

    Pruning runs only at epoch boundaries. In-progress datasets are excluded by
    _discover_zarr_paths and therefore never deleted here.
    """
    keep_total = int(keep_total)
    if keep_total < 2:
        print(
            f"Epoch prune: safety guard engaged (target keep_total={keep_total} < 2); "
            "skipping pruning."
        )
        return discovered

    if len(discovered) <= keep_total:
        return discovered

    n_delete = len(discovered) - keep_total
    delete_candidates = discovered[:n_delete]  # oldest-first input
    removed = []

    print(
        f"Epoch prune: {len(discovered)} complete dataset(s) on disk; "
        f"keeping newest {keep_total}, deleting oldest {n_delete}."
    )
    for p in delete_candidates:
        ds_dir = Path(p).parent
        if not ds_dir.name.startswith("seismic__"):
            print(f"  WARNING: refusing to delete unexpected folder: {ds_dir}")
            continue
        try:
            shutil.rmtree(ds_dir)
            removed.append(ds_dir.name)
        except Exception as exc:
            print(f"  WARNING: failed to delete {ds_dir.name}: {exc}")

    if removed:
        print(f"  Removed oldest dataset(s): {removed}")

    # Re-scan disk to get a fresh oldest-first list after deletions.
    return _discover_zarr_paths(data_folder, dataset_glob)


def _update_split(discovered: list, train_paths: list, val_paths: list,
                  num_train: int, num_val: int) -> tuple:
    """Maintain train/val assignment with permanent side exclusivity.

    train_paths / val_paths are CUMULATIVE historical lists — never shrunk.
    A path once assigned to one side stays there permanently (even after
    deletion from disk), so it can never migrate to the other side.

    Active deficit = target count minus the number of historical assignments
    still on disk.  Newly-discovered paths fill deficits (val first, then
    train).  Returns extended lists (appended-only).

    Callers compute the active window (newest num_train/num_val on disk) via
    _active_paths and pass that slice to _build_loaders.
    """
    discovered_set = set(discovered)

    # On-disk subsets — used for deficit counting only, NOT for exclusivity.
    active_train = [p for p in train_paths if p in discovered_set]
    active_val   = [p for p in val_paths   if p in discovered_set]

    # Exclusivity guard: full historical sets prevent any path from crossing
    # sides even if it was deleted and then re-discovered.
    known = set(train_paths) | set(val_paths)

    # Fill deficits from newly-discovered paths (val first, then train)
    new_paths = [p for p in discovered if p not in known]
    added_train, added_val = [], []
    for p in new_paths:
        val_need   = num_val   - len(active_val)   - len(added_val)
        train_need = num_train - len(active_train) - len(added_train)
        if val_need > 0:
            added_val.append(p)
        elif train_need > 0:
            added_train.append(p)
        else:
            break  # at capacity

    if added_train or added_val:
        print(f"Epoch split: {len(added_train) + len(added_val)} new dataset(s) assigned, "
              f"{len(added_train)} to train, {len(added_val)} to val:")
        for p in added_train:
            print(f"  train: {Path(p).parent.name}")
        for p in added_val:
            print(f"    val: {Path(p).parent.name}")
    else:
        n_t = min(len(active_train), num_train)
        n_v = min(len(active_val),   num_val)
        missing = max(0, num_train - len(active_train)) + max(0, num_val - len(active_val))
        if missing:
            print(f"Epoch split: {missing} slot(s) below target "
                  f"({n_t}/{num_train} train, {n_v}/{num_val} val) — waiting for new datasets.")
        else:
            print(f"Epoch split: no changes ({n_t} train, {n_v} val active).")

    # Return full historical lists — never shrunk, ensures permanent exclusivity.
    return train_paths + added_train, val_paths + added_val


def _active_paths(historical: list, n: int, discovered_set: set) -> list:
    """Return the newest n paths from historical that are currently on disk."""
    def _mtime(p: str) -> float:
        return Path(p).parent.stat().st_mtime if Path(p).parent.exists() else 0.0
    on_disk = [p for p in historical if p in discovered_set]
    on_disk.sort(key=_mtime)
    return on_disk[-n:] if on_disk else []


def _resolve_target_counts(
    total_datasets: int,
    val_split_ratio: float,
) -> tuple[int, int]:
    """Resolve train/val target counts from validation split ratio."""
    if total_datasets <= 0:
        return 0, 0

    if total_datasets == 1:
        return 1, 0

    ratio = max(0.0, min(1.0, float(val_split_ratio)))
    n_val = int(round(total_datasets * ratio))
    n_val = max(1, min(n_val, total_datasets - 1))
    n_train = total_datasets - n_val
    return n_train, n_val


def _build_loaders(
    train_paths: list,
    val_paths: list,
    loader_kwargs: dict,
    train_batches_per_epoch: int | None = None,
    val_batches_per_epoch: int | None = None,
) -> tuple[DataLoader | None, list[tuple[str, DataLoader]]]:
    """Build one merged train DataLoader and per-dataset val DataLoaders.

    Train datasets are merged into a single ConcatDataset-backed DataLoader so
    every mini-batch draws samples uniformly from all source datasets.
    Val datasets remain separate for per-dataset loss reporting.

    Returns:
        train_loader: Single shuffled DataLoader over merged train data, or
            None if no train dataset could be opened.
        val_loaders: List of (name, DataLoader) pairs for per-dataset val.
    """
    def _mtime(p: str) -> float:
        parent = Path(p).parent
        return parent.stat().st_mtime if parent.exists() else 0.0

    # --- train: build per-dataset loaders then merge ---
    train_per_ds: list[tuple[str, DataLoader]] = []
    print("  Loading train datasets...")
    for path in sorted(train_paths, key=_mtime):
        name = Path(path).parent.name
        try:
            loader = create_dataloader(path, augment=True, **loader_kwargs)
            print(f"    {name}: {len(loader.dataset)} samples, {len(loader)} batches")
            train_per_ds.append((name, loader))
        except Exception as e:
            print(f"    WARNING: skipping {name} (train) — {e}")

    if train_per_ds:
        merged_dataset = ConcatDataset([ldr.dataset for _, ldr in train_per_ds])
        base = train_per_ds[0][1]
        train_loader: DataLoader | None = DataLoader(
            merged_dataset,
            batch_size=int(base.batch_size) if base.batch_size is not None else 1,
            shuffle=True,
            num_workers=base.num_workers,
            pin_memory=base.pin_memory,
        )
    else:
        train_loader = None

    # --- val: keep per-dataset ---
    val_loaders: list[tuple[str, DataLoader]] = []
    if val_paths:
        print("  Loading val datasets...")
        for path in sorted(val_paths, key=_mtime):
            name = Path(path).parent.name
            try:
                loader = create_dataloader(path, augment=False, **loader_kwargs)
                print(f"    {name}: {len(loader.dataset)} samples, {len(loader)} batches")
                val_loaders.append((name, loader))
            except Exception as e:
                print(f"    WARNING: skipping {name} (val) — {e}")

    def _safe_loader_len(loader: DataLoader | None) -> int:
        if loader is None:
            return 0
        try:
            return len(loader)
        except Exception as e:
            print(f"  WARNING: loader length unavailable; treating as 0 batches — {e}")
            return 0

    natural_train = _safe_loader_len(train_loader)
    natural_val = sum(_safe_loader_len(l) for _, l in val_loaders)
    shown_train = natural_train if train_batches_per_epoch is None else train_batches_per_epoch
    shown_val = natural_val if val_batches_per_epoch is None else val_batches_per_epoch
    if shown_train == natural_train and shown_val == natural_val:
        print(f"  Batches this epoch: {shown_train} train, {shown_val} val")
    else:
        print(
            f"  Batches this epoch: {shown_train} train, {shown_val} val "
            f"(natural loader sizes: {natural_train} train, {natural_val} val)"
        )
    return train_loader, val_loaders


def _log_per_dataset_figures(
    model: nn.Module,
    merged_loader: DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    epoch_loss: float,
) -> None:
    """Log one 4-panel cross-section figure per source dataset to TensorBoard.

    Runs a single index-0 inference sample per sub-dataset in eval mode.
    Called once at the end of each training epoch; cost is negligible relative
    to the epoch itself.
    """
    def _get_live_example(requested_ds, all_datasets):
        candidates = [requested_ds] + [ds for ds in all_datasets if ds is not requested_ds]
        for candidate in candidates:
            try:
                inp, tgt, mask = candidate[0]
                return candidate, inp, tgt, mask
            except RuntimeError as exc:
                if "All array keys unavailable in zarr store" not in str(exc):
                    raise
        return None

    if not isinstance(merged_loader.dataset, ConcatDataset):
        import warnings
        warnings.warn(
            "_log_per_dataset_figures: merged_loader.dataset is not a ConcatDataset; "
            "skipping per-dataset figures.",
            stacklevel=2,
        )
        return

    model.eval()
    try:
        with torch.no_grad():
            import warnings
            all_datasets = list(merged_loader.dataset.datasets)
            for ds in all_datasets:
                ds_name = Path(ds.data_path).parent.name
                sample = _get_live_example(ds, all_datasets)
                if sample is None:
                    warnings.warn(
                        "_log_per_dataset_figures: no live zarr datasets remained at epoch end; "
                        "skipping remaining per-dataset figures.",
                        stacklevel=2,
                    )
                    break
                sample_ds, inp, tgt, _ = sample
                inp_t = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).float().to(device)
                out_t = model(inp_t)
                tgt_t = torch.from_numpy(tgt).unsqueeze(0)
                sample_ds_name = Path(sample_ds.data_path).parent.name
                title = (
                    f"{ds_name}  |  epoch {epoch + 1}  |  loss {epoch_loss:.4f}"
                )
                if sample_ds is not ds:
                    title = f"{title}  |  example from {sample_ds_name}"
                fig = make_4panel_figure(
                    inp_t[0].cpu(), out_t[0].cpu(), tgt_t.cpu(), title
                )
                writer.add_figure(f"train/{ds_name}", fig, global_step=epoch + 1)
                plt.close(fig)
    finally:
        model.train()


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler=None,
    writer: SummaryWriter = None,
    epoch: int = 0,
    output_dir: Path = None,
    train_paths: list = None,
    val_paths: list = None,
    thermal_guard: ThermalGuard = None,
    grad_accum_steps: int = 1,
    grad_clip_norm: float = 0.0,
    ema: ModelEMA = None,
    ema_update_every: int = 1,
    max_batches: int | None = None,
    return_details: bool = False,
    args=None,  # TODO: remove this line to disable QC print statements
) -> float | dict:
    """
    Train for one epoch using a single merged train DataLoader.

    The loader is expected to draw samples from all source datasets through
    ConcatDataset + shuffle so each optimizer step sees mixed data.
    """
    model.train()
    total_loss = 0.0
    total_batches = 0
    accum_steps = max(1, int(grad_accum_steps))
    ema_every = max(1, int(ema_update_every))
    optimizer_steps = 0
    micro_batches = 0
    optimizer.zero_grad(set_to_none=True)

    window_start = time.monotonic()
    nz_pct_sum = 0.0
    last_input = None
    last_output = None
    last_target = None

    try:
        natural_batches = len(train_loader)
    except Exception as e:
        print(f"    WARNING: train loader length unavailable — {e}")
        natural_batches = 0

    if natural_batches == 0:
        avg_loss = float("nan")
        if return_details:
            return {
                "loss": avg_loss,
                "batches_processed": 0,
                "reload_requested": False,
            }
        return avg_loss

    target_batches = natural_batches if max_batches is None else max(1, int(max_batches))
    iter_start_t0 = time.monotonic()
    loader_iter = iter(train_loader)
    iter_elapsed_min = (time.monotonic() - iter_start_t0) / 60.0
    print(f"    Train iterator/sampler startup: {iter_elapsed_min:04.1f}m")
    reload_requested = False
    for batch_idx in range(target_batches):
        try:
            input_data, target, mask = next(loader_iter)
            input_data = input_data.unsqueeze(1).float().to(device, non_blocking=True)
            target = target.unsqueeze(1).float().to(device, non_blocking=True)
            mask = mask.unsqueeze(1).to(device, non_blocking=True)
        except StopIteration:
            loader_iter = iter(train_loader)
            try:
                input_data, target, mask = next(loader_iter)
                input_data = input_data.unsqueeze(1).float().to(device, non_blocking=True)
                target = target.unsqueeze(1).float().to(device, non_blocking=True)
                mask = mask.unsqueeze(1).to(device, non_blocking=True)
            except Exception as e:
                print(f"    WARNING: loader exhausted/unavailable at batch {batch_idx} — {e}")
                reload_requested = True
                break
        except Exception as e:
            print(f"    WARNING: skipping batch {batch_idx} — {e}")
            reload_requested = True
            break

        try:
            with autocast_context(device):
                output = model(input_data)

            # Compute losses in FP32 for numeric stability on MPS autocast.
            # SSIM has multiple reductions/divisions and is more sensitive than MSE.
            loss = _compute_masked_loss(
                criterion,
                output.float(),
                target.float(),
                mask,
            )
            if not torch.isfinite(loss):
                print(f"    WARNING: non-finite loss at train batch {batch_idx}; skipping this batch.")
                optimizer.zero_grad(set_to_none=True)
                micro_batches = 0
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
                continue
            batch_loss = loss.item()
            scaled_loss = loss / accum_steps

            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            micro_batches += 1
            do_step = (micro_batches >= accum_steps) or (batch_idx == target_batches - 1)
            if do_step:
                if scaler is not None:
                    if grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if grad_clip_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                micro_batches = 0
                optimizer_steps += 1
                if ema is not None and optimizer_steps % ema_every == 0:
                    ema.update(model)
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg:
                print(f"    WARNING: OOM at train batch {batch_idx}; clearing cache and skipping this batch.")
                optimizer.zero_grad(set_to_none=True)
                micro_batches = 0
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
                continue
            raise

        ### TODO: remove block ----------- qc for batch -------------- start
        if args is not None:  # TODO: remove this condition and the block to disable QC print statements
            # QC block: report the already-scaled study losses so runtime logs make
            # the effective weighting explicit (SSIM is shown after the 200x scale).
            huber_criterion = nn.HuberLoss(delta=args.huber_delta, reduction="mean")  # TODO: remove line
            huber_loss = _compute_masked_loss(  # TODO: remove line
                huber_criterion,  # TODO: remove line
                output.float(),  # TODO: remove line
                target.float(),  # TODO: remove line
                mask,  # TODO: remove line
            )  # TODO: remove line
            ssim_criterion = SSIMMSELoss3D(  # TODO: remove line
                    data_range=args.ssim_data_range,  # TODO: remove line
                    window_size=args.ssim_window_size,  # TODO: remove line
                    sigma=args.ssim_sigma,  # TODO: remove line
                    alpha=args.ssim_alpha,  # TODO: remove line
                    min_valid_ratio=args.ssim_min_valid_ratio,  # TODO: remove line
                ).to(device)  # TODO: remove line
            ssim_loss = _compute_masked_loss(  # TODO: remove line - compute missing ssim_loss
                ssim_criterion,  # TODO: remove line
                output.float(),  # TODO: remove line
                target.float(),  # TODO: remove line
                mask,  # TODO: remove line
            ) * args.ssim_alpha  # TODO: remove line - apply scaling factor
            mse_criterion = nn.MSELoss()  # TODO: remove line
            mse_loss = _compute_masked_loss(  # TODO: remove line
                mse_criterion,  # TODO: remove line
                output.float(),  # TODO: remove line
                target.float(),  # TODO: remove line
                mask,  # TODO: remove line
            )  # TODO: remove line
            print(  # TODO: remove line
                f"         .. Batch {batch_idx + 1}/{target_batches}: \n"  # TODO: remove line
                f"          . mse={mse_loss.item():.6f}, huber={huber_loss.item():.6f}, ssim(x200)={ssim_loss.item():.6f}\n"  # TODO: remove line
                f"          . input  min/mean/max/std: {input_data.min().item():.3f}/{input_data.mean().item():.3f}/{input_data.max().item():.3f}/{input_data.std().item():.3f}\n"  # TODO: remove line
                f"          . output min/mean/max/std: {output.min().item():.3f}/{output.mean().item():.3f}/{output.max().item():.3f}/{output.std().item():.3f}\n"  # TODO: remove line
                f"          . target min/mean/max/std: {target.min().item():.3f}/{target.mean().item():.3f}/{target.max().item():.3f}/{target.std().item():.3f}\n"  # TODO: remove line
            )  # TODO: remove line
        ### TODO: remove block ----------- qc for batch -------------- end

        temp_c = None
        if thermal_guard is not None:
            temp_c = thermal_guard.sample_temperature(batch_idx)

        if thermal_guard is not None:
            thermal_guard.maybe_pause(
                epoch=epoch,
                ds_idx=-1,
                batch_idx=batch_idx,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                train_paths=train_paths,
                val_paths=val_paths,
                temp_c=temp_c,
                ema_state=ema.state_dict() if ema is not None else None,
            )

        total_loss += batch_loss
        total_batches += 1

        with torch.no_grad():
            x_nz = (input_data != 0).sum().item()
            y_nz = (target != 0).sum().item()
            batch_pct = (x_nz / y_nz * 100.0) if y_nz > 0 else 0.0
        nz_pct_sum += batch_pct

        # Keep last batch tensors for end-of-epoch diagnostic plotting.
        last_input = input_data[0].detach().cpu()
        last_output = output[0].detach().cpu()
        last_target = target[0].detach().cpu()

        if (batch_idx + 1) % 10 == 0:
            avg_pct = nz_pct_sum / max(total_batches, 1)
            elapsed_dhm = _format_elapsed_dhm(window_start)
            window_start = time.monotonic()
            temp_str = ""
            if thermal_guard is not None and thermal_guard.last_temp_c is not None:
                temp_str = f", CPU temp: {thermal_guard.last_temp_c:.1f}C"
            elif thermal_guard is not None and thermal_guard.last_pressure_level is not None:
                temp_str = f", Thermal pressure: {thermal_guard.last_pressure_level}"
            print(
                f"    Train batch {batch_idx}/{target_batches}, Elapsed DHM: {elapsed_dhm}, "
                f"Loss: {batch_loss:.4f}, Augmentation non-zero percentage: {avg_pct:.1f}%{temp_str}"
            )
            if output_dir is not None:
                _save_checkpoint(
                    output_dir / "partial_latest.pt",
                    model, optimizer, scaler, epoch,
                    train_loss=total_loss / max(total_batches, 1), val_loss=float('nan'),
                    train_paths=train_paths, val_paths=val_paths,
                    ds_idx=-1,
                    ema_state=ema.state_dict() if ema is not None else None,
                )

    if writer is not None and last_input is not None:
        avg_epoch_loss = total_loss / max(total_batches, 1)
        title = f"merged-train  |  epoch {epoch + 1}  |  loss {avg_epoch_loss:.4f}"
        fig = make_4panel_figure(last_input, last_output, last_target, title)
        writer.add_figure("train/merged", fig, global_step=epoch + 1)
        plt.close(fig)

    avg_loss = total_loss / max(total_batches, 1)
    if return_details:
        return {
            "loss": avg_loss,
            "batches_processed": total_batches,
            "reload_requested": reload_requested,
        }
    return avg_loss


def validate(
    model: nn.Module,
    val_loaders: list,
    criterion: nn.Module,
    device: torch.device,
    writer: SummaryWriter = None,
    epoch: int = 0,
    thermal_guard: ThermalGuard = None,
    max_batches: int | None = None,
) -> float:
    """
    Validate the model across all validation datasets.

    At the end of each validation dataset, logs 4 separate cross-section figures
    to TensorBoard (input & output × center-X & center-Y). In the TensorBoard UI,
    select tag prefixes to toggle between input/output for each slice direction.
    """
    if not val_loaders:
        return float('nan')

    model.eval()
    total_loss = 0.0
    total_batches = 0
    val_start = time.monotonic()
    window_start = val_start
    remaining_batches = None if max_batches is None else max(1, int(max_batches))

    if remaining_batches is None:
        per_loader_targets = [None] * len(val_loaders)
    else:
        n_loaders = max(1, len(val_loaders))
        base = remaining_batches // n_loaders
        remainder = remaining_batches % n_loaders
        per_loader_targets = [
            base + (1 if idx < remainder else 0)
            for idx in range(len(val_loaders))
        ]

    with torch.no_grad():
        for ds_idx, (ds_name, loader) in enumerate(val_loaders):
            target_for_loader = per_loader_targets[ds_idx]
            if target_for_loader is not None and target_for_loader <= 0:
                continue

            try:
                loader_len = len(loader)
            except Exception as e:
                print(f"\n  WARNING: skipping val dataset {ds_name} — loader unavailable ({e})")
                continue

            target_ds_batches = loader_len if target_for_loader is None else min(loader_len, target_for_loader)
            if target_ds_batches <= 0:
                print(f"\n  WARNING: skipping val dataset {ds_name} — 0 available batches")
                continue
            keys_str = ", ".join(loader.dataset.available_cubes)
            print(f"\n  Val dataset {ds_name}, {keys_str} [{ds_idx + 1}/{len(val_loaders)}]")
            first_input = None
            first_output = None
            first_target = None
            ds_loss = 0.0
            ds_batches = 0
            ds_nonzero_pct_sum = 0.0

            try:
                loader_iter = iter(loader)
                for batch_idx in range(target_ds_batches):
                    try:
                        input_data, target, mask = next(loader_iter)
                    except StopIteration:
                        break
                    input_data = input_data.unsqueeze(1).float().to(device, non_blocking=True)
                    target = target.unsqueeze(1).float().to(device, non_blocking=True)
                    mask = mask.unsqueeze(1).to(device, non_blocking=True)

                    with autocast_context(device):
                        output = model(input_data)

                    loss = _compute_masked_loss(
                        criterion,
                        output.float(),
                        target.float(),
                        mask,
                    )
                    if not torch.isfinite(loss):
                        print(f"    WARNING: non-finite val loss in {ds_name} batch {batch_idx}; skipping batch.")
                        continue

                    if thermal_guard is not None:
                        thermal_guard.sample_temperature(batch_idx)

                    batch_loss = loss.item()
                    ds_loss += batch_loss
                    ds_batches += 1
                    total_loss += batch_loss
                    total_batches += 1

                    # Track non-zero ratio
                    x_nz = (input_data != 0).sum().item()
                    y_nz = (target != 0).sum().item()
                    batch_pct = (x_nz / y_nz * 100.0) if y_nz > 0 else 0.0
                    ds_nonzero_pct_sum += batch_pct

                    # Capture first batch only for plotting
                    if first_input is None:
                        first_input = input_data[0].detach().cpu()
                        first_output = output[0].detach().cpu()
                        first_target = target[0].detach().cpu()

                    if (batch_idx + 1) % 10 == 0:
                        avg_pct = ds_nonzero_pct_sum / ds_batches
                        elapsed_dhm = _format_elapsed_dhm(window_start)
                        window_start = time.monotonic()
                        temp_str = ""
                        if thermal_guard is not None and thermal_guard.last_temp_c is not None:
                            temp_str = f", CPU temp: {thermal_guard.last_temp_c:.1f}C"
                        elif thermal_guard is not None and thermal_guard.last_pressure_level is not None:
                            temp_str = f", Thermal pressure: {thermal_guard.last_pressure_level}"
                        print(
                            f"    Val batch {batch_idx}/{target_ds_batches}, Elapsed DHM: {elapsed_dhm}, "
                            f"Loss: {batch_loss:.4f}, Augmentation non-zero percentage: {avg_pct:.1f}%{temp_str}"
                        )
            except Exception as e:
                print(f"    WARNING: val dataset {ds_name} failed mid-epoch — {e}")

            # --- Per-val-dataset: 4 separate TensorBoard images ---
            # Tags are structured so TensorBoard shows paired input/output
            # under the same group for each slice direction.
            if writer is not None and first_input is not None:
                avg_ds_loss = ds_loss / max(ds_batches, 1)
                title_base = (
                    f"{ds_name}  |  epoch {epoch + 1}  |  val loss {avg_ds_loss:.4f}"
                )
                for axis in ("x", "y"):
                    for kind, vol in (("input", first_input), ("output", first_output), ("label", first_target)):
                        title = f"{title_base}  |  center-{axis.upper()}  |  {kind}"
                        fig = make_crosssection_figure(vol, title, axis=axis)
                        # Tag path: val_centerX/input/dataset_name
                        #           val_centerX/output/dataset_name
                        # TensorBoard groups these under val_centerX so you
                        # can click between input and output for that direction.
                        tag = f"val_center{axis.upper()}/{kind}/{ds_name}"
                        writer.add_figure(tag, fig, global_step=epoch + 1)
                        plt.close(fig)

    return total_loss / max(total_batches, 1)


DEFAULT_ARRAY_KEYS = [
    "seismicCubes_cumsum__17_degrees",
    # "seismicCubes_cumsum__17_degrees_normalized",
    "seismicCubes_cumsum__29_degrees",
    # "seismicCubes_cumsum__29_degrees_normalized",
    "seismicCubes_cumsum__5_degrees",
    # "seismicCubes_cumsum__5_degrees_normalized",
    # "seismicCubes_cumsum_17_degrees_normalized_augmented",
    # "seismicCubes_cumsum_29_degrees_normalized_augmented",
    # "seismicCubes_cumsum_5_degrees_normalized_augmented",
    "seismicCubes_cumsum_fullstack",
    # "seismicCubes_cumsum_fullstack_noise_free"
]


def main():
    parser = argparse.ArgumentParser(description="Train Seismic 3D Mamba")
    parser.add_argument("--data_paths", type=str, nargs='*', default=[],
                       help="Explicit zarr paths (optional if --data_folder provided)")
    parser.add_argument("--data_folder", type=str, default=None,
                       help="Folder scanned each epoch for zarr datasets (enables dynamic discovery)")
    parser.add_argument("--dataset_glob", type=str, default="seismic__*/model_data.zarr",
                       help="Glob pattern relative to --data_folder for zarr discovery (default: seismic__*/model_data.zarr)")
    parser.add_argument("--array_keys", type=str, nargs='+', default=DEFAULT_ARRAY_KEYS,
                       help="One or more 3D array keys inside each Zarr dataset; one is picked randomly per sample")
    parser.add_argument("--val_split_ratio", type=float, default=0.2,
                       help="Validation split ratio over discovered datasets (default: 0.2)")
    parser.add_argument("--train_batches_per_epoch", type=int, default=None,
                       help="Optional fixed number of train batches per epoch. If set, loader cycles as needed.")
    parser.add_argument("--val_batches_per_epoch", type=int, default=None,
                       help="Optional fixed number of validation batches per epoch.")
    parser.add_argument("--refresh_every_batches", type=int, default=10,
                       help="Deprecated compatibility flag; dataset discovery/pruning now happens only at epoch boundaries.")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument("--loss_type", type=str, default="huber",
                       choices=["mse", "huber", "ssim_mse", "sliding_stats"],
                       help="Loss function (default: huber)")
    parser.add_argument("--huber_delta", type=float, default=0.1,
                       help="Delta parameter for Huber loss (default: 0.1; only used when --loss_type=huber)")
    parser.add_argument("--ssim_window_size", type=int, default=16,
                       help="3D SSIM Gaussian window size (default: 16; only used when --loss_type=ssim_mse)")
    parser.add_argument("--ssim_sigma", type=float, default=(16.0 / 6.0),
                       help="Gaussian sigma for 3D SSIM window (default: 16/6; only used when --loss_type=ssim_mse)")
    parser.add_argument("--ssim_data_range", type=float, default=30.0,
                       help="Data range for SSIM stabilization constants (default: 30.0 for scaled seismic)")
    parser.add_argument("--ssim_alpha", type=float, default=(1.0 / 6.0),
                       help="Blend factor in [0,1] for mixed SSIM+MSE loss; 0=MSE, 1=SSIM (default: 1/6)")
    parser.add_argument("--ssim_min_valid_ratio", type=float, default=0.5,
                       help="Minimum local valid-mask support ratio for SSIM pooling in [0,1] (default: 0.5)")
    parser.add_argument(
        "--sliding_stats_window",
        type=int,
        nargs=3,
        default=[9, 9, 9],
        help="3D sliding-window size (z y x) for local stats loss (default: 9 9 9; only used when --loss_type=sliding_stats)",
    )
    parser.add_argument(
        "--sliding_stats_mean_weight",
        type=float,
        default=1.0,
        help="Weight for sliding-window mean MAE term (default: 1.0; only used when --loss_type=sliding_stats)",
    )
    parser.add_argument(
        "--sliding_stats_std_weight",
        type=float,
        default=1.0,
        help="Weight for sliding-window std-ratio term (default: 1.0; only used when --loss_type=sliding_stats)",
    )
    parser.add_argument(
        "--sliding_stats_min_weight",
        type=float,
        default=1.0,
        help="Weight for sliding-window minima MAE term (default: 1.0; only used when --loss_type=sliding_stats)",
    )
    parser.add_argument(
        "--sliding_stats_max_weight",
        type=float,
        default=1.0,
        help="Weight for sliding-window maxima MAE term (default: 1.0; only used when --loss_type=sliding_stats)",
    )
    parser.add_argument(
        "--sliding_stats_mae_weight",
        type=float,
        default=1.0,
        help="Weight for voxelwise MAE term (default: 1.0; only used when --loss_type=sliding_stats)",
    )
    parser.add_argument(
        "--sliding_stats_mse_weight",
        type=float,
        default=1.0,
        help="Weight for voxelwise MSE term (default: 1.0; only used when --loss_type=sliding_stats)",
    )
    parser.add_argument(
        "--sliding_stats_eps",
        type=float,
        default=1e-6,
        help="Epsilon for local-std stabilization in sliding stats loss (default: 1e-6)",
    )
    parser.add_argument(
        "--sliding_stats_std_ratio_clip",
        type=float,
        default=10.0,
        help="Clip magnitude for local std ratio target/pred in sliding stats loss (default: 10.0)",
    )
    parser.add_argument(
        "--sliding_stats_all_voxels",
        action="store_true",
        default=False,
        help="Apply sliding stats loss to all voxels instead of only valid-mask voxels",
    )
    parser.add_argument(
        "--enable-cluster-loss",
        action="store_true",
        default=False,
        help="Enable composite cluster-aware loss that upweights traces near masked clusters",
    )
    parser.add_argument(
        "--cluster-kernel-size",
        type=int,
        default=5,
        help="Kernel size for 2D uniform filter applied to trace mask (odd int, default: 5)",
    )
    parser.add_argument(
        "--cluster-eps",
        type=float,
        default=1e-6,
        help="Epsilon threshold for smoothed cluster mask (default: 1e-6)",
    )
    parser.add_argument(
        "--cluster-base-weight",
        type=float,
        default=1.0 / 3.0,
        help="Weight for base loss in composite (default: 1/3)",
    )
    parser.add_argument(
        "--cluster-cluster-weight",
        type=float,
        default=2.0 / 3.0,
        help="Weight for cluster loss in composite (default: 2/3)",
    )
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--lr_schedule", type=str, default="poly",
                       choices=["poly", "cosine", "constant"],
                       help="Epoch LR schedule (default: poly with warmup, common for 3D medical UNet training)")
    parser.add_argument("--lr_poly_power", type=float, default=0.9,
                       help="Polynomial decay power when --lr_schedule=poly (default: 0.9)")
    parser.add_argument("--lr_min", type=float, default=1e-6,
                       help="Minimum LR floor for poly/cosine schedules (default: 1e-6)")
    parser.add_argument("--lr_warmup_epochs", type=int, default=5,
                       help="Warmup epochs before decay schedules (default: 5)")
    parser.add_argument("--lr_warmup_start_factor", type=float, default=0.1,
                       help="Warmup start as fraction of base LR (default: 0.1)")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                       help="Gradient accumulation steps (effective batch = batch_size * this value)")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0,
                       help="Clip gradient global norm to this value; set <=0 to disable")
    parser.add_argument("--ema_decay", type=float, default=0.999,
                       help="EMA decay for model weights; set <=0 to disable")
    parser.add_argument("--ema_update_every", type=int, default=1,
                       help="Update EMA every N optimizer steps (default: 1)")
    parser.add_argument("--sample_shape", type=int, nargs=3, default=[128, 128, 128],
                       help="Sample shape (x y z)")
    # NOTE: legacy `trace_mask_ratio` removed; use --target_masked_fraction instead
    parser.add_argument("--target_masked_fraction", type=float, default=0.15,
                       help="Target final masked fraction after cluster size/probability effects (default: 0.15)")
    parser.add_argument("--cluster_shape", type=int, default=3,
                       help="Odd cluster edge size for masking neighborhoods, e.g. 3, 5, 7 (default: 3)")
    parser.add_argument("--center_selection_method", type=str, default="random_mixture",
                       choices=["random_mixture", "mitchell_best_candidate", "poisson_disc", "uniform_random"],
                       help="Cluster-center sampling method for masking (default: random_mixture)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto, cuda, mps, cpu)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--use_mamba", action="store_true",
                       help="Use U-Mamba hybrid blocks in encoder (requires CUDA + mamba_ssm; falls back to ResBlock3d on MPS/CPU)")
    parser.add_argument(
        "--pre_head_mode",
        type=str,
        default="identity",
        choices=["identity", "norm", "norm_gelu"],
        help=(
            "Pre-head block applied to decoder features before the final 1x1 conv. "
            "'identity': direct linear path (recommended for reconstruction — preserves "
            "polarity and full feature range). "
            "'norm': InstanceNorm3d only (affine=True). "
            "'norm_gelu': InstanceNorm3d + GELU — soft-rectifies negative features, "
            "suppressing the model's ability to predict negative amplitudes; "
            "do NOT use for seismic reconstruction. "
            "(default: identity)"
        ),
    )
    parser.add_argument(
        "--print_model_summary",
        action="store_true",
        default=False,
        help="Print safe static per-layer model summary at startup (no forward pass; production-safe)",
    )
    parser.add_argument(
        "--print_model_summary_full",
        action="store_true",
        default=False,
        help="DEBUG ONLY: print full torchinfo model summary via real forward pass (can be slow/unstable)",
    )
    parser.add_argument("--thermal_max_c", type=float, default=85.0,
                       help="Pause when CPU temperature exceeds this in Celsius; set <=0 to disable")
    parser.add_argument("--thermal_cooldown_sec", type=int, default=300,
                       help="Cooldown sleep duration in seconds after a thermal pause")
    parser.add_argument("--thermal_check_every_batches", type=int, default=10,
                       help="Check CPU temperature every N training batches")
    parser.add_argument("--thermal_pressure_trip_level", type=str, default="serious",
                       choices=["off", "nominal", "fair", "serious", "critical"],
                       help="Pause on thermal pressure at or above this level (default: serious). Use 'off' to disable pressure-based pausing")
    parser.add_argument("--no_monitor", action="store_true", dest="monitor_disabled",
                       help="Disable background process-tree resource monitor CSV logging (enabled by default)")
    parser.add_argument("--monitor_interval_sec", type=float, default=300.0,
                       help="Monitor sampling interval in seconds (default: 300)")
    parser.add_argument("--monitor_csv_path", type=str, default=None,
                       help="CSV output path for monitor rows (default: cpu_mem_stats_<pid>.csv)")

    args = parser.parse_args()

    if not (0.0 < args.val_split_ratio < 1.0):
        parser.error("--val_split_ratio must be between 0 and 1 (exclusive)")
    if args.train_batches_per_epoch is not None and args.train_batches_per_epoch <= 0:
        parser.error("--train_batches_per_epoch must be > 0")
    if args.val_batches_per_epoch is not None and args.val_batches_per_epoch <= 0:
        parser.error("--val_batches_per_epoch must be > 0")
    if args.refresh_every_batches < 0:
        parser.error("--refresh_every_batches must be >= 0")
    if args.monitor_interval_sec <= 0:
        parser.error("--monitor_interval_sec must be > 0")
    if not (0.0 <= args.target_masked_fraction <= 1.0):
        parser.error("--target_masked_fraction must be between 0 and 1")
    if args.cluster_shape <= 0 or args.cluster_shape % 2 == 0:
        parser.error("--cluster_shape must be a positive odd integer")
    if args.huber_delta <= 0:
        parser.error("--huber_delta must be > 0")
    if args.ssim_window_size < 3:
        parser.error("--ssim_window_size must be >= 3")
    if args.ssim_sigma <= 0:
        parser.error("--ssim_sigma must be > 0")
    if args.ssim_data_range <= 0:
        parser.error("--ssim_data_range must be > 0")
    if not (0.0 <= args.ssim_alpha <= 1.0):
        parser.error("--ssim_alpha must be between 0 and 1")
    if not (0.0 <= args.ssim_min_valid_ratio <= 1.0):
        parser.error("--ssim_min_valid_ratio must be between 0 and 1")

    if not args.data_paths and not args.data_folder:
        parser.error("At least one of --data_paths or --data_folder must be provided")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_default_device(args.device)
    print_device_summary(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if device.type == "mps" and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # --- Dataset split (done once; restored from checkpoint on resume) ---
    # Build initial path list: explicit --data_paths + discover from --data_folder
    all_paths = list(dict.fromkeys(args.data_paths))  # deduplicate preserving order
    if args.data_folder:
        discovered_at_start = _discover_zarr_paths(args.data_folder, args.dataset_glob)
        known = set(all_paths)
        all_paths = all_paths + [p for p in discovered_at_start if p not in known]

    # Check for a saved split in the resume checkpoint BEFORE shuffling
    saved_train_paths = None
    saved_val_paths   = None
    if args.resume and Path(args.resume).exists():
        _peek = torch.load(args.resume, map_location="cpu")
        saved_train_paths = _peek.get("train_paths")
        saved_val_paths   = _peek.get("val_paths")
        del _peek

    initial_num_train, initial_num_val = _resolve_target_counts(
        len(all_paths), args.val_split_ratio
    )

    if saved_train_paths is not None and saved_val_paths is not None:
        supplied = set(all_paths)

        # Drop paths that no longer exist in the supplied list; deduplicate in case
        # a previous run saved a corrupt split with duplicate entries
        kept_train = list(dict.fromkeys(p for p in saved_train_paths if p in supplied))
        kept_val   = list(dict.fromkeys(p for p in saved_val_paths   if p in supplied))
        dropped    = [p for p in (saved_train_paths + saved_val_paths) if p not in supplied]
        if dropped:
            print(f"Checkpoint split: dropped {len(dropped)} path(s) no longer supplied:")
            for p in dropped:
                print(f"  - {Path(p).parent.name}")

        # Identify new paths not present in the checkpoint split at all
        checkpoint_all = set(saved_train_paths) | set(saved_val_paths)
        new_paths = [p for p in all_paths if p not in checkpoint_all]

        if new_paths:
            # Assign new paths to fill deficits (val first when both short)
            new_train, new_val = [], []
            for p in new_paths:
                train_need = initial_num_train - len(kept_train) - len(new_train)
                val_need   = initial_num_val   - len(kept_val)   - len(new_val)
                if val_need > 0:
                    new_val.append(p)
                elif train_need > 0:
                    new_train.append(p)
                else:
                    break
            kept_train += new_train
            kept_val   += new_val
            if new_train or new_val:
                print(f"Checkpoint split: {len(new_train) + len(new_val)} new dataset(s) assigned, "
                      f"{len(new_train)} to train, {len(new_val)} to val:")
                for p in new_train:
                    print(f"  train: {Path(p).parent.name}")
                for p in new_val:
                    print(f"    val: {Path(p).parent.name}")

        train_paths = kept_train
        val_paths   = kept_val
        split_target_train = len(train_paths)
        split_target_val = len(val_paths)
        print(f"Restored split: {len(train_paths)} train, {len(val_paths)} val datasets.")
    else:
        # Use newest (num_train + num_val) datasets; all_paths is oldest-first.
        # Assign newest num_val to val, next num_train to train.
        target_total = initial_num_train + initial_num_val
        pool = all_paths[-target_total:] if len(all_paths) > target_total else all_paths
        val_paths   = pool[-initial_num_val:]  if initial_num_val > 0 else []
        train_paths = pool[:-initial_num_val]  if initial_num_val > 0 else list(pool)
        train_paths = train_paths[-initial_num_train:] if len(train_paths) > initial_num_train else train_paths
        split_target_train = initial_num_train
        split_target_val = initial_num_val

    _dset_startup = set(discovered_at_start) if args.data_folder else set(all_paths)
    _at = _active_paths(train_paths, split_target_train, _dset_startup)
    _av = _active_paths(val_paths,   split_target_val,   _dset_startup)
    print(
        f"Dataset split ({split_target_train} train, {split_target_val} val target): "
        f"{len(_at)} train, {len(_av)} val"
    )
    print(f"  Train: {[Path(p).parent.name for p in _at]}")
    if _av:
        print(f"  Val:   {[Path(p).parent.name for p in _av]}")
    print()

    # --- Model + memory diagnostic (must run before dataloaders so we can set batch size) ---
    print("Creating model...")
    if args.use_mamba and not _MAMBA_AVAILABLE:
        print("WARNING: --use_mamba requested but mamba_ssm not installed; falling back to ResBlock3d")
    model = create_model(
        use_mamba=args.use_mamba,
        input_channels=1,
        hidden_dims=(32, 64, 128, 256),
        spatial_size=tuple(args.sample_shape),
        pre_head_mode=args.pre_head_mode,
    ).to(device)
    print(f"Pre-head mode: {args.pre_head_mode}")
    if args.print_model_summary:
        _print_keras_like_model_summary(
            model,
            tuple(args.sample_shape),
            device,
            show_trainable=True,
        )
    if args.print_model_summary_full:
        _print_keras_like_model_summary_full(
            model,
            tuple(args.sample_shape),
            device,
            show_trainable=True,
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    weights_bytes  = sum(p.numel() * p.element_size() for p in model.parameters())
    grads_bytes    = weights_bytes
    adam_bytes     = 2 * weights_bytes
    fixed_bytes    = weights_bytes + grads_bytes + adam_bytes

    S = args.sample_shape
    hidden = (32, 64, 128, 256)
    def _fm(b, c, s): return b * c * s[0] * s[1] * s[2] * 4
    act_per_sample = 2 * (
        _fm(1, hidden[0], S)
        + _fm(1, hidden[1], [d//2 for d in S])
        + _fm(1, hidden[2], [d//4 for d in S])
        + _fm(1, hidden[3], [d//8 for d in S])
        + _fm(1, hidden[2], [d//4 for d in S])
        + _fm(1, hidden[1], [d//2 for d in S])
        + _fm(1, hidden[0], S)
    )
    io_per_sample  = 2 * int(np.prod(S)) * 4
    per_sample_var = act_per_sample + io_per_sample

    # Peak-overhead factor: empirically calibrated from OOM crashes on M4 24 GB.
    #   batch=7 (with grad checkpointing): MPS allocated 24.09 GiB at crash.
    #   formula raw per-sample: 1.433 GB.  Observed: 24.09/7 = 3.44 GB → ratio 2.40x.
    # Use 2.5 for a small margin above observed.
    PEAK_FACTOR = 2.5
    per_sample_peak = per_sample_var * PEAK_FACTOR

    def _total(bs): return fixed_bytes + bs * per_sample_peak

    mem_info   = get_memory_info(device)
    total_mem  = mem_info["total_bytes"]

    # MPS can exceed reported RAM via unified memory.  The actual ceiling
    # (PYTORCH_MPS_HIGH_WATERMARK_RATIO default) is ~1.17 × reported RAM.
    # Observed: 30.19 GiB limit on a 25.77 GB device → ratio 1.172.
    MPS_WATERMARK = 1.172 if device.type == "mps" else 1.0
    mps_ceiling   = total_mem * MPS_WATERMARK

    # "other allocations" (Python, CPU tensors, MPS driver bookkeeping).
    # Observed stable at ~6 GB across all OOM crashes.
    OTHER_ALLOCS  = 6 * 1024**3
    available     = mps_ceiling - OTHER_ALLOCS
    safe_limit    = available * 0.85   # 15% headroom within available MPS model budget

    # Respect the requested batch size.  Compute the max safe size only for diagnostics
    # and to clamp obviously unsafe requests.
    safe_max_bs = 1
    while _total(safe_max_bs + 1) < safe_limit:
        safe_max_bs += 1

    requested_batch_size = max(1, int(args.batch_size))
    if requested_batch_size > safe_max_bs:
        print(
            f"WARNING: requested batch size {requested_batch_size} exceeds estimated safe max "
            f"{safe_max_bs}; clamping to {safe_max_bs}."
        )
        batch_size = safe_max_bs
    else:
        batch_size = requested_batch_size

    pressure   = "OK" if _total(batch_size) < safe_limit else "PRESSURE"
    current_gb = _total(batch_size) / 1e9

    print(f"""Memory estimate (batch={batch_size}):
  Weights:              {weights_bytes/1e9:.2f} GB
  Gradients:            {grads_bytes/1e9:.2f} GB
  Adam states:          {adam_bytes/1e9:.2f} GB
  Activations+temps:    {batch_size * per_sample_peak/1e9:.2f} GB  ({batch_size} x {per_sample_peak/1e9:.2f} GB/sample, {PEAK_FACTOR}x peak factor)
  -------------------------------------------------
  Total estimated:      {current_gb:.2f} GB  [{pressure}]
  MPS ceiling:          {mps_ceiling/1e9:.2f} GB  ({MPS_WATERMARK}x reported RAM)
  Other allocations:    ~{OTHER_ALLOCS/1e9:.2f} GB  (Python + MPS driver)
  Available for model:  {available/1e9:.2f} GB  (safe limit: {safe_limit/1e9:.2f} GB)
  Headroom:             {(safe_limit - _total(batch_size))/1e9:.2f} GB
    Safe max batch size:  {safe_max_bs}
    Using batch size:     {batch_size}""", flush=True)
    print()

    # macOS multiprocessing workers crash with zarr + MPS (exit code 255).
    # Use num_workers=0 (main-process loading) on macOS; workers only on Linux.
    import platform
    _num_workers = 0 if platform.system() == "Darwin" else min(4, os.cpu_count() or 1)

    loader_kwargs = dict(
        batch_size=batch_size,
        sample_shape=tuple(args.sample_shape),
        num_workers=_num_workers,
        pin_memory=(device.type == "cuda"),
        normalize=True,
        target_std=1.0,
        target_masked_fraction=args.target_masked_fraction,
        cluster_shape=args.cluster_shape,
        center_selection_method=args.center_selection_method,
        array_keys=args.array_keys,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.loss_type == "huber":
        criterion = nn.HuberLoss(delta=args.huber_delta, reduction="mean")
    elif args.loss_type == "ssim_mse":
        criterion = SSIMMSELoss3D(
            data_range=args.ssim_data_range,
            window_size=args.ssim_window_size,
            sigma=args.ssim_sigma,
            alpha=args.ssim_alpha,
            min_valid_ratio=args.ssim_min_valid_ratio,
        ).to(device)
    elif args.loss_type == "sliding_stats":
        criterion = SlidingWindowStatsLoss3D(
            window_size=tuple(args.sliding_stats_window),
            mean_weight=args.sliding_stats_mean_weight,
            std_weight=args.sliding_stats_std_weight,
            min_weight=args.sliding_stats_min_weight,
            max_weight=args.sliding_stats_max_weight,
            mae_weight=args.sliding_stats_mae_weight,
            mse_weight=args.sliding_stats_mse_weight,
            eps=args.sliding_stats_eps,
            std_ratio_clip=args.sliding_stats_std_ratio_clip,
            apply_to_all_voxels=args.sliding_stats_all_voxels,
        ).to(device)
    else:
        criterion = nn.MSELoss()
    # Optionally wrap SSIM-MSE criterion in the composite cluster-aware loss.
    if args.enable_cluster_loss:
        if isinstance(criterion, SSIMMSELoss3D):
            criterion = CompositeClusterAwareLoss(
                base_criterion=criterion,
                kernel_size=args.cluster_kernel_size,
                eps=args.cluster_eps,
                base_weight=args.cluster_base_weight,
                cluster_weight=args.cluster_cluster_weight,
            )
            # Move wrapper to device as well.
            criterion = criterion.to(device)
        else:
            print("--enable-cluster-loss ignored: base loss is not SSIMMSELoss3D")
    # Log cluster-aware loss status so it's visible in startup stdout.
    if getattr(args, "enable_cluster_loss", False):
        if isinstance(criterion, CompositeClusterAwareLoss):
            print(
                "Cluster-aware loss: enabled",
                f"(kernel={args.cluster_kernel_size}, eps={args.cluster_eps},",
                f"base_weight={args.cluster_base_weight}, cluster_weight={args.cluster_cluster_weight})",
            )
        else:
            print("Cluster-aware loss: requested but not enabled (base loss incompatible)")
    else:
        print("Cluster-aware loss: disabled")
    scaler = create_grad_scaler(device)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    thermal_guard = ThermalGuard(
        max_c=args.thermal_max_c,
        cooldown_sec=args.thermal_cooldown_sec,
        check_every_batches=args.thermal_check_every_batches,
        output_dir=output_dir,
        pressure_trip_level=args.thermal_pressure_trip_level,
    )
    _print_thermal_monitor_status(args.thermal_max_c, args.thermal_pressure_trip_level)

    # TensorBoard writer — view with: tensorboard --logdir checkpoints/runs
    tb_log_dir = output_dir / "runs"
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    print(f"TensorBoard logs: {tb_log_dir}")
    print("  Launch viewer: tensorboard --logdir checkpoints/runs")

    monitor = None
    if not args.monitor_disabled:
        monitor_csv_path = args.monitor_csv_path
        if not monitor_csv_path:
            monitor_csv_path = f"cpu_mem_stats_{os.getpid()}.csv"
        monitor = ProcessTreeCsvMonitor(
            root_pid=os.getpid(),
            csv_path=monitor_csv_path,
            interval_sec=float(args.monitor_interval_sec),
            include_children=True,
            device=device,
        )
        monitor.start()
        print(
            f"Background monitor: enabled (interval={float(args.monitor_interval_sec):.1f}s, "
            f"csv={monitor_csv_path}, root_pid={os.getpid()}, include_children=true)"
        )

    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        if ema is not None and checkpoint.get("ema_state") is not None:
            ema.load_state_dict(checkpoint["ema_state"])
        start_epoch = checkpoint["epoch"] + 1
        ds_idx_done = checkpoint.get("ds_idx", -1)
        if ds_idx_done >= 0:
            print(f"  Partial epoch {start_epoch}: completed datasets 0..{ds_idx_done}")
            print("  Note: epoch restarts from the beginning (datasets are randomly ordered)")
        print(f"  Continuing from epoch {start_epoch + 1}")

    scheduler = _build_lr_scheduler(optimizer, args)
    if scheduler is not None and start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()
    if scheduler is None:
        print(f"LR schedule: constant (lr={optimizer.param_groups[0]['lr']:.3e})")
    else:
        print(
            f"LR schedule: {args.lr_schedule} "
            f"(start={args.lr:.3e}, min={args.lr_min:.3e}, warmup={args.lr_warmup_epochs} epochs)"
        )
    if args.loss_type == "huber":
        print(f"Loss: Huber (delta={args.huber_delta})")
    elif args.loss_type == "ssim_mse":
        print(
            "Loss: SSIM-MSE "
            f"(window={args.ssim_window_size}, sigma={args.ssim_sigma:.4g}, "
            f"range={args.ssim_data_range:.4g}, alpha={args.ssim_alpha:.4g}, "
            f"min_valid_ratio={args.ssim_min_valid_ratio:.3g})"
        )
    elif args.loss_type == "sliding_stats":
        print(
            "Loss: SlidingStats "
            f"(window={tuple(args.sliding_stats_window)}, "
            f"mean_weight={args.sliding_stats_mean_weight:.4g}, "
            f"std_weight={args.sliding_stats_std_weight:.4g}, "
            f"min_weight={args.sliding_stats_min_weight:.4g}, "
            f"max_weight={args.sliding_stats_max_weight:.4g}, "
            f"mae_weight={args.sliding_stats_mae_weight:.4g}, "
            f"mse_weight={args.sliding_stats_mse_weight:.4g}, "
            f"eps={args.sliding_stats_eps:.3g}, "
            f"std_ratio_clip={args.sliding_stats_std_ratio_clip:.4g}, "
            f"all_voxels={bool(args.sliding_stats_all_voxels)})"
        )
    else:
        print(f"Loss: MSE")
    print(f"Loss module: {criterion.__class__.__name__}")
    print(
        "Masking: "
        f"target_masked_fraction={args.target_masked_fraction:.4g}, "
        f"cluster_shape={args.cluster_shape}, "
        f"cluster_prob=0.8, "
        f"center_selection_method={args.center_selection_method}"
    )
    print(f"Grad accumulation: {max(1, args.grad_accum_steps)} step(s)")
    if args.train_batches_per_epoch is not None:
        print(
            f"Train epoch length: fixed {args.train_batches_per_epoch} batches "
            "(dataset list is fixed within each epoch; refreshed at epoch start)"
        )
    else:
        print("Train epoch length: all batches from merged train loader")
    if args.val_batches_per_epoch is not None:
        print(f"Validation epoch length: fixed {args.val_batches_per_epoch} batches")
    else:
        print("Validation epoch length: all batches from val loaders")
    if args.grad_clip_norm > 0:
        print(f"Grad clipping: enabled (max norm {args.grad_clip_norm:.2f})")
    else:
        print("Grad clipping: disabled")
    if ema is not None:
        print(f"EMA: enabled (decay={args.ema_decay}, update every {max(1, args.ema_update_every)} step(s))")
    else:
        print("EMA: disabled")

    try:
        print("\nStarting training...")
        training_start = time.monotonic()
        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.monotonic()
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_stamp = datetime.now().strftime("%Y-%-m-%-d %H:%M:%S")
            print(f"\nEpoch {epoch + 1}/{args.epochs} | LR: {current_lr:.3e} | {epoch_stamp}")

            # Re-scan once per epoch, prune oldest on disk to fixed target count,
            # then keep the active train/val set fixed until next epoch.
            if args.data_folder:
                discovered = _discover_zarr_paths(args.data_folder, args.dataset_glob)
                keep_total = split_target_train + split_target_val
                discovered = _prune_oldest_to_target(
                    args.data_folder,
                    args.dataset_glob,
                    discovered,
                    keep_total,
                )
                train_paths, val_paths = _update_split(
                    discovered, train_paths, val_paths, split_target_train, split_target_val
                )
                _dset = set(discovered)
            else:
                _dset = {p for p in (train_paths + val_paths) if Path(p).parent.exists()}

            active_train = _active_paths(train_paths, split_target_train, _dset)
            active_val   = _active_paths(val_paths,   split_target_val,   _dset)
            print(f"Dataset split ({split_target_train} train, {split_target_val} val target): "
                  f"{len(active_train)} train, {len(active_val)} val")
            print(f"  Train: {[Path(p).parent.name for p in active_train]}")
            if active_val:
                print(f"  Val:   {[Path(p).parent.name for p in active_val]}")
            train_loader = None
            val_loaders = []

            if args.train_batches_per_epoch is None:
                train_loader, val_loaders = _build_loaders(
                    active_train,
                    active_val,
                    loader_kwargs,
                    train_batches_per_epoch=args.train_batches_per_epoch,
                    val_batches_per_epoch=args.val_batches_per_epoch,
                )
                if train_loader is None:
                    print("  WARNING: No usable training datasets this epoch; skipping.")
                    continue

                train_loss = train_epoch(
                    model, train_loader, optimizer, criterion, device,
                    scaler=scaler, writer=writer, epoch=epoch, output_dir=output_dir,
                    train_paths=train_paths, val_paths=val_paths,
                    thermal_guard=thermal_guard,
                    grad_accum_steps=args.grad_accum_steps,
                    grad_clip_norm=args.grad_clip_norm,
                    ema=ema,
                    ema_update_every=args.ema_update_every,
                    args=args,  # TODO: remove to disable QC print statements
                )
            else:
                target_batches = max(1, int(args.train_batches_per_epoch))
                batches_done = 0
                weighted_loss_sum = 0.0
                pending_chunk_reload = False

                while batches_done < target_batches:
                    _reload_t0 = time.monotonic()
                    train_loader, val_loaders = _build_loaders(
                        active_train,
                        active_val,
                        loader_kwargs,
                        train_batches_per_epoch=args.train_batches_per_epoch,
                        val_batches_per_epoch=args.val_batches_per_epoch,
                    )
                    _reload_elapsed = time.monotonic() - _reload_t0
                    if pending_chunk_reload:
                        print(f"  Reloaded train/val loaders in {_reload_elapsed:.2f}s")
                        pending_chunk_reload = False
                    if train_loader is None:
                        print("  WARNING: No usable training datasets this epoch; skipping remaining batches.")
                        break

                    remaining = target_batches - batches_done

                    details = train_epoch(
                        model, train_loader, optimizer, criterion, device,
                        scaler=scaler, writer=writer, epoch=epoch, output_dir=output_dir,
                        train_paths=train_paths, val_paths=val_paths,
                        thermal_guard=thermal_guard,
                        grad_accum_steps=args.grad_accum_steps,
                        grad_clip_norm=args.grad_clip_norm,
                        ema=ema,
                        ema_update_every=args.ema_update_every,
                        max_batches=remaining,
                        return_details=True,
                        args=args,  # TODO: remove to disable QC print statements
                    )
                    chunk_batches = int(details["batches_processed"])
                    if chunk_batches <= 0:
                        print("  WARNING: train epoch chunk processed 0 batches; stopping epoch early.")
                        break

                    weighted_loss_sum += float(details["loss"]) * chunk_batches
                    batches_done += chunk_batches

                    if not bool(details["reload_requested"]):
                        break

                    pending_chunk_reload = True

                train_loss = weighted_loss_sum / max(1, batches_done)

            if writer is not None:
                _log_per_dataset_figures(
                    model, train_loader, device, writer, epoch, train_loss
                )

            using_ema = ema is not None
            if using_ema:
                ema.store(model)
                ema.copy_to(model)
            val_loss = validate(
                model, val_loaders, criterion, device,
                writer=writer, epoch=epoch, thermal_guard=thermal_guard,
                max_batches=args.val_batches_per_epoch,
            )
            if using_ema:
                ema.restore(model)

            # Log scalar losses to TensorBoard
            writer.add_scalar("loss/train", train_loss, global_step=epoch + 1)
            writer.add_scalar("lr", current_lr, global_step=epoch + 1)
            if val_loaders:
                writer.add_scalar("loss/val", val_loss, global_step=epoch + 1)

            if val_loaders:
                print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Train Loss: {train_loss:.4f}")

            # End-of-epoch versioned checkpoint (never overwritten)
            epoch_ckpt = output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt"
            _save_checkpoint(epoch_ckpt, model, optimizer, scaler, epoch,
                             train_loss=train_loss, val_loss=val_loss,
                             train_paths=train_paths, val_paths=val_paths,
                             ema_state=ema.state_dict() if ema is not None else None)
            print(f"Saved checkpoint: {epoch_ckpt}")

            if scheduler is not None:
                scheduler.step()

            # --- timing ---
            now = time.monotonic()
            epochs_done = epoch + 1 - start_epoch
            epochs_left = args.epochs - (epoch + 1)
            epoch_elapsed = now - epoch_start
            total_elapsed = now - training_start
            avg_per_epoch = total_elapsed / epochs_done
            remaining_secs = avg_per_epoch * epochs_left

            def _fmt_duration(secs: float) -> str:
                secs = int(secs)
                h, rem = divmod(secs, 3600)
                m, s = divmod(rem, 60)
                if h:
                    return f"{h}h {m:02d}m {s:02d}s"
                if m:
                    return f"{m}m {s:02d}s"
                return f"{s}s"

            eta_dt = datetime.now() + timedelta(seconds=remaining_secs)
            eta_str = eta_dt.strftime("%d %b %Y %H:%M")
            print(
                f"Epoch time: {_fmt_duration(epoch_elapsed)} | "
                f"Elapsed: {_fmt_duration(total_elapsed)} | "
                f"Remaining: {_fmt_duration(remaining_secs)} | "
                f"ETA: {eta_str}"
            )

        final_path = output_dir / "final_model.pt"
        if ema is not None:
            ema.store(model)
            ema.copy_to(model)
            torch.save(model.state_dict(), final_path)
            ema.restore(model)
            raw_final_path = output_dir / "final_model_raw.pt"
            torch.save(model.state_dict(), raw_final_path)
            print(f"Saved raw non-EMA model: {raw_final_path}")
        else:
            torch.save(model.state_dict(), final_path)
        print(f"Training complete. Final model: {final_path}")
    finally:
        if monitor is not None:
            monitor.stop()
            print("Background monitor: stopped")
        writer.close()


if __name__ == "__main__":
    main()
