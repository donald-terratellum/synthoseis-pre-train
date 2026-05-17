"""GPU utility helpers for training and inference."""

import contextlib
import csv
from datetime import datetime
import os
import platform
import re
import subprocess
import threading
from pathlib import Path
from typing import Dict, Optional, List, Any

# Environment passed to all child processes — set Malloc* vars to "0" rather
# than omitting them, so libmalloc sees an explicit disable signal.
_CLEAN_ENV: dict = {
    **{k: v for k, v in os.environ.items()},
    "MallocStackLogging": "0",
    "MallocStackLoggingNoCompact": "0",
}

import torch


_POWERMETRICS_SAMPLER: Optional[str] = None
# Set to True after all sampler discovery fails so future calls skip subprocess
# overhead entirely (stops the 12 fork+exec calls per thermal check cycle).
_POWERMETRICS_UNAVAILABLE: bool = False
_POWERMETRICS_GPU_UNAVAILABLE: bool = False


def _as_text(value) -> str:
    """Best-effort conversion of subprocess output to text."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def get_default_device(preferred: str = "auto") -> torch.device:
    """Return the best available device based on preference and hardware."""
    desired = preferred.lower() if preferred else "auto"
    if desired == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if desired == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if desired == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_system_memory_bytes() -> int:
    """Estimate the system memory size in bytes on macOS or fallback."""
    if platform.system() == "Darwin":
        try:
            output = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True,
                env=_CLEAN_ENV,
            )
            return int(output.strip())
        except Exception:
            pass
    return 24 * 1024 ** 3


def get_memory_info(device: torch.device) -> Dict[str, Optional[int]]:
    """Return memory usage info for CUDA/MPS/CPU devices."""
    info = {
        "device": device.type,
        "total_bytes": None,
        "free_bytes": None,
    }

    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else 0
        free, total = torch.cuda.mem_get_info(idx)
        info["total_bytes"] = int(total)
        info["free_bytes"] = int(free)
    elif device.type == "mps":
        info["total_bytes"] = get_system_memory_bytes()
        info["free_bytes"] = info["total_bytes"]
    else:
        info["total_bytes"] = get_system_memory_bytes()
        info["free_bytes"] = info["total_bytes"]

    return info


def print_device_summary(preferred: str = "auto") -> None:
    """Print device selection and basic hardware info."""
    device = get_default_device(preferred)
    memory_info = get_memory_info(device)

    print("=== GPU / Device Summary ===")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Selected device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Total memory: {memory_info['total_bytes'] / 1024 ** 3:.2f} GB")
    if memory_info["free_bytes"] is not None:
        print(f"Free memory: {memory_info['free_bytes'] / 1024 ** 3:.2f} GB")


def autocast_context(device: torch.device):
    """Return fp16 autocast context for devices that support it."""
    if device.type in ["cuda", "mps"]:
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return contextlib.nullcontext()


def create_grad_scaler(device: torch.device):
    """Create a GradScaler for CUDA and return None on other devices."""
    if device.type == "cuda":
        return torch.cuda.amp.GradScaler()
    return None


def get_cpu_temperature_c() -> Optional[float]:
    """Return CPU die temperature in Celsius on macOS, else None.

    Uses non-interactive sudo so long-running training never blocks on a
    password prompt. If sudo credentials are unavailable, returns None.
    """
    global _POWERMETRICS_SAMPLER

    global _POWERMETRICS_UNAVAILABLE

    if platform.system() != "Darwin" or _POWERMETRICS_UNAVAILABLE:
        return None

    def _parse_temp(out: str) -> Optional[float]:
        # Prefer explicit CPU temperature lines.
        m = re.search(
            r"CPU[^\n:]*temperature[^\n:]*:\s*([0-9]+(?:\.[0-9]+)?)\s*([CF])?",
            out,
            re.IGNORECASE,
        )
        if m:
            v = float(m.group(1))
            unit = (m.group(2) or "C").upper()
            return (v - 32.0) * 5.0 / 9.0 if unit == "F" else v
        # Fallback: any temperature line.
        m = re.search(
            r"temperature[^\n:]*:\s*([0-9]+(?:\.[0-9]+)?)\s*([CF])?",
            out,
            re.IGNORECASE,
        )
        if m:
            v = float(m.group(1))
            unit = (m.group(2) or "C").upper()
            return (v - 32.0) * 5.0 / 9.0 if unit == "F" else v

        # Last resort: any '<number><optional degree> C/F' on a line mentioning temp.
        for line in out.splitlines():
            if "temp" not in line.lower():
                continue
            mm = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:°\s*)?([CF])", line, re.IGNORECASE)
            if not mm:
                continue
            v = float(mm.group(1))
            unit = mm.group(2).upper()
            return (v - 32.0) * 5.0 / 9.0 if unit == "F" else v
        return None

    def _run_once(sampler: str) -> Optional[str]:
        base = ["sudo", "-n", "/usr/bin/powermetrics", "--samplers", sampler]
        cmd_variants = [
            base + ["--once"],                    # Newer macOS
            base + ["-n", "1", "-i", "1000"],  # Older macOS
            base + ["-i", "1000", "-n", "1"],  # Same, different order
        ]

        for cmd in cmd_variants:
            try:
                proc = subprocess.run(cmd, text=True, capture_output=True, timeout=4, env=_CLEAN_ENV)
                out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
            except subprocess.TimeoutExpired as exc:
                # Some builds stream indefinitely unless interrupted; parse what we got.
                out = _as_text(exc.stdout) + ("\n" + _as_text(exc.stderr) if exc.stderr else "")
            except Exception:
                continue

            msg = out.lower()
            if "unrecognized sampler" in msg or "invalid sampler" in msg:
                return None
            if "unrecognized option" in msg and "--once" in msg:
                # Try the next command variant without --once.
                continue
            if out.strip():
                return out

        return ""

    # Fast path: use previously discovered sampler.
    if _POWERMETRICS_SAMPLER is not None:
        out = _run_once(_POWERMETRICS_SAMPLER)
        if out:
            temp = _parse_temp(out)
            if temp is not None:
                return temp

    # Discover compatible sampler on this macOS build.
    for sampler in ("smc", "thermal", "cpu_power"):
        out = _run_once(sampler)
        if out is None:
            continue
        if out:
            temp = _parse_temp(out)
            if temp is not None:
                _POWERMETRICS_SAMPLER = sampler
                return temp
    # All discovery attempts failed — mark unavailable so future calls are no-ops.
    _POWERMETRICS_UNAVAILABLE = True
    return None


def get_thermal_pressure_level() -> Optional[str]:
    """Return thermal pressure level on macOS (e.g., Nominal/Serious/Critical)."""
    if platform.system() != "Darwin" or _POWERMETRICS_UNAVAILABLE:
        return None

    cmd_variants = [
        ["sudo", "-n", "/usr/bin/powermetrics", "--samplers", "thermal", "--once"],
        ["sudo", "-n", "/usr/bin/powermetrics", "--samplers", "thermal", "-n", "1", "-i", "1000"],
        ["sudo", "-n", "/usr/bin/powermetrics", "--samplers", "thermal", "-i", "1000", "-n", "1"],
    ]

    for cmd in cmd_variants:
        try:
            proc = subprocess.run(cmd, text=True, capture_output=True, timeout=4, env=_CLEAN_ENV)
            out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        except subprocess.TimeoutExpired as exc:
            out = _as_text(exc.stdout) + ("\n" + _as_text(exc.stderr) if exc.stderr else "")
        except Exception:
            continue

        msg = out.lower()
        if "unrecognized option" in msg and "--once" in msg:
            continue
        m = re.search(r"Current pressure level:\s*([A-Za-z]+)", out, re.IGNORECASE)
        if m:
            level = m.group(1).strip().capitalize()
            return level
    return None


def get_system_gpu_percent() -> Optional[float]:
    """Return best-effort system GPU activity percent on macOS, else None.

    This is system-level GPU activity, not per-process usage.
    """
    global _POWERMETRICS_GPU_UNAVAILABLE
    if platform.system() != "Darwin" or _POWERMETRICS_GPU_UNAVAILABLE:
        return None

    cmd_variants = [
        ["sudo", "-n", "/usr/bin/powermetrics", "--samplers", "gpu_power", "--once"],
        ["sudo", "-n", "/usr/bin/powermetrics", "--samplers", "gpu_power", "-n", "1", "-i", "1000"],
        ["sudo", "-n", "/usr/bin/powermetrics", "--samplers", "gpu_power", "-i", "1000", "-n", "1"],
    ]

    for cmd in cmd_variants:
        try:
            proc = subprocess.run(cmd, text=True, capture_output=True, timeout=4, env=_CLEAN_ENV)
            out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        except subprocess.TimeoutExpired as exc:
            out = _as_text(exc.stdout) + ("\n" + _as_text(exc.stderr) if exc.stderr else "")
        except Exception:
            continue

        msg = out.lower()
        if "unrecognized sampler" in msg or "invalid sampler" in msg:
            _POWERMETRICS_GPU_UNAVAILABLE = True
            return None
        if "unrecognized option" in msg and "--once" in msg:
            continue

        m = re.search(r"GPU[^\n:]*active[^\n:]*:\s*([0-9]+(?:\.[0-9]+)?)%", out, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    return None


def get_mps_allocated_gb(device: Optional[torch.device] = None) -> tuple[Optional[float], Optional[float]]:
    """Return (current_allocated_gb, driver_allocated_gb) for MPS when available."""
    try:
        if device is not None and getattr(device, "type", None) != "mps":
            return None, None
        if not torch.backends.mps.is_available():
            return None, None
        if not hasattr(torch, "mps"):
            return None, None

        current = None
        driver = None
        if hasattr(torch.mps, "current_allocated_memory"):
            current = float(torch.mps.current_allocated_memory()) / (1024.0 ** 3)
        if hasattr(torch.mps, "driver_allocated_memory"):
            driver = float(torch.mps.driver_allocated_memory()) / (1024.0 ** 3)
        return current, driver
    except Exception:
        return None, None


def _collect_process_tree_usage(root_pid: int, include_children: bool = True) -> tuple[int, float, float]:
    """Return (process_count, cpu_percent_sum, rss_gb_sum) for a pid tree.

    CPU percent uses the instantaneous %CPU reported by ps.
    """
    try:
        proc = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,%cpu=,rss="],
            text=True,
            capture_output=True,
            timeout=3,
            env=_CLEAN_ENV,
        )
    except Exception:
        return 0, 0.0, 0.0

    rows = []
    for line in (proc.stdout or "").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            cpu = float(parts[2])
            rss_kb = int(parts[3])
        except Exception:
            continue
        rows.append((pid, ppid, cpu, rss_kb))

    if not rows:
        return 0, 0.0, 0.0

    children: dict[int, list[int]] = {}
    metrics: dict[int, tuple[float, int]] = {}
    for pid, ppid, cpu, rss_kb in rows:
        children.setdefault(ppid, []).append(pid)
        metrics[pid] = (cpu, rss_kb)

    if root_pid not in metrics:
        return 0, 0.0, 0.0

    selected = {root_pid}
    if include_children:
        stack = [root_pid]
        while stack:
            cur = stack.pop()
            for ch in children.get(cur, []):
                if ch not in selected:
                    selected.add(ch)
                    stack.append(ch)

    cpu_sum = 0.0
    rss_kb_sum = 0
    for pid in selected:
        cpu, rss_kb = metrics.get(pid, (0.0, 0))
        cpu_sum += cpu
        rss_kb_sum += rss_kb

    rss_gb = float(rss_kb_sum) / (1024.0 ** 2)
    return len(selected), cpu_sum, rss_gb


class ProcessTreeCsvMonitor:
    """Background process-tree monitor that appends one CSV row per interval."""

    def __init__(
        self,
        root_pid: int,
        csv_path: str,
        interval_sec: float = 15.0,
        include_children: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.root_pid = int(root_pid)
        self.csv_path = Path(csv_path)
        self.interval_sec = max(1.0, float(interval_sec))
        self.include_children = bool(include_children)
        self.device = device
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._warned_runtime_error = False
        self._header = [
            "timestamp",
            "root_pid",
            "process_count",
            "cpu_percent_tree",
            "memory_gb_tree",
            "gpu_percent_system",
            "gpu_mps_current_allocated_gb",
            "gpu_mps_driver_allocated_gb",
            "cpu_temp_c",
            "thermal_pressure_level",
        ]

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = (not self.csv_path.exists()) or self.csv_path.stat().st_size == 0
        if write_header:
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self._header)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="process-tree-csv-monitor", daemon=True)
        self._thread.start()

    def stop(self, timeout_sec: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, float(timeout_sec)))

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._write_row()
            except Exception as exc:
                # Never let monitoring crash the training process.
                if not self._warned_runtime_error:
                    print(f"Background monitor warning: row sampling failed ({exc})")
                    self._warned_runtime_error = True
            self._stop_event.wait(self.interval_sec)

    def _write_row(self) -> None:
        proc_count, cpu_pct, mem_gb = _collect_process_tree_usage(
            self.root_pid,
            include_children=self.include_children,
        )
        gpu_pct = get_system_gpu_percent()
        mps_current_gb, mps_driver_gb = get_mps_allocated_gb(self.device)
        temp_c = get_cpu_temperature_c()
        pressure = get_thermal_pressure_level()

        row = [
            datetime.now().isoformat(timespec="seconds"),
            self.root_pid,
            proc_count,
            round(cpu_pct, 3),
            round(mem_gb, 6),
            "" if gpu_pct is None else round(gpu_pct, 3),
            "" if mps_current_gb is None else round(mps_current_gb, 6),
            "" if mps_driver_gb is None else round(mps_driver_gb, 6),
            "" if temp_c is None else round(temp_c, 3),
            "" if pressure is None else pressure,
        ]

        try:
            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
        except Exception:
            # Keep training unaffected if monitor IO fails.
            return
