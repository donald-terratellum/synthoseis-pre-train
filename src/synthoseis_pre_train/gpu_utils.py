"""GPU utility helpers for training and inference."""

import contextlib
import os
import platform
import re
import subprocess
from typing import Dict, Optional

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
                out = (exc.stdout or "") + ("\n" + exc.stderr if exc.stderr else "")
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
            out = (exc.stdout or "") + ("\n" + exc.stderr if exc.stderr else "")
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
