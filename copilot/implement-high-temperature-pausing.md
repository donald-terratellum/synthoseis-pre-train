You’ve already got a really solid training skeleton here—nice separation of dataloaders, model factory, and checkpointing. You even wrote:

> “Train for one epoch across all training datasets.
> Loop order: dataset -> batch.”

and

> “Save a resumable checkpoint. ds_idx=-1 means end-of-epoch.”

So let’s bolt a thermal guard onto this without wrecking that structure.

---

# Thermal‑Guard Integration for Seismic 3D Training

This document describes all required code changes to add:

- CPU temperature monitoring on macOS  
- Automatic thermal throttling  
- Automatic mid‑epoch checkpointing  
- Automatic resume from thermal checkpoints  
- Full integration with the existing 3D seismic training pipeline  

Your existing code already supports mid‑epoch checkpointing (`partial_latest.pt`) and full resume (`--resume`).  
We extend it with a thermal guard that uses the same mechanism.

---

## 1. Modify `gpu_utils.py` — Add CPU temperature helper

Target file: `synthoseis_pre_train/gpu_utils.py`

Add the following import near the top:

```python
import re



## 2. Add this function near the bottom of the file:

def get_cpu_temperature_c() -> Optional[float]:
    """
    Return CPU die temperature in Celsius on macOS using powermetrics.
    Returns None if unavailable or unsupported.
    """
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.check_output(
            ["sudo", "powermetrics", "--samplers", "smc", "--once"],
            text=True,
        )
        m = re.search(r"CPU die temperature:\s+([0-9.]+)", out)
        return float(m.group(1)) if m else None
    except Exception:
        return None



## 3. Modify train.py — Add CLI arguments
- Inside the main() parser section, add:

parser.add_argument("--thermal_max_c", type=float, default=85.0,
                    help="Pause/exit when CPU temperature exceeds this (C). Set <=0 to disable.")
parser.add_argument("--thermal_cooldown_sec", type=int, default=300,
                    help="Cooldown sleep in seconds after thermal trip.")
parser.add_argument("--thermal_check_every_batches", type=int, default=10,
                    help="Check temperature every N batches.")
parser.add_argument("--thermal_exit_on_trip", action="store_true",
                    help="Exit process on thermal trip (resume later with --resume).")


- Also import the helper:

from synthoseis_pre_train.gpu_utils import get_cpu_temperature_c


## 4. Add a ThermalGuard helper class
- Place this class near the top of train.py:

class ThermalGuard:
    def __init__(self, max_c, cooldown_sec, check_every_batches, exit_on_trip, output_dir):
        self.max_c = max_c
        self.cooldown_sec = cooldown_sec
        self.check_every_batches = max(1, check_every_batches)
        self.exit_on_trip = exit_on_trip
        self.output_dir = output_dir

    def maybe_throttle(self, epoch, ds_idx, batch_idx,
                       model, optimizer, scaler,
                       train_paths, val_paths):
        """
        Returns True if a thermal trip occurred.
        Saves a checkpoint 'thermal_latest.pt' before pausing/exiting.
        """
        if self.max_c <= 0:
            return False

        if batch_idx % self.check_every_batches != 0:
            return False

        temp = get_cpu_temperature_c()
        if temp is None or temp < self.max_c:
            return False

        print(f"\n🔥 Thermal trip: CPU {temp:.1f}°C >= {self.max_c:.1f}°C "
              f"(epoch {epoch+1}, dataset {ds_idx}, batch {batch_idx})")

        ckpt_path = self.output_dir / "thermal_latest.pt"
        _save_checkpoint(
            ckpt_path,
            model, optimizer, scaler, epoch,
            train_loss=float("nan"),
            val_loss=float("nan"),
            train_paths=train_paths,
            val_paths=val_paths,
            ds_idx=ds_idx,
        )
        print(f"  Saved thermal checkpoint: {ckpt_path}")

        if self.cooldown_sec > 0:
            print(f"  Cooling down for {self.cooldown_sec} seconds...")
            time.sleep(self.cooldown_sec)

        return True


## 4. Modify train_epoch to call the thermal guard
- Change the function signature:

def train_epoch(..., thermal_guard=None):


- Inside the batch loop, after optimizer step, insert:

if thermal_guard is not None:
    tripped = thermal_guard.maybe_throttle(
        epoch=epoch,
        ds_idx=ds_idx,
        batch_idx=batch_idx,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        train_paths=train_paths,
        val_paths=val_paths,
    )
    if tripped:
        return total_loss / max(total_batches, 1)


## 5. Instantiate the thermal guard in main()
- Add this before the training loop:

thermal_guard = ThermalGuard(
    max_c=args.thermal_max_c,
    cooldown_sec=args.thermal_cooldown_sec,
    check_every_batches=args.thermal_check_every_batches,
    exit_on_trip=args.thermal_exit_on_trip,
    output_dir=output_dir,
)


- Pass it into train_epoch:

train_loss = train_epoch(
    model, train_loaders, optimizer, criterion, device,
    scaler=scaler, writer=writer, epoch=epoch, output_dir=output_dir,
    train_paths=train_paths, val_paths=val_paths,
    thermal_guard=thermal_guard,
)


## 6. Stop the run cleanly after a thermal trip
- After calling train_epoch, add:

if (output_dir / "thermal_latest.pt").exists():
    print("Thermal checkpoint detected — stopping this run.")
    break


- Resume later with:

python train.py --resume checkpoints/thermal_latest.pt


## 7. Recommended thermal settings for Mac mini (M‑series)
- thermal_max_c 85
- thermal_cooldown_sec 180 to 300
- thermal_check_every_batches 10 to 20
- thermal_exit_on_trip optional

