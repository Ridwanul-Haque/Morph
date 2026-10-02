"""Where the heavy work runs, and what it costs.

Only gait training (PPO on a GPU, ~300M simulation steps) is heavy. Everything else (trials, setup,
planning) runs on this machine's CPU in seconds to minutes; Nemotron always runs in the cloud and is
billed per token.

Cloud GPUs are Nebius AI Cloud on-demand prices from https://nebius.com/prices (checked Sept 30 2026). Nebius
raises them on Oct 1 2026 (H100 $3.85 -> $4.50, H200 $4.50 -> $5.40, B200 $7.15 -> $8.50); the table below
switches on that date by itself. Cloud training times are estimates: the local time is measured, the
per-GPU speed-ups relative to it are assumptions, and a fixed setup overhead is added (VM start, install,
JAX compile), which is billed too.

Nemotron's price is read from Nebius's model list (the API reports it per token) and falls back to the
last known one. On this computer, training costs only electricity: estimated from the GPU's power
limit and an electricity price you can set (MORPH_KWH_PRICE, default $0.17 per kWh).

Morph never starts paid cloud resources by itself: choosing the cloud prepares a job folder (script +
instructions) that you run with your own Nebius account.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRICES_CHECKED = "Sept 30 2026"
PRICE_URL = "https://nebius.com/prices"
PRICE_CHANGE = datetime.date(2026, 10, 1)

# $/GPU-hour on demand (before, from Oct 1 2026), assumed speed-up vs the measured local run
CLOUD_GPUS = {
    "L40S": {"label": "NVIDIA L40S", "prices": (1.55, 1.55), "speedup": 1.5},
    "RTX PRO 6000": {"label": "NVIDIA RTX PRO 6000", "prices": (1.80, 1.80), "speedup": 2.0},
    "H100": {"label": "NVIDIA H100", "prices": (3.85, 4.50), "speedup": 3.0},
    "H200": {"label": "NVIDIA H200", "prices": (4.50, 5.40), "speedup": 3.3},
    "B200": {"label": "NVIDIA B200", "prices": (7.15, 8.50), "speedup": 4.5},
}
SETUP_OVERHEAD_S = 8 * 60
NEMOTRON_PRICE = {"input": 0.30, "output": 0.90}  # $ per 1M tokens, Nebius Token Factory (fallback)
KWH_PRICE = float(os.environ.get("MORPH_KWH_PRICE", "0.17"))


def gpu_price(key: str, today: datetime.date | None = None) -> float:
    old, new = CLOUD_GPUS[key]["prices"]
    return new if (today or datetime.date.today()) >= PRICE_CHANGE else old


def price_note(today: datetime.date | None = None) -> str:
    if (today or datetime.date.today()) >= PRICE_CHANGE:
        return "prices as raised on Oct 1 2026"
    return "prices rise on Oct 1 2026 (H100 $4.50/h, H200 $5.40/h, B200 $8.50/h)"


_LIVE_PRICE: dict = {}


def nemotron_price(model: str | None = None) -> dict:
    """$ per 1M tokens for the configured model, from Nebius's model list (once per session)."""
    model = model or os.environ.get("MORPH_NEMOTRON_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    if model in _LIVE_PRICE:
        return _LIVE_PRICE[model]
    out = dict(NEMOTRON_PRICE, source="last known price")
    key, base = os.environ.get("NEBIUS_API_KEY"), os.environ.get("NEBIUS_BASE_URL")
    if key and base:
        try:
            req = urllib.request.Request(base.rstrip("/") + "/models?verbose=true", headers={"Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=8) as r:
                for m in json.loads(r.read()).get("data", []):
                    if m.get("id") == model and m.get("pricing"):
                        out = {"input": float(m["pricing"]["prompt"]) * 1e6, "output": float(m["pricing"]["completion"]) * 1e6,
                               "source": "Nebius, live"}
        except Exception:  # noqa: BLE001  (offline: use the last known price)
            pass
    _LIVE_PRICE[model] = out
    return out


def local_power_w() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def local_gpu() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip().splitlines()[0] or "unknown GPU"
    except Exception:  # noqa: BLE001
        return "no NVIDIA GPU found"


def measured_training() -> dict:
    """The last gait training run on this machine (steps, seconds) as the baseline."""
    for meta in sorted((ROOT / "artifacts" / "walk").glob("*/train_meta.json")):
        m = json.loads(meta.read_text())
        return {"steps": m["steps"], "wall_s": m["wall_s"], "robot": m.get("robot", meta.parent.name)}
    return {"steps": 300_000_000, "wall_s": 1940.0, "robot": "go2"}


def estimates(steps: int = 300_000_000) -> dict:
    base = measured_training()
    local_s = base["wall_s"] * steps / base["steps"]
    cloud = {}
    for key, g in CLOUD_GPUS.items():
        s = local_s / g["speedup"] + SETUP_OVERHEAD_S
        price = gpu_price(key)
        cloud[key] = {"label": g["label"], "price_per_hour": price, "minutes": round(s / 60),
                      "cost": round(s / 3600 * price, 2)}
    return {"steps": steps, "local": _local(local_s, f"measured: {base['robot']}, {base['steps'] / 1e6:.0f}M steps in "
                                                      f"{base['wall_s'] / 60:.0f} min"),
            "cloud": cloud, "prices_checked": PRICES_CHECKED, "price_url": PRICE_URL,
            "note": "cloud times are estimates (assumed speed-ups + ~8 min setup, which is billed); " + price_note()}


def _local(local_s: float, basis: str) -> dict:
    """This computer: no bill, only electricity (GPU at its power limit, the rest of the PC ~100 W)."""
    w = local_power_w()
    kwh = local_s / 3600 * ((w or 250.0) + 100.0) / 1000
    return {"gpu": local_gpu(), "minutes": round(local_s / 60), "cost": 0.0, "basis": basis,
            "electricity": round(kwh * KWH_PRICE, 2), "kwh": round(kwh, 2), "kwh_price": KWH_PRICE}


def task_estimates(steps: int = 60_000_000) -> dict:
    """Learning a task: measured runs of it if there are any, else scaled from the gait training."""
    rates = []
    for meta in (ROOT / "runs" / "learned").glob("*/train_meta.json"):
        m = json.loads(meta.read_text())
        if m.get("wall_s"):
            rates.append(m["steps"] / m["wall_s"])
    e = estimates(steps)
    if rates:
        local_s = steps / (sum(rates) / len(rates))
        e["local"] = _local(local_s, f"measured on this computer ({len(rates)} earlier run{'s' if len(rates) > 1 else ''})")
        for key, g in e["cloud"].items():
            s = local_s / CLOUD_GPUS[key]["speedup"] + SETUP_OVERHEAD_S
            g["minutes"], g["cost"] = round(s / 60), round(s / 3600 * g["price_per_hour"], 2)
    return e


def nemotron_cost(records) -> dict:
    """From the token counts Nebius returns with every reply (reasoning tokens are part of the output)."""
    tin = sum((r.usage or {}).get("prompt_tokens", 0) for r in records)
    tout = sum((r.usage or {}).get("completion_tokens", 0) for r in records)
    p = nemotron_price()
    return {"calls": len(records), "input_tokens": tin, "output_tokens": tout,
            "cost": round(tin / 1e6 * p["input"] + tout / 1e6 * p["output"], 4),
            "price_per_1m": {"input": round(p["input"], 4), "output": round(p["output"], 4)}, "price_source": p["source"]}


def cloud_task_job(task_json: str, robot: str, gpu: str, steps: int = 60_000_000, randomize: str = "light") -> dict:
    """Job folder for learning a task on a Nebius GPU (same shape as the gait job)."""
    est = task_estimates(steps)["cloud"][gpu]
    src = Path(task_json).parent
    d = ROOT / "runs" / "cloud_jobs" / f"{src.name}_{gpu.replace(' ', '_')}"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(task_json, d / "task.json")
    (d / "run.sh").write_text(f"""#!/usr/bin/env bash
# Learn '{src.name}' for {robot} on a Nebius {CLOUD_GPUS[gpu]['label']} VM.
# Estimate: ~{est['minutes']} min, ~${est['cost']} at ${est['price_per_hour']}/h on demand (prices checked {PRICES_CHECKED}; {price_note()}).
# task.json points at the robot and scene files by path: copy those to the same paths on the VM (or edit task.json).
set -euo pipefail
git clone <your Morph repo URL> morph && cd morph
python3 -m venv .venv-rl && .venv-rl/bin/pip install -r requirements-rl.lock
mkdir -p runs/learned/{src.name} && cp ../task.json runs/learned/{src.name}/
MORPH_TASK_GPU_MEM=0.6 .venv-rl/bin/python scripts/train_task.py runs/learned/{src.name}/task.json --steps {steps} --randomize {randomize}
echo "done: copy runs/learned/{src.name}/ back into your local Morph, then stop the VM"
""")
    return {"folder": str(d), "estimate": est, "gpu": CLOUD_GPUS[gpu]["label"],
            "steps": ["Create a Nebius VM with one " + CLOUD_GPUS[gpu]["label"], f"Copy {d.name}/ to it and run: bash run.sh",
                      f"Copy runs/learned/{src.name}/ back here; Morph uses the learned skill automatically",
                      "Stop the VM (billed per minute)"]}


def cloud_job(model_path: str, robot: str, gpu: str, steps: int = 300_000_000) -> dict:
    """Write a self-contained job folder for training this robot's gait on a Nebius GPU VM."""
    est = estimates(steps)["cloud"][gpu]
    d = ROOT / "runs" / "cloud_jobs" / f"{robot}_{gpu.replace(' ', '_')}"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(model_path, d / Path(model_path).name)  # the model file (mesh assets: copy its folder too)
    run = f"""#!/usr/bin/env bash
# Train a walking policy for {robot} on a Nebius {CLOUD_GPUS[gpu]['label']} VM.
# Estimate: ~{est['minutes']} min, ~${est['cost']} at ${est['price_per_hour']}/h on demand (prices checked {PRICES_CHECKED}; {price_note()}).
set -euo pipefail
git clone <your Morph repo URL> morph && cd morph
python3 -m venv .venv-rl && .venv-rl/bin/pip install -r requirements-rl.lock
bash scripts/fetch_assets.sh --extra
MORPH_GPU_MEM=0.85 .venv-rl/bin/python scripts/train_walk.py {Path(model_path).relative_to(ROOT) if str(model_path).startswith(str(ROOT)) else Path(model_path).name} \\
    --steps {steps} --out artifacts/walk/{robot}
echo "done: copy artifacts/walk/{robot}/ back into your local Morph checkout, then stop the VM"
"""
    (d / "run.sh").write_text(run)
    (d / "README.txt").write_text(
        f"""Cloud gait training for {robot}

1. In the Nebius console, create a VM with one {CLOUD_GPUS[gpu]['label']} and an Ubuntu + CUDA image.
2. Copy this folder to it and run:  bash run.sh
3. Copy artifacts/walk/{robot}/ back into Morph (same path). Morph picks the gait up automatically.
4. Stop the VM: it is billed per minute while running.

Estimated: ~{est['minutes']} min, ~${est['cost']} (${est['price_per_hour']}/GPU-hour on demand; prices from
{PRICE_URL}, checked {PRICES_CHECKED}; {price_note()}). Preemptible VMs are cheaper but can be
stopped mid-run.
""")
    return {"folder": str(d), "estimate": est, "gpu": CLOUD_GPUS[gpu]["label"],
            "steps": ["Create a Nebius VM with one " + CLOUD_GPUS[gpu]["label"],
                      f"Copy {d.name}/ to it and run: bash run.sh",
                      f"Copy artifacts/walk/{robot}/ back here; Morph uses it automatically",
                      "Stop the VM (billed per minute)"]}
