"""Train the Go1 walking (velocity-tracking) policy with MuJoCo Playground + Brax PPO on a GPU.

Uses the separate .venv-rl environment (JAX + MJX/Warp + Playground). The trained network is
exported to a NumPy file so Morph can run it on CPU without JAX.

    MORPH_GPU_MEM=0.7 .venv-rl/bin/python scripts/train_go1_walk.py --steps 200_000_000 --envs 4096

Task: Playground's Go1JoystickFlatTerrain (Apache-2.0). Observation (48): local linear velocity,
gyro, gravity direction, joint angles - default pose, joint velocities, last action, command
(vx, vy, wz). Action (12): joint targets = default pose + 0.5 * action, PD Kp=35, Kd=0.5, 50 Hz.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", os.environ.get("MORPH_GPU_MEM", "0.7"))
os.environ.setdefault("MUJOCO_GL", "egl")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

if not hasattr(jax, "device_put_replicated"):
    # Brax 0.14.2 still calls this; JAX 0.11 removed it. Drop-in for the replicated leading device axis
    # (see https://docs.jax.dev/en/latest/migrate_pmap.html#drop-in-replacements).
    def _device_put_replicated(x, devices):
        return jax.tree.map(lambda a: jax.device_put_sharded([a] * len(devices), devices)
                            if hasattr(jax, "device_put_sharded")
                            else jax.device_put(jnp.stack([a] * len(devices))), x)
    jax.device_put_replicated = _device_put_replicated
import mujoco.mjx._src.io as _mjx_io  # noqa: E402

_orig_put_model = _mjx_io.mjwp.put_model


def _quiet_put_model(*args, **kwargs):
    # Playground deliberately runs 1 solver / 5 line-search iterations for speed; MuJoCo Warp then
    # printf's an overflow warning from the GPU on nearly every step (374 MB of log in 20 min),
    # which throttled training. Silence the print only; the solver behaviour is unchanged.
    m = _orig_put_model(*args, **kwargs)
    m.opt.warn_overflow = 0
    return m


_mjx_io.mjwp.put_model = _quiet_put_model
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402
from mujoco_playground import registry, wrapper  # noqa: E402
from mujoco_playground.config import locomotion_params  # noqa: E402

ENV = "Go1JoystickFlatTerrain"


def export_numpy(params, out: Path, obs_key: str = "state") -> dict:
    """Flatten (normalizer, policy) params into an .npz Morph can evaluate with NumPy."""
    norm, policy = params[0], params[1]
    arrays = {}
    mean, std = norm.mean, norm.std
    if isinstance(mean, dict):
        mean, std = mean[obs_key], std[obs_key]
    arrays["obs_mean"], arrays["obs_std"] = np.asarray(mean), np.asarray(std)
    layers = policy["params"]
    names = sorted(layers.keys(), key=lambda k: int(k.split("_")[-1]))
    for i, n in enumerate(names):
        arrays[f"W{i}"] = np.asarray(layers[n]["kernel"])
        arrays[f"b{i}"] = np.asarray(layers[n]["bias"])
    np.savez(out, **arrays)
    return {"layers": [list(arrays[f"W{i}"].shape) for i in range(len(names))]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000_000)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--evals", type=int, default=10)
    ap.add_argument("--out", default="artifacts/go1_walk")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    env_cfg = registry.get_default_config(ENV)
    env = registry.load(ENV, config=env_cfg)
    eval_env = registry.load(ENV, config=env_cfg)
    p = locomotion_params.brax_ppo_config(ENV)
    p.num_timesteps = a.steps
    p.num_envs = a.envs
    p.num_evals = a.evals
    net = dict(p.network_factory)
    kw = {k: v for k, v in p.items() if k != "network_factory"}

    log, t0 = [], time.time()

    def progress(step, metrics):
        rec = {"step": int(step), "wall_s": round(time.time() - t0, 1),
               "reward": float(metrics.get("eval/episode_reward", float("nan"))),
               "reward_std": float(metrics.get("eval/episode_reward_std", float("nan"))),
               "episode_length": float(metrics.get("eval/avg_episode_length", float("nan")))}
        log.append(rec)
        print(json.dumps(rec), flush=True)

    train = functools.partial(
        ppo.train, **kw, seed=a.seed, progress_fn=progress,
        network_factory=functools.partial(ppo_networks.make_ppo_networks, **net))
    make_inference_fn, params, _ = train(environment=env, eval_env=eval_env,
                                         wrap_env_fn=wrapper.wrap_for_brax_training)
    wall = time.time() - t0

    # Export and verify the NumPy re-implementation against Brax's own inference on random observations.
    shapes = export_numpy(params, out / "go1_walk_policy.npz")
    infer = make_inference_fn(params, deterministic=True)
    rng = np.random.default_rng(0)
    obs_dim = int(np.load(out / "go1_walk_policy.npz")["obs_mean"].shape[0])
    obs = rng.normal(size=(8, obs_dim)).astype(np.float32)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from morph.robots.go1_learned import NumpyPolicy
    npol = NumpyPolicy(out / "go1_walk_policy.npz")
    ref = np.stack([np.asarray(infer({"state": o, "privileged_state": np.zeros(1)}, jax.random.PRNGKey(0))[0]) for o in obs])
    mine = np.stack([npol(o) for o in obs])
    err = float(np.max(np.abs(ref - mine)))

    meta = {"env": ENV, "steps": a.steps, "num_envs": a.envs, "wall_s": round(wall, 1),
            "steps_per_s": round(a.steps / wall), "gpu": str(jax.devices()[0]), "jax": jax.__version__,
            "numpy_export_max_abs_err": err, "network": shapes, "curve": log,
            "env_config": {"ctrl_dt": env_cfg.ctrl_dt, "sim_dt": env_cfg.sim_dt, "Kp": env_cfg.Kp, "Kd": env_cfg.Kd,
                           "action_scale": env_cfg.action_scale}}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "curve"}, indent=2))


if __name__ == "__main__":
    main()
