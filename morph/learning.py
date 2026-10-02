"""Behaviour cloning + DAgger for pick-and-place, trained on the source arm only.

Two policies with the same network, data budget and training states, but a different
interface to the body:

* ``joint``: sees the arm's joint angles and outputs joint deltas (the usual way a
  robot policy gets "copied").
* ``task``: sees and outputs only object/gripper-space quantities; each arm turns the
  action into joint targets with its own IK.

Both get evaluated zero-shot on other arms to see what actually transfers.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .arm_skills import (ArmState, expert_action, joint_obs, run_episode, task_obs, task_to_joint_label)
from .robots.arm import ArmAdapter
from .sim import PhysicalParams, SimulationUnstable
from .task import default_spec

torch.set_num_threads(int(__import__("os").environ.get("MORPH_TORCH_THREADS", "4")))


# ------------------------------------------------------------------ sampling
@dataclass(frozen=True)
class Scenario:
    seed: int
    block: tuple[float, float, float]
    goal: tuple[float, float, float]
    params: PhysicalParams

    def as_dict(self) -> dict:
        return {"seed": self.seed, "block": list(self.block), "goal": list(self.goal), "params": self.params.as_dict()}


def sample_scenario(seed: int, perturb: bool = True) -> Scenario:
    """Initial block pose, goal pose and physics. Seeds < 10_000 are training, >= 10_000 held-out."""
    rng = np.random.default_rng(seed)
    block = (rng.uniform(0.36, 0.56), rng.uniform(-0.30, -0.08), rng.uniform(-math.pi, math.pi))
    goal = (rng.uniform(0.36, 0.56), rng.uniform(0.08, 0.30), rng.uniform(-math.pi, math.pi))
    params = PhysicalParams(block_mass=float(rng.uniform(0.1, 0.4)), block_friction=float(rng.uniform(0.4, 0.8))) \
        if perturb else PhysicalParams()
    return Scenario(seed, tuple(float(v) for v in block), tuple(float(v) for v in goal), params)


TRAIN_SEEDS = range(0, 10_000)
HELDOUT_SEEDS = range(10_000, 20_000)


# ------------------------------------------------------------------ model
class Policy(nn.Module):
    def __init__(self, n_obs: int, n_act: int, hidden: int = 256):
        super().__init__()
        self.n_act = n_act
        self.net = nn.Sequential(nn.Linear(n_obs, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_act))
        self.register_buffer("mu", torch.zeros(n_obs))
        self.register_buffer("sd", torch.ones(n_obs))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net((obs - self.mu) / self.sd)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        out = self.forward(torch.as_tensor(obs, dtype=torch.float32)[None])[0].numpy()
        a = np.tanh(out[:-1])
        grip = 1.0 if out[-1] > 0 else -1.0  # last output is a grip logit
        return np.concatenate([a, [grip]]).astype(np.float32)


def fit(policy: Policy, X: np.ndarray, Y: np.ndarray, epochs: int = 40, lr: float = 1e-3, seed: int = 0,
        renorm: bool = True) -> list[float]:
    torch.manual_seed(seed)
    X_t, Y_t = torch.as_tensor(X), torch.as_tensor(Y)
    if renorm:  # fine-tuning keeps the pretrained normalisation so the old weights still mean the same thing
        policy.mu.copy_(X_t.mean(0))
        policy.sd.copy_(X_t.std(0) + 1e-3)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(X_t)
    losses = []
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            out = policy(X_t[idx])
            cont = nn.functional.mse_loss(torch.tanh(out[:, :-1]), Y_t[idx, :-1])
            grip = nn.functional.binary_cross_entropy_with_logits(out[:, -1], (Y_t[idx, -1] > 0).float())
            loss = cont + 0.1 * grip
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        losses.append(tot / n)
    return losses


# ------------------------------------------------------------------ DAgger
def _label(arm: ArmAdapter, kind: str, s: ArmState) -> tuple[np.ndarray, np.ndarray] | None:
    a, _ = expert_action(s)
    if kind == "task":
        return task_obs(s), a
    lab = task_to_joint_label(arm, s, a)
    return (joint_obs(s), lab) if lab is not None else None


def collect(arm: ArmAdapter, kind: str, seeds, policy: Policy | None, beta: float, noise: float,
            max_time: float = 10.0) -> tuple[list, list, dict]:
    """Roll out (expert if policy is None, else policy with prob 1-beta per tick) and label every state with the expert."""
    X, Y = [], []
    stats = {"episodes": 0, "success": 0, "unstable": 0}
    for sd in seeds:
        sc = sample_scenario(sd)
        rng = np.random.default_rng(sd + 777)
        arm.reset(sd, sc.block, sc.params)
        spec = default_spec(*sc.goal)

        def label_fn(s):
            r = _label(arm, kind, s)
            if r is not None:
                X.append(r[0])
                Y.append(r[1])

        def pol(s):
            ea, ph = expert_action(s)
            if policy is None or rng.random() < beta:
                if kind == "task":
                    return ea, ph
                lab = task_to_joint_label(arm, s, ea)
                return (lab if lab is not None else np.zeros(len(s.q) + 1, np.float32)), ph
            o = task_obs(s) if kind == "task" else joint_obs(s)
            return policy.act(o), "policy"

        try:
            r = run_episode(arm, spec, pol, mode=kind, max_time=max_time, label_fn=label_fn, noise=noise, rng=rng,
                            stop_when_done=policy is None)
            stats["success"] += int(r.outcome.success)
        except SimulationUnstable:
            stats["unstable"] += 1
        stats["episodes"] += 1
    return X, Y, stats


def train_dagger(kind: str, out_dir: Path, n_expert: int = 150, n_iter: int = 3, n_per_iter: int = 60,
                 epochs: int = 150, iter_epochs: int = 60, noise: float = 0.1, log=print,
                 source: str = "arm_A") -> tuple[Policy, dict]:
    from .robots.registry import make_arm
    from .arm_skills import N_OBS_TASK
    arm = make_arm(source)
    nq = len(arm.home_q)
    n_obs = N_OBS_TASK if kind == "task" else N_OBS_TASK + nq
    n_act = 5 if kind == "task" else nq + 1
    policy = Policy(n_obs, n_act)
    seeds = iter(TRAIN_SEEDS)
    t0 = time.time()
    X, Y, st = collect(arm, kind, [next(seeds) for _ in range(n_expert)], None, 1.0, noise=noise)
    history = [{"iter": 0, "source": "expert+noise", **st, "samples": len(X)}]
    log(f"[{kind}] expert data: {len(X)} samples, {st}")
    for it in range(1, n_iter + 1):
        losses = fit(policy, np.array(X, np.float32), np.array(Y, np.float32), epochs=iter_epochs, seed=it)
        beta = 0.5 ** it
        Xn, Yn, st = collect(arm, kind, [next(seeds) for _ in range(n_per_iter)], policy, beta, noise=0.0)
        X += Xn
        Y += Yn
        history.append({"iter": it, "beta": beta, "final_loss": losses[-1], **st, "samples": len(X)})
        log(f"[{kind}] iter {it}: loss {losses[-1]:.4f}, rollouts (beta={beta:.2f}) success {st['success']}/{st['episodes']}, "
            f"samples {len(X)}, {time.time() - t0:.0f}s")
    losses = fit(policy, np.array(X, np.float32), np.array(Y, np.float32), epochs=epochs, seed=99)
    history.append({"iter": "final", "final_loss": losses[-1], "samples": len(X)})
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": kind, "n_obs": n_obs, "n_act": n_act, "source": source, "state_dict": policy.state_dict()},
               out_dir / f"policy_{kind}.pt")
    meta = {"kind": kind, "trained_on": source, "train_seed_range": [0, 10_000], "history": history,
            "config": {"n_expert": n_expert, "n_iter": n_iter, "n_per_iter": n_per_iter, "epochs": epochs,
                       "iter_epochs": iter_epochs, "dart_noise": noise, "hidden": 256, "layers": 3},
            "wall_s": round(time.time() - t0, 1), "torch": torch.__version__}
    (out_dir / f"policy_{kind}.json").write_text(json.dumps(meta, indent=2))
    return policy, meta


def adapt_policy(arm: ArmAdapter, base: Policy, seeds, n_iter: int = 3, n_per_iter: int = 25, epochs: int = 60,
                 log=print) -> tuple[Policy, list]:
    """Fine-tune a transferred task-space policy on a new body with DAgger: the transferred policy drives,
    the skill layer (parameterised by this arm's own carry height etc.) labels every visited state.
    Used when trials show the transferred policy's assumptions don't hold for the new body."""
    policy = Policy(base.net[0].in_features, base.n_act)
    policy.load_state_dict(base.state_dict())
    seeds = iter(seeds)
    X, Y, history = [], [], []
    t0 = time.time()
    for it in range(n_iter):
        beta = 0.5 ** (it + 1)
        Xn, Yn, st = collect(arm, "task", [next(seeds) for _ in range(n_per_iter)], policy, beta, noise=0.0)
        X += Xn
        Y += Yn
        losses = fit(policy, np.array(X, np.float32), np.array(Y, np.float32), epochs=epochs, lr=3e-4, seed=it,
                     renorm=False)
        history.append({"iter": it + 1, "beta": beta, **st, "samples": len(X), "loss": losses[-1]})
        log(f"  adapt iter {it + 1}: rollouts (beta={beta:.2f}) success {st['success']}/{st['episodes']}, "
            f"{len(X)} samples, loss {losses[-1]:.4f}, {time.time() - t0:.0f}s")
    return policy, history


def load_policy(path: Path) -> Policy:
    ck = torch.load(path, weights_only=True)
    p = Policy(ck["n_obs"], ck["n_act"])
    p.load_state_dict(ck["state_dict"])
    p.eval()
    return p
