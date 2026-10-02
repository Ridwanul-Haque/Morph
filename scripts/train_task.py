"""Learn a task from scratch for any robot (MuJoCo Warp + Brax PPO on the GPU).

Reads a task file written by Morph (morph/learn_task.py): robot, scene, and the numeric target that the
goal checklist boils down to. Reward = the checklist made continuous: the object's progress towards the
target, closeness, the robot reaching the object, the object staying upright (and turning to the goal
heading if asked), a bonus while every condition holds, smooth actions; robots with a floating base also
get a small reward for staying up and the episode ends if they fall.

    MORPH_GPU_MEM=0.7 .venv-rl/bin/python scripts/train_task.py runs/learned/<name>/task.json --steps 150_000_000
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

# Contact-rich scenes need GPU memory outside JAX's pool (MuJoCo Warp's own buffers): leave it room.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", os.environ.get("MORPH_TASK_GPU_MEM", "0.35"))
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import numpy as np  # noqa: E402
from ml_collections import config_dict  # noqa: E402
from mujoco import mjx  # noqa: E402

from train_go1_walk import export_numpy  # noqa: E402  (also installs the Brax/JAX and Warp shims)
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402
from mujoco_playground import wrapper  # noqa: E402
from mujoco_playground._src import mjx_env  # noqa: E402
from mujoco_playground.config import locomotion_params  # noqa: E402

from morph.goal_terms import HOLD_TICKS, condition_violation, is_generic, is_temporal, parse, violation  # noqa: E402
from morph.task_model import CTRL_DT, SIM_DT, build_task_model, indices, load_task, measure, next_ctrl, obs_size, observe  # noqa: E402


RANDOMIZE = {  # (low, high) multipliers; "light" halves each spread
    "floor_friction": (0.6, 1.3), "object_friction": (0.6, 1.4), "object_mass": (0.6, 1.6),
    "motor_strength": (0.8, 1.2), "obs_noise": 0.01, "delay_prob": 0.5}


def ranges(level: str) -> dict:
    if level == "off":
        return {}
    k = 1.0 if level == "full" else 0.5
    return {n: (1 - k * (1 - v[0]), 1 + k * (v[1] - 1)) if isinstance(v, tuple) else v * k for n, v in RANDOMIZE.items()}


def domain_randomize(env: "TaskLearn", level: str):
    """Brax randomization_fn: one physics variant per practice copy."""
    r = ranges(level)
    m = env.mj_model
    floor = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 and m.geom_type[g] == 0])
    obj = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == env.ix.block])
    blk = env.ix.block

    def fn(model, rng):
        @jax.vmap
        def one(key):
            k = jax.random.split(key, 4)
            u = lambda kk, lohi, shape=(): jax.random.uniform(kk, shape, minval=lohi[0], maxval=lohi[1])
            fr = model.geom_friction
            fr = fr.at[floor, 0].multiply(u(k[0], r["floor_friction"]))
            fr = fr.at[obj, 0].multiply(u(k[1], r["object_friction"]))
            s = u(k[2], r["object_mass"])
            mass = model.body_mass.at[blk].multiply(s)
            inertia = model.body_inertia.at[blk].multiply(s)
            g = u(k[3], r["motor_strength"], (model.nu,))
            gain = model.actuator_gainprm.at[:, 0].multiply(g)
            bias = model.actuator_biasprm.at[:, 1].multiply(g).at[:, 2].multiply(g)
            force = model.actuator_forcerange * g[:, None]
            return fr, mass, inertia, gain, bias, force

        fr, mass, inertia, gain, bias, force = one(rng)
        names = ("geom_friction", "body_mass", "body_inertia", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange")
        in_axes = jax.tree_util.tree_map(lambda x: None, model).tree_replace({n: 0 for n in names})
        return model.tree_replace(dict(zip(names, (fr, mass, inertia, gain, bias, force)))), in_axes

    return fn


class TaskLearn(mjx_env.MjxEnv):
    def __init__(self, task: dict, envs: int = 2048, randomize: str = "off"):
        cfg = config_dict.create(ctrl_dt=CTRL_DT, sim_dt=SIM_DT, episode_length=int(task["episode_s"] / CTRL_DT),
                                 action_repeat=1, impl="warp", naconmax=24 * envs, njmax=160)
        super().__init__(cfg, None)
        self.task = task
        self._mj_model = build_task_model(task)
        self._mjx_model = mjx.put_model(self._mj_model, impl=cfg.impl)
        self.ix = indices(self._mj_model, task)
        t = task["target"]
        self.target = (t["x"], t["y"], t["z"], t["yaw"])
        self.tol, self.yaw_tol = t["pos_tol"], t["yaw_tol"]
        self.heading, self.upright = t["heading"], t["upright"]
        # Push-type goals: "reaching" means getting behind the object, on the far side from the target
        # (reaching its middle taught the UR5e to hover over the parcel without moving it).
        self.has_place = t.get("has_place", True)
        self.push = self.has_place and not t.get("lift", False)
        self.r_obj = float(np.hypot(task["block"]["half_x"], task["block"]["half_y"]))
        # conditions that aren't a place (lying on its side, lifted, away from X, not in X, a formula): measured
        # by goal_terms, the same code that checks the result
        self.generic = [c for c in task["conditions"] if is_generic(c) and not is_temporal(c)]
        # conditions about a moment during the task: when it lets go (throwing), the whole time, at some point
        self.temporal = [c for c in task["conditions"] if is_temporal(c)]
        self.fx = task.get("fixtures", {})
        # the steps to go about it (skill_steps): reward for reaching each in order
        self.steps = [(parse(st["done_when"], self.fx), parse(st["keep"], self.fx) if st.get("keep") else None)
                      for st in task.get("steps", [])]
        b = task["block"]
        self.half = (b["half_x"], b["half_y"], b["half_z"])
        self.nu = self._mj_model.nu
        self.scale = task["action_scale"]
        q = self._mj_model.qpos0.copy()
        q[self.ix.home_adr] = self.ix.home_val
        bx, by, byaw = task["start"]
        q[self.ix.block_q:self.ix.block_q + 7] = [bx, by, task["block"]["half_z"], np.cos(byaw / 2), 0, 0, np.sin(byaw / 2)]
        if self.ix.base >= 0:
            mx, my, myaw = task["mount"]
            z = task.get("base_z", 0.4)
            q[self.ix.base_q:self.ix.base_q + 7] = [mx, my, z, np.cos(myaw / 2), 0, 0, np.sin(myaw / 2)]
        self._q_init = jp.array(q)
        self._ctrl0 = jp.array(np.clip(self.ix.q0, self.ix.act_lo, self.ix.act_hi))
        self._lo, self._hi = jp.array(self.ix.act_lo), jp.array(self.ix.act_hi)
        self.rand = ranges(randomize)

    def _dist(self, data, info=None):
        """How far from done, in metres: distance to the place (if the goal has one) plus how unmet the
        other conditions are (goal_terms units, 1 ~ 30 cm). Returns (distance, updated tracking of the
        conditions about a moment during the task)."""
        p = data.xpos[self.ix.block]
        tx, ty, tz, _ = self.target
        d = jp.hypot(p[0] - tx, p[1] - ty) + jp.abs(p[2] - tz) if self.has_place else jp.zeros(())
        if not self.generic and not self.temporal:
            return d, {}
        v = measure(jp, data.xpos, data.xmat, data.qvel, self.ix, self.task)
        unmet = sum((condition_violation(jp, c, v, self.fx, self.half) for c in self.generic), jp.zeros(()))
        track = {}
        if self.temporal:
            x = jp.stack([condition_violation(jp, {**c, "when": "end"}, v, self.fx, self.half) for c in self.temporal])
            held = v["held"]
            if info is None:  # the start of an episode
                tv, released, let_go = x, jp.zeros(()), jp.zeros(())
            else:
                tv, released = info["tv"], info["released"]
                let_go = ((info["was_held"] > 0.5) & (held < 0.5)).astype(jp.float32)
            released = jp.maximum(released, let_go)
            terms = []
            for k, c in enumerate(self.temporal):
                if c["when"] == "release":  # the moment it lets go counts; until then, as if it let go now (+ a little)
                    tv = tv.at[k].set(jp.where(let_go > 0.5, x[k], tv[k]))
                    terms.append(jp.where(released > 0.5, tv[k], x[k] + 0.1))
                elif c["when"] == "always":  # the worst moment so far
                    tv = tv.at[k].set(jp.maximum(tv[k], x[k]))
                    terms.append(tv[k])
                else:  # ever: the best moment so far
                    tv = tv.at[k].set(jp.minimum(tv[k], x[k]))
                    terms.append(tv[k])
            unmet = unmet + sum(terms)
            track = {"tv": tv, "released": released, "was_held": held}
        return d + 0.3 * unmet, track

    def _steps(self, data, step, streak):
        """(next step index, potential, keep-rule violation, streaks). The potential grows by 1 per step reached
        and by up to 1 as the current step gets closer to done, so the reward pulls towards the next step."""
        n = len(self.steps)
        if not n:
            return step, jp.zeros(()), jp.zeros(()), streak
        v = measure(jp, data.xpos, data.xmat, data.qvel, self.ix, self.task)
        done = jp.stack([violation(jp, d, v) for d, _ in self.steps])
        keep = jp.stack([violation(jp, k, v) if k is not None else jp.zeros(()) for _, k in self.steps])
        i = jp.minimum(step, n - 1)
        j = jp.minimum(step + 1, n - 1)
        cur = jp.where(step < n, done[i], 0.0)
        nxt = jp.where(step + 1 < n, done[j], 1.0)
        # in order, one at a time, and only once it has held for HOLD_TICKS (no flukes from a bounce); a step
        # that's too strict can be skipped if the one after it holds that long
        s0 = jp.where((step < n) & (cur <= 1e-3), streak[0] + 1, 0)
        s1 = jp.where((step + 1 < n) & (cur > 1e-3) & (nxt <= 1e-3), streak[1] + 1, 0)
        adv = jp.where(s0 >= HOLD_TICKS, 1, jp.where(s1 >= HOLD_TICKS, 2, 0))
        step = jp.minimum(step + adv, n)
        streak = jp.where(adv > 0, jp.zeros(2, jp.int32), jp.stack([s0, s1]).astype(jp.int32))
        i = jp.minimum(step, n - 1)
        cur = jp.where(step < n, done[i], 0.0)
        phi = step.astype(jp.float32) + jp.where(step < n, 1.0 - jp.tanh(3.0 * cur), 0.0)
        return step, phi, jp.where(step < n, keep[i], 0.0), streak

    def reset(self, rng):
        rng, k1, k2, k3 = jax.random.split(rng, 4)
        q = self._q_init
        dxy = jax.random.uniform(k1, (2,), minval=-0.02, maxval=0.02)
        q = q.at[self.ix.block_q:self.ix.block_q + 2].add(dxy)
        dy = jax.random.uniform(k2, (), minval=-0.17, maxval=0.17)
        yaw0 = self.task["start"][2] + dy
        q = q.at[self.ix.block_q + 3:self.ix.block_q + 7].set(jp.array([jp.cos(yaw0 / 2), 0.0, 0.0, jp.sin(yaw0 / 2)]))
        data = mjx_env.make_data(self._mj_model, qpos=q, qvel=jp.zeros(self._mjx_model.nv), ctrl=self._ctrl0,
                                 impl=self._mjx_model.impl.value, naconmax=self._config.naconmax, njmax=self._config.njmax)
        data = mjx.forward(self._mjx_model, data)
        delay = (jax.random.uniform(k3) < self.rand["delay_prob"]).astype(jp.float32) if self.rand else jp.zeros(())
        d0, track = self._dist(data)
        info = {"rng": rng, "last_act": jp.zeros(self.nu), "prev_d": d0, "ctrl": self._ctrl0, "delay": delay,
                "step": jp.zeros((), jp.int32), "streak": jp.zeros(2, jp.int32), **track}
        info["phi"] = self._steps(data, info["step"], info["streak"])[1]
        metrics = {k: jp.zeros(()) for k in ("progress", "close", "reach", "success", "dist", "steps_done")}
        obs = self._obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def _obs(self, data, info):
        o = observe(jp, data.qvel, data.actuator_length, data.actuator_velocity, data.xpos, data.xmat, info["last_act"],
                    self.ix, self.target)
        if not self.rand:
            return {"state": o, "privileged_state": o}
        info["rng"], k = jax.random.split(info["rng"])
        return {"state": o + self.rand["obs_noise"] * jax.random.normal(k, o.shape), "privileged_state": o}

    def step(self, state, action):
        # with a delay, the robot acts on the previous command (like a real control loop that lags a step)
        applied = action + state.info["delay"] * (state.info["last_act"] - action)
        ctrl = next_ctrl(jp, state.info["ctrl"], applied, self._ctrl0, self.scale, self._lo, self._hi)
        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)
        ix, info = self.ix, state.info
        d, track = self._dist(data, info)
        info.update(track)
        bp = data.xpos[ix.block]
        R = data.xmat[ix.block].reshape(3, 3)
        tilt = jp.arccos(jp.clip(R[2, 2], -1.0, 1.0))
        yaw_err = jp.abs(jp.arctan2(jp.sin(jp.arctan2(R[1, 0], R[0, 0]) - self.target[3]),
                                    jp.cos(jp.arctan2(R[1, 0], R[0, 0]) - self.target[3])))
        if self.push:
            to_t = jp.array([self.target[0], self.target[1]]) - bp[:2]
            u = to_t / (jp.linalg.norm(to_t) + 1e-6)
            aim = jp.concatenate([bp[:2] - u * (self.r_obj + 0.03), bp[2:3]])
        else:
            aim = bp
        reach = jp.min(jp.linalg.norm(data.xpos[ix.tips] - aim, axis=1))
        progress = (info["prev_d"] - d) / self.dt
        close = jp.exp(-d / 0.03)
        place_ok = d < (self.tol if self.has_place else 1e-3)  # d already includes the unmet conditions
        if any(c["when"] == "release" for c in self.temporal):
            place_ok = place_ok & (info["released"] > 0.5)
        ok = place_ok & ((yaw_err < self.yaw_tol) | (not self.heading)) & ((tilt < 0.17) | (not self.upright))
        # smooth pulls that work from far away (the tool starts ~0.5 m from the object), sharper ones up close
        r = (2.0 * jp.clip(progress, -1.0, 1.0) + 1.0 * (1 - jp.tanh(d / 0.2)) + 2.0 * close
             + 0.5 * (1 - jp.tanh(reach / 0.25)) * (1 - close) + 0.5 * jp.exp(-reach / 0.05) * (1 - close)
             + (0.5 * jp.exp(-yaw_err ** 2 / 0.3) * close if self.heading else 0.0)
             - (1.0 * tilt ** 2 if self.upright else 0.0) + 3.0 * ok
             - 0.02 * jp.sum(jp.square(action - info["last_act"]))
             - 1.0 * jp.maximum(jp.linalg.norm(data.qvel[ix.block_v:ix.block_v + 3]) - 0.5, 0.0))  # no flinging
        done = jp.zeros(())
        if ix.base >= 0:
            up = data.xmat[ix.base].reshape(3, 3)[2, 2]
            fell = (up < 0.4) | (data.xpos[ix.base][2] < 0.4 * self.task.get("base_z", 0.4))
            r = r + 0.2 - 5.0 * fell
            done = fell.astype(jp.float32)
        shaped = jp.zeros(())
        if self.steps:  # the steps a person would take: reward reaching each, in order; mind the keep rules
            step, phi, keep, info["streak"] = self._steps(data, info["step"], info["streak"])
            shaped = 0.5 * (phi - info["phi"])  # a one-off bonus per step reached (not a rate: not scaled by dt)
            r = r + 1.0 * step / len(self.steps) - 1.0 * jp.minimum(keep, 2.0)
            info["step"], info["phi"] = step, phi
        info["last_act"], info["prev_d"], info["ctrl"] = action, d, ctrl
        z = lambda x: jp.where(jp.isnan(x), 0.0, x)  # a blown-up copy shouldn't turn the logged averages into NaN
        state.metrics.update(progress=z(progress), close=z(close), reach=z(reach), success=ok.astype(jp.float32), dist=z(d),
                             steps_done=info["step"].astype(jp.float32))
        obs = self._obs(data, info)
        # a copy whose physics blew up (NaN) ends now and counts for nothing, as in MuJoCo Playground's own
        # environments; otherwise one bad copy turns the whole batch's numbers into NaN
        bad = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any() | jp.isnan(r) | jp.isnan(shaped)
        reward = jp.where(bad, 0.0, r * self.dt + shaped)
        done = jp.where(bad, 1.0, done)
        obs = jax.tree_util.tree_map(lambda o: jp.where(jp.isnan(o), 0.0, o), obs)
        return state.replace(data=data, obs=obs, reward=reward, done=done)

    @property
    def xml_path(self):
        return self.task["robot_xml"]

    @property
    def action_size(self):
        return self.nu

    @property
    def mj_model(self):
        return self._mj_model

    @property
    def mjx_model(self):
        return self._mjx_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--steps", type=int, default=60_000_000)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--evals", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomize", choices=("off", "light", "full"), default="light",
                    help="vary friction, object mass, motor strength, delay and sensor noise between practice copies")
    a = ap.parse_args()
    task_path = Path(a.task)
    task = load_task(task_path)
    out = task_path.parent
    env, eval_env = TaskLearn(task, a.envs, a.randomize), TaskLearn(task, 128, a.randomize)
    print(json.dumps({"obs": obs_size(env.ix), "actions": env.nu, "tips": [env.mj_model.body(int(b)).name for b in env.ix.tips]}),
          flush=True)
    p = locomotion_params.brax_ppo_config("Go1JoystickFlatTerrain")
    p.num_timesteps, p.num_envs, p.num_evals = a.steps, a.envs, a.evals
    p.episode_length = int(task["episode_s"] / CTRL_DT)
    net = dict(p.network_factory)
    kw = {k: v for k, v in p.items() if k != "network_factory"}
    log, t0 = [], time.time()

    def progress(step, metrics):
        rec = {"step": int(step), "wall_s": round(time.time() - t0, 1),
               "reward": float(metrics.get("eval/episode_reward", float("nan"))),
               "success": float(metrics.get("eval/episode_success", float("nan"))),
               "dist": float(metrics.get("eval/episode_dist", float("nan"))),
               # how many of the steps it gets through, on average by the end of an episode (the metric is
               # summed over the episode, so the running count's mean is a lower bound)
               "steps_done": float(metrics.get("eval/episode_steps_done", 0.0)) / p.episode_length}
        log.append(rec)
        print(json.dumps(rec), flush=True)

    ckpts = out / "checkpoints"
    ckpts.mkdir(exist_ok=True)

    def checkpoint(step, _make_policy, params):  # the app plays each one so you can watch it improve
        path = ckpts / f"step_{int(step):012d}.npz"
        export_numpy(params, path)
        print(json.dumps({"checkpoint": str(path), "step": int(step)}), flush=True)

    rand = domain_randomize(env, a.randomize) if a.randomize != "off" else None
    train = functools.partial(ppo.train, **kw, seed=a.seed, progress_fn=progress, policy_params_fn=checkpoint,
                              randomization_fn=rand,
                              network_factory=functools.partial(ppo_networks.make_ppo_networks, **net))
    _, params, _ = train(environment=env, eval_env=eval_env, wrap_env_fn=wrapper.wrap_for_brax_training)
    shapes = export_numpy(params, out / "policy.npz")
    meta = {"task": str(task_path), "steps": a.steps, "num_envs": a.envs, "randomize": a.randomize,
            "randomize_ranges": ranges(a.randomize), "wall_s": round(time.time() - t0, 1),
            "network": shapes, "curve": log, "obs_size": obs_size(env.ix), "actions": env.nu}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "curve"}), flush=True)


if __name__ == "__main__":
    main()
