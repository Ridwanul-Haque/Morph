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

from morph.goal_terms import (BODY_FUNCS, FLICKER_TICKS, HOLD_TICKS, condition_violation, is_generic, is_temporal,  # noqa: E402
                              normalize, parse, term_value, uses, violation)
from morph.reward_parts import weights as reward_weights  # noqa: E402
from morph.task_model import (CTRL_DT, SIM_DT, Residual, apply_action, build_task_model, indices, load_task, measure,  # noqa: E402
                              next_ctrl,  # noqa: E402
                              obs_size, observe, soft)  # noqa: E402


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
    def __init__(self, task: dict, envs: int = 2048, randomize: str = "off", practice_starts: bool = True,
                 eased: bool = True):
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
        # early termination (DeepMimic, Peng et al. 2018): once a "the whole time" condition is broken the try can't
        # succeed, so it ends there instead of practising on in a state that no longer counts (the thumb on the pen,
        # the pen on the floor). Only conditions that held when the try began can end it: a stand-in measure that
        # reads differently from the judge's at the start mustn't end every try at once
        self.stop_rules = [c for c in self.temporal if c["when"] == "always"] if task.get("early_stop", True) else []
        # an easier version of the goal first (task["ease"], set by learn_task._ease between rounds): each condition may
        # miss by up to its slack (goal_terms units), scaled per try by a random share, so practice covers the range
        # from the eased goal to the real one (as automatic domain randomization widens what it practises on only as
        # the policy copes). The evaluation, and every judgement of success, is on the real goal
        slack = (task.get("ease") or {}).get("slack") or {} if eased else {}
        idx = {id(c): i for i, c in enumerate(task["conditions"])}
        at = lambda cs: jp.array([float(slack.get(str(idx[id(c)]), 0.0)) for c in cs])  # noqa: E731
        self._slack_g, self._slack_t, self._slack_s = at(self.generic), at(self.temporal), at(self.stop_rules)
        self._eased = bool(slack) and any(float(v) > 0 for v in slack.values())
        # every condition about the robot's own body (joint_frac, part_z: a fist, a raised arm), none about the object
        self.body_goal = bool(task["conditions"]) and all(
            c.get("kind") == "expression" and c.get("expr") and uses(normalize(c["expr"])) & BODY_FUNCS
            for c in task["conditions"])
        self.fx = task.get("fixtures", {})
        # the grip Morph's grasp planning found holds best (contact_plan.plan_grasp), measured on the robot: reaching
        # means getting the point between the fingers there, fingers closing across the planned sides, hand coming in
        # the planned way. Without it, the nearest fingertip is pulled to the object's centre, and on a flat object the
        # learner pokes a fingertip down beside or under it instead of closing both fingers across it from above.
        g = task.get("grasp") if not self.push and not self.body_goal else None
        self.grasp = None
        if g and g.get("fingers") and g.get("close") is not None:
            try:
                body = lambda n: self._mj_model.body(n).id  # noqa: E731
                self.grasp = {"body": body(g["body"]), "offset": jp.array(g["offset"]), "axis": jp.array(g["axis"]),
                              "fa": body(g["fingers"][0][0]), "oa": jp.array(g["fingers"][0][1]),
                              "fb": body(g["fingers"][1][0]), "ob": jp.array(g["fingers"][1][1]),
                              "point": jp.array(g["point"]), "close": jp.array(g["close"]),
                              "approach": jp.array(g["approach"])}
            except KeyError:  # a different robot model than the plan was made on
                self.grasp = None
        # the steps to go about it (skill_steps): reward for reaching each in order
        self.steps = [(parse(st["done_when"], self.fx), parse(st["keep"], self.fx) if st.get("keep") else None)
                      for st in task.get("steps", [])]
        # how much each part of the reward counts (reward_parts: 1 = Morph's default; a suggestion can change it)
        self.w = reward_weights(task)
        self.parts = ["toward", "near", "success", "reach", "smooth", "fling"] + (["heading"] if self.heading else []) \
            + (["upright"] if self.upright else []) + (["standing"] if self.ix.base >= 0 else []) \
            + (["steps"] if task.get("steps") else [])  # + "copy" once the demonstration is loaded (below)
        # extra shaping terms Nemotron wrote (Eureka-style), each logged on its own so it can see what they did
        self.terms = [(parse(t["formula"], self.fx), float(t["weight"])) for t in task.get("reward_terms", [])]
        b = task["block"]
        self.half = (b["half_x"], b["half_y"], b["half_z"])
        self.nu = self._mj_model.nu
        self.scale = jp.array(self.ix.act_scale)  # per actuator (task_model.action_units)
        self._step = jp.array(self.ix.act_step)
        q = self._mj_model.qpos0.copy()
        q[self.ix.home_adr] = self.ix.home_val
        bx, by, byaw = task["start"]
        q[self.ix.block_q:self.ix.block_q + 7] = [bx, by, task["block"]["half_z"], np.cos(byaw / 2), 0, 0, np.sin(byaw / 2)]
        if self.ix.base >= 0:
            mx, my, myaw = task["mount"]
            z = task.get("base_z", 0.4)
            q[self.ix.base_q:self.ix.base_q + 7] = [mx, my, z, np.cos(myaw / 2), 0, 0, np.sin(myaw / 2)]
        self._q_init = jp.array(q)
        self._in_hand = bool(task.get("start_state"))
        self._hand_starts = None
        if self._in_hand:  # every try starts with the object in the hand (morph/in_hand.py), in one of its poses
            self._q_init = jp.array(task["start_state"]["qpos"])
            poses = task.get("start_states") or [task["start_state"]]
            self._hand_starts = jp.array([st["qpos"] for st in poses])
        # practice from where it got closest last round (diagnose saves them): a share of resets start there
        st = task.get("starts")
        self._starts, self._start_share = None, 0.0
        if practice_starts and st and (Path(task.get("_dir", ".")) / st["file"]).exists():
            z = np.load(Path(task["_dir"]) / st["file"])
            self._starts = jp.array(z["qpos"])
            # the commands at that moment (e.g. the gripper shut on it), or NaN: hold the pose it's in
            self._start_ctrl = jp.array(z["ctrl"]) if "ctrl" in z else jp.full((len(z["qpos"]), self._mj_model.nu), jp.nan)
            # how far through the recipe each start is (for corrections on top of it), and which ones this round
            # may use: easy-first practice starts near the end of the recipe and works back (learn_in_rounds)
            self._start_t = jp.array(z["t"], dtype=jp.int32) if "t" in z else jp.zeros(len(z["qpos"]), jp.int32)
            self._start_share = float(st.get("share", 0.3))
            lo = int(st.get("from_t", 0))
            self._start_ok = self._start_t >= lo if "t" in z and lo > 0 and bool(np.any(z["t"] >= lo)) else None
        self._ctrl0 = jp.array(np.clip(self.ix.q0, self.ix.act_lo, self.ix.act_hi))
        self._lo, self._hi = jp.array(self.ix.act_lo), jp.array(self.ix.act_hi)
        self.rand = ranges(randomize)
        # a demonstration (a program that worked): in situations like the ones it went through, reward acting the
        # way it did, so practice refines the copied skill instead of drifting away from it
        self._demo = None
        if task.get("demo") and (Path(task.get("_dir", ".")) / task["demo"]).exists():
            z = np.load(Path(task["_dir"]) / task["demo"])
            o, act = z["obs"].astype(np.float32), np.clip(z["act"], -1, 1).astype(np.float32)
            mu, sd = o.mean(0), o.std(0) + 1e-3
            self._demo = (jp.array(mu), jp.array(sd), jp.array((o - mu) / sd), jp.array(act))
            self.parts.append("copy")
        # corrections on top of the recipe: the action is the recipe's at this moment plus a learned correction
        self.residual = Residual.of(task, task.get("_dir", "."))
        # asymmetric actor-critic (Pinto et al. 2018): the critic also sees what the robot can't (how well it really
        # grips, how far through the steps it is, this copy's true parcel weight and friction); the policy is unchanged
        # (version 2 also tells the critic how eased this try's goal is)
        self.n_extra = {True: 9, 1: 9, 2: 10}.get(task.get("critic_extras"), 0)
        self._obj_geom = int(np.flatnonzero(np.asarray(self._mj_model.geom_bodyid) == self.ix.block)[0])
        self._mass0 = float(self._mj_model.body_mass[self.ix.block])
        self._fric0 = float(self._mj_model.geom_friction[self._obj_geom, 0])

    def _dist(self, data, info=None, u=None):
        """How far from done, in metres: distance to the place (if the goal has one) plus how unmet the
        other conditions are (goal_terms units, 1 ~ 30 cm). Returns (distance, updated tracking of the
        conditions about a moment during the task)."""
        p = data.xpos[self.ix.block]
        tx, ty, tz, _ = self.target
        d = jp.hypot(p[0] - tx, p[1] - ty) + jp.abs(p[2] - tz) if self.has_place else jp.zeros(())
        if not self.generic and not self.temporal:
            return d, {}
        u = info["ease_u"] if u is None and info is not None and "ease_u" in info else (u if u is not None else 0.0)
        # graded "held" (grip) here: this is the learner's sense of how far it is, not the judgement of success
        v = soft(measure(jp, data.xpos, data.xmat, data.qvel, self.ix, self.task, ctrl=data.ctrl, qpos=data.qpos))
        unmet = sum((jp.maximum(condition_violation(jp, c, v, self.fx, self.half) - u * self._slack_g[k], 0.0)
                     for k, c in enumerate(self.generic)), jp.zeros(()))
        track = {}
        if self.temporal:
            x = jp.stack([condition_violation(jp, {**c, "when": "end"}, v, self.fx, self.half) for c in self.temporal])
            x = jp.maximum(x - u * self._slack_t, 0.0)
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
                elif c["when"] == "last":  # keep it up: practise holding it now (the check takes the last seconds)
                    terms.append(x[k])
                else:  # ever: the best moment so far
                    tv = tv.at[k].set(jp.minimum(tv[k], x[k]))
                    terms.append(tv[k])
            unmet = unmet + sum(terms)
            track = {"tv": tv, "released": released, "was_held": held}
        return d + 0.3 * unmet, track

    def _broken(self, data, u=0.0):
        """Per "the whole time" condition: is it broken now (strict, as judged; by more than its slack when eased)."""
        v = measure(jp, data.xpos, data.xmat, data.qvel, self.ix, self.task, ctrl=data.ctrl, qpos=data.qpos)
        return jp.stack([condition_violation(jp, {**c, "when": "end"}, v, self.fx, self.half) > u * self._slack_s[k] + 1e-6
                         for k, c in enumerate(self.stop_rules)])

    def _grasp_aim(self, data, bp, R):
        """(distance from the planned grip point, 0..1 how well the hand matches the planned grip there). The fingers
        must close along the planned axis (either way round) and the hand point the way it came in; a hand that has
        gone past the grip point (under the object, for a grip from above) gets no credit for being close."""
        g = self.grasp
        Rw = data.xmat[g["body"]].reshape(3, 3)
        hand = data.xpos[g["body"]] + Rw @ g["offset"]
        pointing = Rw @ g["axis"]
        fa = data.xpos[g["fa"]] + data.xmat[g["fa"]].reshape(3, 3) @ g["oa"]
        fb = data.xpos[g["fb"]] + data.xmat[g["fb"]].reshape(3, 3) @ g["ob"]
        fingers = (fb - fa) / (jp.linalg.norm(fb - fa) + 1e-6)
        target, close_w, come_in = bp + R @ g["point"], R @ g["close"], R @ g["approach"]
        e = hand - target
        dist = jp.linalg.norm(e)
        past = jp.maximum(e @ come_in, 0.0)  # how far beyond the grip point, along the way the hand comes in
        orient = 0.5 * jp.abs(fingers @ close_w) + 0.5 * jp.maximum(pointing @ come_in, 0.0)
        return dist, (0.3 + 0.7 * orient) * jp.exp(-past / 0.01)

    def _steps(self, data, step, streak):
        """(next step index, potential, keep-rule violation, streaks). The potential grows by 1 per step reached
        and by up to 1 as the current step gets closer to done, so the reward pulls towards the next step."""
        n = len(self.steps)
        if not n:
            return step, jp.zeros(()), jp.zeros(()), streak
        v = measure(jp, data.xpos, data.xmat, data.qvel, self.ix, self.task, ctrl=data.ctrl, qpos=data.qpos)
        vs = soft(v)
        done = jp.stack([violation(jp, d, v) for d, _ in self.steps])  # strict: has it really happened
        near = jp.stack([violation(jp, d, vs) for d, _ in self.steps])  # graded: how close it is getting
        keep = jp.stack([violation(jp, k, vs) if k is not None else jp.zeros(()) for _, k in self.steps])
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
        cur = jp.where(step < n, near[i], 0.0)
        phi = step.astype(jp.float32) + jp.where(step < n, 1.0 - jp.tanh(3.0 * cur), 0.0)
        return step, phi, jp.where(step < n, keep[i], 0.0), streak

    def reset(self, rng):
        rng, k1, k2, k3 = jax.random.split(rng, 4)
        q = self._q_init
        if self._hand_starts is not None:
            rng, kh = jax.random.split(rng)
            q = self._hand_starts[jax.random.randint(kh, (), 0, self._hand_starts.shape[0])]
        if not self._in_hand:  # vary where it starts (in the hand it stays where the grip put it)
            dxy = jax.random.uniform(k1, (2,), minval=-0.02, maxval=0.02)
            q = q.at[self.ix.block_q:self.ix.block_q + 2].add(dxy)
            dy = jax.random.uniform(k2, (), minval=-0.17, maxval=0.17)
            yaw0 = self.task["start"][2] + dy
            q = q.at[self.ix.block_q + 3:self.ix.block_q + 7].set(jp.array([jp.cos(yaw0 / 2), 0.0, 0.0, jp.sin(yaw0 / 2)]))
        ctrl0 = self._ctrl0
        if self._starts is not None:
            rng, k4, k5 = jax.random.split(rng, 3)
            if self._start_ok is None:
                i_start = jax.random.randint(k4, (), 0, self._starts.shape[0])
            else:  # only the starts this round allows (late in the recipe first)
                i_start = jax.random.categorical(k4, jp.where(self._start_ok, 0.0, -jp.inf))
            pick = self._starts[i_start]
            use = jax.random.uniform(k5) < self._start_share
            q = jp.where(use, pick, q)
        data = mjx_env.make_data(self._mj_model, qpos=q, qvel=jp.zeros(self._mjx_model.nv), ctrl=ctrl0,
                                 impl=self._mjx_model.impl.value, naconmax=self._config.naconmax, njmax=self._config.njmax)
        data = mjx.forward(self._mjx_model, data)
        if self._starts is not None:  # hold the pose it starts in (position motors aim where the joints are)
            held_ctrl = jp.clip(jp.where(self.ix.pos_mask > 0, data.actuator_length, 0.0), self._lo, self._hi)
            saved = self._start_ctrl[i_start]
            held_ctrl = jp.where(jp.isnan(saved), held_ctrl, saved)
            ctrl0 = jp.where(use, held_ctrl, self._ctrl0)
            data = data.replace(ctrl=ctrl0)
        delay = (jax.random.uniform(k3) < self.rand["delay_prob"]).astype(jp.float32) if self.rand else jp.zeros(())
        u = jp.zeros(())
        if self._eased:  # how eased this try's goal is: anywhere from the real goal (0) to the eased one (1)
            rng, ke = jax.random.split(rng)
            u = jax.random.uniform(ke)
        d0, track = self._dist(data, u=u)
        t0 = jp.where(use, self._start_t[i_start], 0) if self._starts is not None else jp.zeros((), jp.int32)
        info = {"rng": rng, "last_act": jp.zeros(self.nu), "prev_d": d0, "ctrl": ctrl0, "delay": delay,
                "step": jp.zeros((), jp.int32), "streak": jp.zeros(2, jp.int32), "t": t0.astype(jp.int32), "ease_u": u,
                **track}
        info["phi"] = self._steps(data, info["step"], info["streak"])[1]
        if self.stop_rules:
            info["armed"] = ~self._broken(data, u)
            info["broke"] = jp.zeros(len(self.stop_rules), jp.int32)
        metrics = {k: jp.zeros(()) for k in ("progress", "close", "reach", "grasp_aim", "success", "dist", "steps_done",
                                             "ended_early")}
        metrics |= {f"term_{k}": jp.zeros(()) for k in range(len(self.terms))}
        metrics |= {f"termw_{k}": jp.zeros(()) for k in range(len(self.terms))}
        metrics |= {f"part_{k}": jp.zeros(()) for k in self.parts}
        obs = self._obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def _obs(self, data, info):
        o = observe(jp, data.qvel, data.actuator_length, data.actuator_velocity, data.xpos, data.xmat, info["last_act"],
                    self.ix, self.target)
        if self.residual is not None:
            o = jp.concatenate([o, self.residual.phase(jp, info["t"])])
        priv = jp.concatenate([o, self._critic_extras(data, info)]) if self.n_extra else o
        if not self.rand:
            return {"state": o, "privileged_state": priv}
        info["rng"], k = jax.random.split(info["rng"])
        return {"state": o + self.rand["obs_noise"] * jax.random.normal(k, o.shape), "privileged_state": priv}

    def _critic_extras(self, data, info):
        v = measure(jp, data.xpos, data.xmat, data.qvel, self.ix, self.task, ctrl=data.ctrl, qpos=data.qpos)
        n = max(len(self.steps), 1)
        m = self._mjx_model  # this copy's own physics (domain randomisation swaps it in)
        return jp.stack([v["grip"], jp.minimum(v["touch_dist"], 0.5) * 4.0, jp.minimum(v["lifted"], 0.5) * 4.0,
                         v["closure"], jp.asarray(v["held"], jp.float32), info["step"].astype(jp.float32) / n,
                         info["t"].astype(jp.float32) * CTRL_DT / self.task["episode_s"],
                         m.body_mass[self.ix.block] / self._mass0, m.geom_friction[self._obj_geom, 0] / self._fric0]
                        + ([info["ease_u"]] if self.n_extra > 9 else []))

    def _demo_reward(self, obs, action):
        """How much this action looks like the demonstration's, in the most similar situation it went through
        (and only to the extent the situation is similar)."""
        mu, sd, dobs, dact = self._demo
        x = (obs[:mu.shape[0]] - mu) / sd  # the robot's own view (the critic's extras come after it)
        d2 = jp.mean(jp.square(dobs - x), axis=1)
        i = jp.argmin(d2)
        near = jp.exp(-d2[i] / 0.5)
        return near * jp.exp(-jp.mean(jp.square(action - dact[i])) / 0.1)

    def step(self, state, action):
        # with a delay, the robot acts on the previous command (like a real control loop that lags a step)
        applied = action + state.info["delay"] * (state.info["last_act"] - action)
        if self.residual is not None:
            applied = self.residual.action(jp, state.info["t"], applied)
        if self.ix.hand is None:
            ctrl = next_ctrl(jp, state.info["ctrl"], applied, self._ctrl0, self.scale, self._lo, self._hi, self._step)
        else:  # acting in hand space: the arm's slots move the hand; Morph's Jacobian step turns that into joint targets
            ctrl = apply_action(jp, state.info["ctrl"], applied, self._ctrl0, self.ix, state.data)
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
        reach = jp.min(jp.linalg.norm(data.xpos[ix.tips] - aim, axis=1)) if len(ix.tips) else jp.asarray(9.0)
        aim_ok = jp.asarray(1.0)  # how well the hand matches the planned grip (1 without a plan)
        if self.grasp is not None:
            reach, aim_ok = self._grasp_aim(data, bp, R)
            if "released" in info:  # thrown or let go on purpose: no pull back towards the object
                reach = jp.where(info["released"] > 0.5, 0.0, reach)
                aim_ok = jp.where(info["released"] > 0.5, 1.0, aim_ok)
        if self.body_goal:  # a goal about the robot's own body: reaching for the object is beside the point
            reach = jp.asarray(9.0)
        progress = (info["prev_d"] - d) / self.dt
        close = jp.exp(-d / 0.03)
        place_ok = d < (self.tol if self.has_place else 1e-3)  # d already includes the unmet conditions
        if any(c["when"] == "release" for c in self.temporal):
            place_ok = place_ok & (info["released"] > 0.5)
        ok = place_ok & ((yaw_err < self.yaw_tol) | (not self.heading)) & ((tilt < 0.17) | (not self.upright))
        # the reward, part by part (reward_parts: each logged on its own and shown in the app; weights from suggestions).
        # Smooth pulls that work from far away (the tool starts ~0.5 m from the object), sharper ones up close
        w = self.w
        part = {"toward": 2.0 * jp.clip(progress, -1.0, 1.0),
                "near": 1.0 * (1 - jp.tanh(d / 0.2)) + 2.0 * close,
                "success": 3.0 * ok,
                "reach": 0.5 * (1 - jp.tanh(reach / 0.25)) * (1 - close) + 0.5 * jp.exp(-reach / 0.05) * aim_ok * (1 - close),
                "smooth": -0.02 * jp.sum(jp.square(action - info["last_act"])),
                # capped: a copy whose physics nearly blew up (a pen at 100 m/s) once outweighed the whole batch
                "fling": -1.0 * jp.clip(jp.linalg.norm(data.qvel[ix.block_v:ix.block_v + 3]) - 0.5, 0.0, 5.0)}
        if self.heading:
            part["heading"] = 0.5 * jp.exp(-yaw_err ** 2 / 0.3) * close
        if self.upright:
            part["upright"] = -1.0 * tilt ** 2
        done = jp.zeros(())
        if ix.base >= 0:
            up = data.xmat[ix.base].reshape(3, 3)[2, 2]
            # tipped over, or (a body that stands up: legs) sunk to under 40% of its standing height. A base that
            # rests on the floor (wheels) only falls by tipping: its height near 0 once counted as fallen every moment
            stand = self.task.get("base_z", 0.4)
            fell = (up < 0.4) | ((data.xpos[ix.base][2] < 0.4 * stand) if stand > 0.1 else False)
            part["standing"] = 0.2 - 5.0 * fell
            done = fell.astype(jp.float32)
        if self._demo is not None:
            part["copy"] = 1.5 * self._demo_reward(state.obs["privileged_state"], action)
        shaped = jp.zeros(())
        if self.steps:  # the steps a person would take: reward reaching each, in order; mind the keep rules
            step, phi, keep, info["streak"] = self._steps(data, info["step"], info["streak"])
            shaped = w["steps"] * 0.5 * (phi - info["phi"])  # a one-off bonus per step reached (not a rate: not scaled by dt)
            part["steps"] = 1.0 * step / len(self.steps) - 1.0 * jp.minimum(keep, 2.0)
            info["step"], info["phi"] = step, phi
        part = {k: w[k] * v for k, v in part.items()}
        r = sum(part.values(), jp.zeros(()))
        ended, end_pen = jp.zeros(()), jp.zeros(())
        if self.stop_rules:  # a "the whole time" condition broken for longer than a contact flicker: the try is over
            info["broke"] = jp.where(self._broken(data, info["ease_u"]), info["broke"] + 1, 0).astype(jp.int32)
            ended = jp.any(info["armed"] & (info["broke"] >= FLICKER_TICKS)).astype(jp.float32)
            # what the rest of the try could have earned is lost, so ending it never pays (otherwise a try whose
            # reward runs negative would learn to break a rule on purpose to stop early)
            left = jp.maximum(self.task["episode_s"] - (info["t"] + 1) * self.dt, 0.0)
            end_pen = ended * 3.0 * left
            done = jp.maximum(done, ended)
        tvals = []
        if self.terms:  # Nemotron's terms, on the graded quantities (each clipped: one term can't swamp the rest)
            vs = soft(measure(jp, data.xpos, data.xmat, data.qvel, ix, self.task, ctrl=data.ctrl, qpos=data.qpos))
            tvals = [jp.clip(term_value(jp, f, vs), -3.0, 3.0) for f, _ in self.terms]
            r = r + sum(wt * tv for (_, wt), tv in zip(self.terms, tvals))
        info["last_act"], info["prev_d"], info["ctrl"] = action, d, ctrl
        info["t"] = info["t"] + 1
        z = lambda x: jp.where(jp.isnan(x), 0.0, x)  # a blown-up copy shouldn't turn the logged averages into NaN
        state.metrics.update(progress=z(progress), close=z(close), reach=z(reach), grasp_aim=z(aim_ok), ended_early=ended,
                             success=ok.astype(jp.float32), dist=z(d),
                             steps_done=info["step"].astype(jp.float32), **{f"term_{k}": z(tv) for k, tv in enumerate(tvals)},
                             # what each part paid this tick (summed over a run: its total), the step bonus included
                             **{f"part_{k}": z(v * self.dt + (shaped if k == "steps" else 0.0)) for k, v in part.items()},
                             **{f"termw_{k}": z(wt * tv * self.dt) for k, ((_, wt), tv) in enumerate(zip(self.terms, tvals))})
        obs = self._obs(data, info)
        # a copy whose physics blew up (NaN) ends now and counts for nothing, as in MuJoCo Playground's own
        # environments; otherwise one bad copy turns the whole batch's numbers into NaN
        bad = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any() | jp.isnan(r) | jp.isnan(shaped)
        reward = jp.where(bad, 0.0, r * self.dt + shaped - end_pen)
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


def imitate(env, net: dict, demo: Path, seed: int, steps: int = 3000, residual: bool = False):
    """Copy a demonstration first (behaviour cloning): train the policy network to output the program's actions
    for the observations along it, so practice starts from something that already grips and lifts instead of
    from random flailing. Returns Brax's (normalizer, policy, value) params for PPO to continue from."""
    import optax
    from brax.training.acme import running_statistics, specs
    z = np.load(demo)
    obs = jp.array(z["obs"], dtype=jp.float32)
    act = jp.array(np.clip(z["act"], -0.95, 0.95), dtype=jp.float32)
    priv = jp.concatenate([obs, jp.zeros((obs.shape[0], env.n_extra))], axis=1) if env.n_extra else obs
    od = {"state": obs, "privileged_state": priv}
    shape = {k: (v.shape[1],) for k, v in od.items()}
    netw = ppo_networks.make_ppo_networks(shape, env.action_size, preprocess_observations_fn=running_statistics.normalize, **net)
    kp, kv, kn = jax.random.split(jax.random.PRNGKey(seed), 3)
    pol, val = netw.policy_network.init(kp), netw.value_network.init(kv)
    if residual:  # corrections start at exactly nothing (whatever it sees): the recipe as it is, then practice adds fixes
        last = sorted(pol["params"])[-1]
        k, b = pol["params"][last]["kernel"], pol["params"][last]["bias"]
        pol = {"params": dict(pol["params"], **{last: {"kernel": k.at[:, :env.action_size].set(0.0),
                                                       "bias": b.at[:env.action_size].set(0.0)}})}
    norm = running_statistics.init_state({k: specs.Array((v.shape[1],), jp.dtype("float32")) for k, v in od.items()})
    # a reading that never changes along the demonstration (a joint it doesn't use) has ~0 spread: dividing by it
    # turns a tiny difference at test time into a huge input. A floor of 2 cm / 0.02 rad keeps the copy sane.
    norm = running_statistics.update(norm, od, std_min_value=0.02)
    n = env.action_size

    spread = jp.maximum(obs.std(0), 0.02)

    def loss(params, key):
        # the same situations, slightly off (DART-style noise): the copy has to answer sensibly near the recorded path
        # too, not only exactly on it; otherwise its first small slip feeds back (it sees its own last action) and grows
        o = obs + 0.3 * spread * jax.random.normal(key, obs.shape)
        out = netw.policy_network.apply(norm, params, {"state": o, "privileged_state": priv})
        mean = jp.tanh(out[:, :n])
        std = jax.nn.softplus(out[:, n:])
        # match the program's actions; keep the exploration noise moderate so practice doesn't wreck it at once
        return jp.mean(jp.square(mean - act)) + 0.1 * jp.mean(jp.square(std - 0.25))

    opt = optax.adam(1e-3)
    st = opt.init(pol)

    @jax.jit
    def upd(params, st, key):
        l, g = jax.value_and_grad(loss)(params, key)
        u, st = opt.update(g, st)
        return optax.apply_updates(params, u), st, l

    for i in range(steps):
        pol, st, l = upd(pol, st, jax.random.fold_in(kn, i))
        if i in (0, steps - 1):
            print(json.dumps({"imitation_loss": float(l), "samples": int(obs.shape[0]), "step": i}), flush=True)
    return (norm, pol, val)


def keep_copying(demo: Path, n_act: int, weight: float, batch: int = 512, n_extra: int = 0):
    """DAPG-style: while it practises, keep a pull towards the demonstration's actions in the situations the
    demonstration went through (an extra term in PPO's loss), so practice refines a working copy instead of
    slowly forgetting it. Rajeswaran et al. 2018 ("Learning Complex Dexterous Manipulation with Deep RL and
    Demonstrations") decay it; here it stays, small, because the demonstration is known to reach the goal."""
    from brax.training.agents.ppo import losses as ppo_losses
    z = np.load(demo)
    obs = jp.array(z["obs"], dtype=jp.float32)
    act = jp.array(np.clip(z["act"], -0.95, 0.95), dtype=jp.float32)
    original = ppo_losses.compute_ppo_loss

    def loss(params, normalizer_params, data, rng, ppo_network, *a, **kw):
        total, metrics = original(params, normalizer_params, data, rng, ppo_network, *a, **kw)
        i = jax.random.randint(jax.random.fold_in(rng, 7), (min(batch, obs.shape[0]),), 0, obs.shape[0])
        o = obs[i]
        priv = jp.concatenate([o, jp.zeros((o.shape[0], n_extra))], axis=1) if n_extra else o
        out = ppo_network.policy_network.apply(normalizer_params, params.policy, {"state": o, "privileged_state": priv})
        bc = jp.mean(jp.square(jp.tanh(out[:, :n_act]) - act[i]))
        metrics = dict(metrics, copy_loss=bc)
        return total + weight * bc, metrics

    ppo_losses.compute_ppo_loss = loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--steps", type=int, default=60_000_000)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--evals", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--restore", help="continue from this earlier round's params.pkl (same robot and scene)")
    ap.add_argument("--imitate-only", action="store_true", help="copy the task's demonstration and stop (no practice)")
    ap.add_argument("--keep-copying", type=float, default=0.5,
                    help="with a demonstration: weight of the pull towards its actions during practice (0 = off)")
    ap.add_argument("--randomize", choices=("off", "light", "full"), default="light",
                    help="vary friction, object mass, motor strength, delay and sensor noise between practice copies")
    a = ap.parse_args()
    task_path = Path(a.task)
    task = load_task(task_path)
    task["_dir"] = str(task_path.parent)
    out = task_path.parent
    # the evaluation always starts from the scene's start: the numbers are the real task, not the practice starts
    env, eval_env = TaskLearn(task, a.envs, a.randomize), TaskLearn(task, 128, a.randomize, practice_starts=False,
                                                                        eased=False)
    n_obs = obs_size(env.ix) + (1 if env.residual is not None else 0)
    print(json.dumps({"obs": n_obs, "actions": env.nu, "residual": env.residual is not None, "tips": [env.mj_model.body(int(b)).name for b in env.ix.tips]}),
          flush=True)
    p = locomotion_params.brax_ppo_config("Go1JoystickFlatTerrain")
    if task.get("ppo") == "manipulation":
        # an arm or a gripper (make_task): settings after MuJoCo Playground's tuned PandaPickCube ones. The moment that
        # matters (fingers round it) rarely happens by chance: explore more, and learn more from each batch once it does.
        # The network keeps Morph's size (Playground's 32-wide policy is too small for several steps in a row)
        p.learning_rate, p.entropy_cost, p.unroll_length = 1e-3, 2e-2, 10
        p.batch_size, p.num_minibatches, p.num_updates_per_batch = 512, 32, 8
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
               "steps_done": float(metrics.get("eval/episode_steps_done", 0.0)) / p.episode_length,
               # the share of evaluation tries that ended early because a "the whole time" condition broke
               "ended_early": float(metrics.get("eval/episode_ended_early", 0.0))}
        if env.terms:  # each of Nemotron's reward terms: its average value per control step in the evaluation
            rec["terms"] = [round(float(metrics.get(f"eval/episode_term_{k}", 0.0)) / p.episode_length, 4)
                            for k in range(len(env.terms))]
            rec["term_totals"] = [round(float(metrics.get(f"eval/episode_termw_{k}", 0.0)), 3) for k in range(len(env.terms))]
        # what each part of the reward paid over a practice run (reward_parts), for the app and Nemotron
        rec["parts"] = {k: round(float(metrics.get(f"eval/episode_part_{k}", 0.0)), 3) for k in env.parts}
        log.append(rec)
        print(json.dumps(rec), flush=True)

    ckpts = out / "checkpoints"
    ckpts.mkdir(exist_ok=True)

    import pickle
    latest = {}

    class StopRound(Exception):
        pass

    def checkpoint(step, _make_policy, params):  # the app plays each one so you can watch it improve
        path = ckpts / f"step_{int(step):012d}.npz"
        export_numpy(params, path)
        latest["params"], latest["step"] = jax.device_get(params), int(step)
        print(json.dumps({"checkpoint": str(path), "step": int(step)}), flush=True)
        if (out / "stop_round").exists():  # the person watching asked to change the plan now: end this round here
            (out / "stop_round").unlink(missing_ok=True)
            raise StopRound()

    rand = domain_randomize(env, a.randomize) if a.randomize != "off" else None
    restore = None
    if not a.restore and task.get("demo") and (out / task["demo"]).exists():  # first round: copy the program first
        restore = imitate(env, net, out / task["demo"], a.seed, residual=env.residual is not None)
        export_numpy(restore, out / "policy.npz")  # the copy on its own: Morph tries it before any practice
        (out / "params.pkl").write_bytes(pickle.dumps(jax.device_get(restore)))
        if a.imitate_only:
            print(json.dumps({"imitated": str(out / "policy.npz")}), flush=True)
            return
    if task.get("demo"):  # practice from a copy: small, careful updates, so practice refines it rather than wrecks it
        # ... unless copying it didn't work: then the learner must be free to move away from it (learn_in_rounds sets
        # how free, from how the copy did: task["copy_trust"] 1 = it works, 0 = it never does)
        trust = float(task.get("copy_trust", 1.0))
        kw["learning_rate"] = kw.get("learning_rate", 3e-4) / (1 + 5 * trust)
        kw["entropy_cost"] = kw.get("entropy_cost", 1e-2) / (1 + 9 * trust)
        weight = a.keep_copying * max(trust, 0.2)
        if weight > 0 and (out / task["demo"]).exists():
            keep_copying(out / task["demo"], env.nu, weight, n_extra=env.n_extra)
    tw = task.get("ppo_tweaks") or {}  # this variant's exploration and learning speed (learn_task's population search)
    if tw:
        kw["entropy_cost"] = kw.get("entropy_cost", 1e-2) * float(tw.get("entropy", 1.0))
        kw["learning_rate"] = kw.get("learning_rate", 3e-4) * float(tw.get("lr", 1.0))
        print(json.dumps({"ppo_tweaks": tw, "entropy_cost": kw["entropy_cost"], "learning_rate": kw["learning_rate"]}),
              flush=True)
    if a.restore and Path(a.restore).exists():  # a later round: continue from where the last one got to
        restore = pickle.loads(Path(a.restore).read_bytes())
        print(json.dumps({"restored": a.restore}), flush=True)
    train = functools.partial(ppo.train, **kw, seed=a.seed, progress_fn=progress, policy_params_fn=checkpoint,
                              restore_params=restore, restore_value_fn=True,
                              randomization_fn=rand,
                              network_factory=functools.partial(ppo_networks.make_ppo_networks, **net))
    stopped = False
    (out / "stop_round").unlink(missing_ok=True)
    try:
        _, params, _ = train(environment=env, eval_env=eval_env, wrap_env_fn=wrapper.wrap_for_brax_training)
    except StopRound:  # ended early on request: keep what it learned up to the last snapshot
        params, stopped = latest["params"], True
        print(json.dumps({"stopped_early": latest["step"]}), flush=True)
    shapes = export_numpy(params, out / "policy.npz")
    (out / "params.pkl").write_bytes(pickle.dumps(jax.device_get(params)))  # for the next round to continue from
    meta = {"task": str(task_path), "steps": a.steps, "num_envs": a.envs, "randomize": a.randomize,
            "randomize_ranges": ranges(a.randomize), "wall_s": round(time.time() - t0, 1),
            "network": shapes, "curve": log, "obs_size": n_obs, "actions": env.nu,
            "keep_copying": a.keep_copying if task.get("demo") else 0.0, "stopped_early": stopped, "ppo_tweaks": tw,
            "residual": task.get("residual") if env.residual is not None else None}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "curve"}), flush=True)


if __name__ == "__main__":
    main()
