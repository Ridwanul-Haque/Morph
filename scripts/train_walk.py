"""Learn to walk, for a legged robot Morph hasn't seen before (MuJoCo Warp + Brax PPO on the GPU).

The environment is Playground's Go1 joystick task (Apache-2.0, see third_party/playground_go1)
rewritten to build itself from `morph.robots.legged_model.legged_setup`: torso, feet, standing
pose, gains and sensors all come from the model. Rewards and PPO settings are the Go1 ones; the
velocity commands and foot height are scaled by the robot's standing height.

Observation (policy input) is what the robot's own sensors give: IMU velocity, gyro, gravity
direction, joint angles and speeds, plus the last action and the velocity command. Sensor noise
as in Playground.

    MORPH_GPU_MEM=0.7 .venv-rl/bin/python scripts/train_walk.py third_party/mujoco_menagerie/unitree_go2/go2.xml \
        --steps 200_000_000 --out artifacts/walk/go2
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", os.environ.get("MORPH_GPU_MEM", "0.7"))
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

from morph.robots.legged_model import CTRL_DT, SIM_DT, LeggedSetup, build_model, legged_setup  # noqa: E402

GO1_STAND = 0.278  # Playground's Go1 standing height, the reference for command / foot-height scaling


def default_config(setup: LeggedSetup) -> config_dict.ConfigDict:
    k = setup.stand_height / GO1_STAND
    return config_dict.create(
        ctrl_dt=CTRL_DT, sim_dt=SIM_DT, episode_length=1000, action_repeat=1, action_scale=setup.action_scale,
        soft_joint_pos_limit_factor=0.95, impl="warp", naconmax=4 * 8192, njmax=40,
        noise_config=config_dict.create(level=1.0, scales=config_dict.create(
            joint_pos=0.03, joint_vel=1.5, gyro=0.2, gravity=0.05, linvel=0.1)),
        reward_config=config_dict.create(
            scales=config_dict.create(
                tracking_lin_vel=1.0, tracking_ang_vel=0.5, tracking_fine=0.5, lin_vel_z=-0.5, ang_vel_xy=-0.05, orientation=-5.0,
                dof_pos_limits=-1.0, pose=0.5, termination=-1.0, stand_still=-1.0, torques=-0.0002,
                action_rate=-0.01, energy=-0.001, feet_clearance=-2.0, feet_height=-0.2, feet_slip=-0.1,
                feet_air_time=0.1),
            # exp(-err^2 / 0.25) pays 96% for standing still under a 0.1 m/s command, so slow commands were
            # ignored; the fine term (sigma 0.01) only pays for actually matching them
            tracking_sigma=0.25, fine_sigma=0.01, max_foot_height=0.1 * k),
        # Morph's tasks need slow, precise positioning next to an object, not just running: a share of the
        # commands are small (the first Go2 policy trained on Playground's uniform commands ignored
        # anything under ~0.15 m/s and could barely side-step).
        command_config=config_dict.create(a=[1.5 * k, 0.8 * k, 1.2], b=[0.9, 0.25, 0.5],
                                          slow_frac=0.3, slow=[0.3 * k, 0.3 * k, 0.5]),
    )


class Walk(mjx_env.MjxEnv):
    """Velocity-command walking for any legged robot described by a LeggedSetup."""

    def __init__(self, setup: LeggedSetup, config: config_dict.ConfigDict | None = None):
        super().__init__(config or default_config(setup), None)
        self.setup = setup
        self._mj_model = build_model(setup, training=True)
        self._mj_model.opt.ccd_iterations = 20
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        m = self._mj_model
        n = setup.n
        self._init_q = jp.array(m.keyframe("home").qpos)
        self._default_pose = jp.array(setup.default_pose)
        self._qadr = np.array([m.joint(j).qposadr[0] for j in setup.joints])
        self._vadr = np.array([m.joint(j).dofadr[0] for j in setup.joints])
        lo, hi = m.jnt_range[[m.joint(j).id for j in setup.joints]].T
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * self._config.soft_joint_pos_limit_factor
        self._soft_lo, self._soft_hi = jp.array(mid - half), jp.array(mid + half)
        self._torso = m.body(setup.torso).id
        imu_sensor = m.sensor("morph_gyro")
        self._imu_site = int(imu_sensor.objid[0])
        self._feet_site = np.array([m.site(f"morph_foot{i}").id for i in range(len(setup.feet))])
        self._foot_r = jp.array([f["radius"] for f in setup.feet])
        self._foot_vel_adr = jp.array([list(range(m.sensor(f"morph_foot{i}_global_linvel").adr[0],
                                                  m.sensor(f"morph_foot{i}_global_linvel").adr[0] + 3))
                                       for i in range(len(setup.feet))])
        # the joint that moves each foot's own body counts less in the pose reward (like the Go1's knees)
        foot_bodies = {m.body(f["body"]).id for f in setup.feet}
        self._pose_w = jp.array([0.1 if m.jnt_bodyid[m.joint(j).id] in foot_bodies else 1.0 for j in setup.joints])
        self._cmd_a = jp.array(self._config.command_config.a)
        self._cmd_b = jp.array(self._config.command_config.b)
        self._cmd_slow = jp.array(self._config.command_config.slow)
        self._nf = len(setup.feet)
        assert m.nu == n

    # --- sensors
    def _sens(self, data, name):
        return mjx_env.get_sensor_data(self._mj_model, data, f"morph_{name}")

    def _gravity(self, data):
        return data.site_xmat[self._imu_site].T @ jp.array([0, 0, -1.0])

    # --- env API
    def reset(self, rng):
        qpos, qvel = self._init_q, jp.zeros(self._mjx_model.nv)
        rng, k1, k2, k3, k4 = jax.random.split(rng, 5)
        qpos = qpos.at[0:2].add(jax.random.uniform(k1, (2,), minval=-0.5, maxval=0.5))
        yaw = jax.random.uniform(k2, (1,), minval=-3.14, maxval=3.14)
        from mujoco.mjx._src import math
        qpos = qpos.at[3:7].set(math.quat_mul(qpos[3:7], math.axis_angle_to_quat(jp.array([0, 0, 1.0]), yaw)))
        qvel = qvel.at[0:6].set(jax.random.uniform(k3, (6,), minval=-0.5, maxval=0.5))
        data = mjx_env.make_data(self._mj_model, qpos=qpos, qvel=qvel, ctrl=self._default_pose,
                                 impl=self._mjx_model.impl.value, naconmax=self._config.naconmax,
                                 njmax=self._config.njmax)
        data = mjx.forward(self._mjx_model, data)
        rng, kc1, kc2 = jax.random.split(rng, 3)
        info = {"rng": rng, "command": jax.random.uniform(kc2, (3,), minval=-self._cmd_a, maxval=self._cmd_a),
                "steps_until_next_cmd": jp.round(jax.random.exponential(kc1) * 5.0 / self.dt).astype(jp.int32),
                "last_act": jp.zeros(self.setup.n), "feet_air_time": jp.zeros(self._nf),
                "last_contact": jp.zeros(self._nf, dtype=bool), "swing_peak": jp.zeros(self._nf)}
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}
        obs = self._obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state, action):
        data = mjx_env.step(self._mjx_model, state.data, self._default_pose + action * self._config.action_scale,
                            self.n_substeps)
        foot_z = data.site_xpos[self._feet_site][:, 2]
        contact = foot_z < self._foot_r + 0.005
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], foot_z)
        obs = self._obs(data, state.info)
        done = self._sens(data, "upvector")[-1] < 0.0
        rewards = self._rewards(data, action, state.info, done, first_contact, contact)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        state.info["last_act"] = action
        state.info["steps_until_next_cmd"] -= 1
        state.info["rng"], k1, k2 = jax.random.split(state.info["rng"], 3)
        state.info["command"] = jp.where(state.info["steps_until_next_cmd"] <= 0,
                                         self._sample_command(k1, state.info["command"]), state.info["command"])
        state.info["steps_until_next_cmd"] = jp.where(
            done | (state.info["steps_until_next_cmd"] <= 0),
            jp.round(jax.random.exponential(k2) * 5.0 / self.dt).astype(jp.int32), state.info["steps_until_next_cmd"])
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        return state.replace(data=data, obs=obs, reward=reward, done=done.astype(reward.dtype))

    def _obs(self, data, info):
        c = self._config.noise_config

        def noisy(x, scale):
            info["rng"], k = jax.random.split(info["rng"])
            return x + (2 * jax.random.uniform(k, shape=x.shape) - 1) * c.level * scale

        q, qd = data.qpos[self._qadr], data.qvel[self._vadr]
        state = jp.hstack([noisy(self._sens(data, "local_linvel"), c.scales.linvel),
                           noisy(self._sens(data, "gyro"), c.scales.gyro),
                           noisy(self._gravity(data), c.scales.gravity),
                           noisy(q, c.scales.joint_pos) - self._default_pose,
                           noisy(qd, c.scales.joint_vel), info["last_act"], info["command"]])
        # the critic also gets the clean signals and foot contacts (only used in training)
        privileged = jp.hstack([state, self._sens(data, "local_linvel"), self._sens(data, "gyro"), self._gravity(data),
                                q - self._default_pose, qd, info["last_contact"], info["feet_air_time"]])
        return {"state": state, "privileged_state": privileged}

    def _rewards(self, data, action, info, done, first_contact, contact):
        cmd, rc = info["command"], self._config.reward_config
        cmd_norm = jp.linalg.norm(cmd)
        q = data.qpos[self._qadr]
        lin, gyro = self._sens(data, "local_linvel"), self._sens(data, "gyro")
        glin, gang = self._sens(data, "global_linvel"), self._sens(data, "global_angvel")
        up = self._sens(data, "upvector")
        tau, qd = data.actuator_force, data.qvel[self._vadr]
        feet_vel = data.sensordata[self._foot_vel_adr]
        foot_z = data.site_xpos[self._feet_site][:, 2]
        vel_xy_norm = jp.sqrt(jp.linalg.norm(feet_vel[..., :2], axis=-1))
        return {
            "tracking_lin_vel": jp.exp(-jp.sum(jp.square(cmd[:2] - lin[:2])) / rc.tracking_sigma),
            "tracking_ang_vel": jp.exp(-jp.square(cmd[2] - gyro[2]) / rc.tracking_sigma),
            "tracking_fine": jp.exp(-(jp.sum(jp.square(cmd[:2] - lin[:2])) + jp.square(cmd[2] - gyro[2])) / rc.fine_sigma),
            "lin_vel_z": jp.square(glin[2]),
            "ang_vel_xy": jp.sum(jp.square(gang[:2])),
            "orientation": jp.sum(jp.square(up[:2])),
            "stand_still": jp.sum(jp.abs(q - self._default_pose)) * (cmd_norm < 0.01),
            "termination": done,
            "pose": jp.exp(-jp.sum(jp.square(q - self._default_pose) * self._pose_w)),
            "torques": jp.sqrt(jp.sum(jp.square(tau))) + jp.sum(jp.abs(tau)),
            "action_rate": jp.sum(jp.square(action - info["last_act"])),
            "energy": jp.sum(jp.abs(qd) * jp.abs(tau)),
            "feet_slip": jp.sum(jp.sum(jp.square(feet_vel[..., :2]), axis=-1) * contact) * (cmd_norm > 0.01),
            "feet_clearance": jp.sum(jp.abs(foot_z - self._foot_r - rc.max_foot_height) * vel_xy_norm),
            "feet_height": jp.sum(jp.square((info["swing_peak"] - self._foot_r) / rc.max_foot_height - 1.0)
                                  * first_contact) * (cmd_norm > 0.01),
            "feet_air_time": jp.sum((info["feet_air_time"] - 0.1) * first_contact) * (cmd_norm > 0.01),
            "dof_pos_limits": jp.sum(-jp.clip(q - self._soft_lo, None, 0.0) + jp.clip(q - self._soft_hi, 0.0, None)),
        }

    def _sample_command(self, rng, x_k):
        _, y_rng, w_rng, z_rng, s_rng, sv_rng = jax.random.split(rng, 6)
        y_k = jax.random.uniform(y_rng, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a)
        z_k = jax.random.bernoulli(z_rng, self._cmd_b, shape=(3,))
        w_k = jax.random.bernoulli(w_rng, 0.5, shape=(3,))
        x_kp1 = x_k - w_k * (x_k - y_k * z_k)
        slow = jax.random.uniform(sv_rng, shape=(3,), minval=-self._cmd_slow, maxval=self._cmd_slow)
        return jp.where(jax.random.bernoulli(s_rng, self._config.command_config.slow_frac), slow, x_kp1)

    @property
    def xml_path(self) -> str:
        return self.setup.model_path

    @property
    def action_size(self) -> int:
        return self.setup.n

    @property
    def mj_model(self):
        return self._mj_model

    @property
    def mjx_model(self):
        return self._mjx_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--steps", type=int, default=200_000_000)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--evals", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    setup = legged_setup(a.model)
    print("\n".join(setup.notes), flush=True)
    env, eval_env = Walk(setup), Walk(setup)
    p = locomotion_params.brax_ppo_config("Go1JoystickFlatTerrain")
    p.num_timesteps, p.num_envs, p.num_evals = a.steps, a.envs, a.evals
    net = dict(p.network_factory)
    kw = {k: v for k, v in p.items() if k != "network_factory"}
    log, t0 = [], time.time()

    def progress(step, metrics):
        rec = {"step": int(step), "wall_s": round(time.time() - t0, 1),
               "reward": float(metrics.get("eval/episode_reward", float("nan"))),
               "episode_length": float(metrics.get("eval/avg_episode_length", float("nan"))),
               "tracking_lin_vel": float(metrics.get("eval/episode_reward/tracking_lin_vel", float("nan")))}
        log.append(rec)
        print(json.dumps(rec), flush=True)

    train = functools.partial(ppo.train, **kw, seed=a.seed, progress_fn=progress,
                              network_factory=functools.partial(ppo_networks.make_ppo_networks, **net))
    make_inference_fn, params, _ = train(environment=env, eval_env=eval_env,
                                         wrap_env_fn=wrapper.wrap_for_brax_training)
    wall = time.time() - t0
    shapes = export_numpy(params, out / "walk_policy.npz")
    meta = {"robot": setup.name, "model": setup.model_path, "setup": asdict(setup), "steps": a.steps,
            "num_envs": a.envs, "wall_s": round(wall, 1), "steps_per_s": round(a.steps / wall),
            "gpu": str(jax.devices()[0]), "network": shapes, "curve": log,
            "obs": "local_linvel(3) gyro(3) gravity(3) q-default(n) qd(n) last_action(n) command(3)"}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("curve", "setup")}, indent=2))


if __name__ == "__main__":
    main()
