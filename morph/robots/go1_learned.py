"""Learned Go1 walking policy (trained with MuJoCo Playground + Brax PPO, see
scripts/train_go1_walk.py), run on CPU with NumPy inside Morph's own scene.

The policy tracks a body-velocity command (vx, vy, yaw rate): the push planner picks
where to go, the policy handles the legs and balance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_POLICY = Path(__file__).resolve().parents[2] / "artifacts" / "go1_walk" / "go1_walk_policy.npz"


def _swish(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


class NumpyPolicy:
    """Brax PPO policy network (observation normaliser + swish MLP + tanh mean) in NumPy."""

    def __init__(self, path: Path = DEFAULT_POLICY):
        z = np.load(path)
        self.mean, self.std = z["obs_mean"], z["obs_std"]
        n = len([k for k in z.files if k.startswith("W")])
        self.W = [z[f"W{i}"] for i in range(n)]
        self.b = [z[f"b{i}"] for i in range(n)]
        self.act_dim = self.W[-1].shape[1] // 2

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (np.asarray(obs, dtype=np.float64) - self.mean) / self.std
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = _swish(x)
        return np.tanh(x[: self.act_dim])  # deterministic action = tanh(loc)


# ----------------------------------------------------------------------------- deployment in Morph's scene
import math  # noqa: E402

import mujoco  # noqa: E402

from ..geometry import quat_to_mat, tilt_from_mat, wrap_angle, yaw_from_mat  # noqa: E402
from ..scene import FLOOR_FRICTION, REPO_ROOT, BlockSpec, _block_xml, _common_assets, _goal_xml  # noqa: E402
from ..sim import PhysicalParams, SimulationEnvironment  # noqa: E402

PLAYGROUND_GO1_XML = REPO_ROOT / "third_party" / "playground_go1" / "go1_mjx_fullcollisions.xml"
LEGS = ("FR", "FL", "RR", "RL")
# Playground "home" keyframe (training default pose), order FR, FL, RR, RL x (abduction, hip, knee)
DEFAULT_POSE = np.array([0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8])
ACTION_SCALE = 0.5
LEARNED_KP = 35.0  # the gains the policy was trained with
PUSH_KP, PUSH_KV = 200.0, 2.0  # gains used for the three-leg foot push (same torque limits)
SIM_DT = 0.004


def build_learned_go1_model(block: BlockSpec) -> mujoco.MjModel:
    robot = mujoco.MjSpec.from_file(str(PLAYGROUND_GO1_XML))
    robot.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
    robot.option.timestep = SIM_DT
    robot.option.iterations, robot.option.ls_iterations = 100, 50  # full CPU solver (training used MJX's 1/5)
    scene = mujoco.MjSpec.from_string(f"""<mujoco model="morph_go1_learned">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{SIM_DT}" integrator="implicit" cone="elliptic" impratio="10"/>
  {_common_assets()}
  <worldbody>
    <light pos="0.6 -0.5 2.6" dir="-0.15 0.2 -1" cutoff="45" exponent="2" castshadow="true" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" friction="{FLOOR_FRICTION} 0.005 0.0001"/>
    <camera name="overview" pos="1.6 -1.3 1.1" xyaxes="0.63 0.77 0 -0.4 0.33 0.85"/>
    {_goal_xml()}
    {_block_xml(block)}
    <frame name="robot_mount"/>
  </worldbody>
</mujoco>""")
    scene.worldbody.frames[0].attach_body(robot.body("trunk"), "", "")
    m = scene.compile()
    # Menagerie feet rely on floor friction via priority; keep the foot/floor pair at 0.8 like training.
    return m


class LearnedGo1:
    """Go1 driven by the learned velocity-tracking policy; can hand over to the foot-push primitive."""

    kind = "legged"

    def __init__(self, policy_path: Path = DEFAULT_POLICY, block: BlockSpec | None = None):
        self.block = block or BlockSpec()
        self.model = build_learned_go1_model(self.block)
        self.env = SimulationEnvironment(self.model, control_dt=0.02)
        m = self.model
        self.policy = NumpyPolicy(policy_path)
        self.trunk = m.body("trunk").id
        self._free_q = m.jnt_qposadr[m.body_jntadr[self.trunk]]
        self._free_v = m.jnt_dofadr[m.body_jntadr[self.trunk]]
        self._q = np.array([m.joint(f"{l}_{j}_joint").qposadr[0] for l in LEGS for j in ("hip", "thigh", "calf")])
        self._v = np.array([m.joint(f"{l}_{j}_joint").dofadr[0] for l in LEGS for j in ("hip", "thigh", "calf")])
        self._imu = m.site("imu").id
        self._sens = {n: (m.sensor(n).adr[0], m.sensor(n).dim[0]) for n in ("gyro", "local_linvel")}
        self.last_act = np.zeros(12)
        self.mode = "learned"
        self.env.track_extra = lambda: self.env.data.qpos[self._free_q:self._free_q + 2]
        self.ticks = 0
        self.on_tick = None
        self._set_gains(LEARNED_KP, 0.0)

    def _set_gains(self, kp: float, kv: float) -> None:
        m = self.model
        for i in range(m.nu):
            m.actuator_gainprm[i, 0] = kp
            m.actuator_biasprm[i, 1] = -kp
            m.actuator_biasprm[i, 2] = -kv

    def _sensor(self, name: str) -> np.ndarray:
        a, n = self._sens[name]
        return self.env.data.sensordata[a:a + n].copy()

    def reset(self, seed: int, block_xy_yaw, params: PhysicalParams, base_xy_yaw=(0.0, 0.0, 0.0)) -> None:
        x, y, yaw = base_xy_yaw
        self.env.reset(seed, block_xy_yaw, params)
        self.env.data.qpos[self._free_q:self._free_q + 7] = [x, y, 0.278, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.env.data.qpos[self._q] = DEFAULT_POSE
        self.env.data.ctrl[:] = DEFAULT_POSE
        self.last_act[:] = 0
        self._set_gains(LEARNED_KP, 0.0)
        self.mode = "learned"
        mujoco.mj_forward(self.model, self.env.data)

    # ------------------------------------------------------------------ state
    def base(self):
        q = self.env.data.qpos[self._free_q:self._free_q + 7]
        R = quat_to_mat(q[3:7])
        return q[:3].copy(), yaw_from_mat(R), tilt_from_mat(R)

    def fallen(self) -> bool:
        p, _, tilt = self.base()
        return p[2] < 0.15 or tilt > math.radians(45)

    def obs(self, cmd: np.ndarray) -> np.ndarray:
        d = self.env.data
        R_imu = d.site_xmat[self._imu].reshape(3, 3)
        gravity = R_imu.T @ np.array([0.0, 0.0, -1.0])
        return np.concatenate([self._sensor("local_linvel"), self._sensor("gyro"), gravity,
                               d.qpos[self._q] - DEFAULT_POSE, d.qvel[self._v], self.last_act, cmd])

    # ------------------------------------------------------------------ learned locomotion
    def step(self, cmd) -> None:
        if self.mode != "learned":
            self._set_gains(LEARNED_KP, 0.0)
            self.mode = "learned"
        cmd = np.asarray(cmd, dtype=float)
        a = self.policy(self.obs(cmd))
        self.last_act = a
        log = self.env.advance(DEFAULT_POSE + ACTION_SCALE * a)
        self.ticks += 1
        if self.on_tick is not None:
            self.on_tick(self, log)

    def walk(self, cmd, seconds: float) -> None:
        for _ in range(int(seconds / self.env.control_dt)):
            self.step(cmd)
            if self.fallen():
                return

    def go_to(self, x: float, y: float, yaw: float, tol: float = 0.03, yaw_tol: float = 0.06,
              timeout: float = 20.0, vmax=(0.6, 0.35, 0.9)) -> bool:
        """Velocity-command navigation (the learned policy handles legs and balance)."""
        for _ in range(int(timeout / self.env.control_dt)):
            p, th, _ = self.base()
            dx, dy = x - p[0], y - p[1]
            ex = math.cos(th) * dx + math.sin(th) * dy
            ey = -math.sin(th) * dx + math.cos(th) * dy
            dist = math.hypot(dx, dy)
            # far away: face the target first; close: hold the final heading
            yaw_ref = math.atan2(dy, dx) if dist > 0.25 and abs(wrap_angle(math.atan2(dy, dx) - th)) < 2.0 else yaw
            ey_yaw = wrap_angle(yaw_ref - th)
            if dist < tol and abs(wrap_angle(yaw - th)) < yaw_tol:
                self.walk((0.0, 0.0, 0.0), 0.4)  # settle into a stance
                return True
            cmd = (float(np.clip(1.5 * ex, -vmax[0], vmax[0])), float(np.clip(1.5 * ey, -vmax[1], vmax[1])),
                   float(np.clip(2.0 * ey_yaw, -vmax[2], vmax[2])))
            self.step(cmd)
            if self.fallen():
                return False
        return False
