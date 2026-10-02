"""Any legged robot, walking with the policy Morph trained for it (scripts/train_walk.py), inside
Morph's scene. Same model changes, sensors and observation layout as in training; full collisions
and the full CPU solver here."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from ..geometry import quat_to_mat, tilt_from_mat, wrap_angle, yaw_from_mat
from ..scene import FLOOR_FRICTION, BlockSpec, _block_xml, _common_assets, _goal_xml
from ..sim import PhysicalParams, SimulationEnvironment
from .go1_learned import NumpyPolicy
from .legged_model import SIM_DT, LeggedSetup, apply_joint_gains, build_spec

WALK_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "walk"


def load_walker_meta(policy_dir: Path) -> LeggedSetup:
    meta = json.loads((policy_dir / "train_meta.json").read_text())
    return LeggedSetup(**meta["setup"])


def build_walker_model(setup: LeggedSetup, block: BlockSpec) -> mujoco.MjModel:
    robot = build_spec(setup, training=False, standalone=False)
    for k in list(robot.keys):
        robot.delete(k)
    scene = mujoco.MjSpec.from_string(f"""<mujoco model="morph_{setup.name}_walker">
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
    # contact settings as the robot was trained with (the Go2 model asks for impratio 100)
    scene.option.impratio, scene.option.cone = robot.option.impratio, robot.option.cone
    scene.worldbody.frames[0].attach_body(robot.body(setup.torso), "", "")
    scene.option.iterations, scene.option.ls_iterations = 100, 50
    m = scene.compile()
    apply_joint_gains(m, setup)
    return m


class LearnedWalker:
    """A legged robot driven by its own learned velocity-tracking policy."""

    kind = "legged"

    def __init__(self, policy_dir: Path, block: BlockSpec | None = None):
        self.setup = load_walker_meta(policy_dir)
        self.block = block or BlockSpec()
        self.model = build_walker_model(self.setup, self.block)
        self.env = SimulationEnvironment(self.model, control_dt=0.02)
        self.policy = NumpyPolicy(policy_dir / "walk_policy.npz")
        m, s = self.model, self.setup
        self.torso = m.body(s.torso).id
        free = m.body_jntadr[self.torso]
        self._fq, self._fv = m.jnt_qposadr[free], m.jnt_dofadr[free]
        self._q = np.array([m.joint(j).qposadr[0] for j in s.joints])
        self._v = np.array([m.joint(j).dofadr[0] for j in s.joints])
        self._imu = int(m.sensor("morph_gyro").objid[0])
        self.default = np.array(s.default_pose)
        self.last_act = np.zeros(s.n)
        self.env.track_extra = lambda: self.env.data.qpos[self._fq:self._fq + 2]
        self.ticks = 0
        self.on_tick = None
        self.mode = "learned"
        self._legs = None
        # the foot push holds three feet with stiffer servos; same ratio as the Go1 (35 -> 200, damping x4)
        self.push_gains = (s.kp * 200.0 / 35.0, s.kd * 4.0)

    def _set_gains(self, kp: float, kv: float) -> None:
        m = self.model
        for i in range(m.nu):
            m.actuator_gainprm[i, 0] = kp
            m.actuator_biasprm[i, 1] = -kp
            m.actuator_biasprm[i, 2] = -kv

    def enter_push_mode(self) -> None:
        self._set_gains(*self.push_gains)
        self.mode = "push"

    def legs_adapter(self):
        """QuadrupedAdapter over this scene with leg geometry measured from the model (for the foot push)."""
        from .legs import measure_legs
        from .quadruped import QuadrupedAdapter
        if self._legs is None:
            q = self.model.qpos0.copy()
            q[self._fq:self._fq + 7] = [0, 0, self.setup.stand_height, 1, 0, 0, 0]
            q[self._q] = self.default
            feet = [dict(f, geom_id=self.model.geom(f["geom"]).id) for f in self.setup.feet]
            # resting foot height: stand for 1 s on the servos (scratch data, away from the block)
            sd = mujoco.MjData(self.model)
            sd.qpos[:] = q
            sd.qpos[self._fq:self._fq + 2] = [-5.0, -5.0]
            sd.ctrl[:] = self.default
            for _ in range(int(1.0 / self.model.opt.timestep)):
                mujoco.mj_step(self.model, sd)
            z = float(np.mean([sd.site(f"morph_foot{i}").xpos[2] for i in range(len(feet))]))
            R = sd.xmat[self.torso].reshape(3, 3)
            self._com_off = (R.T @ (sd.subtree_com[self.torso] - sd.xpos[self.torso]))[:2]  # trunk frame
            self._legs = measure_legs(self.model, self.setup.torso, feet, self.setup.joints, q, foot_z=z)
        return QuadrupedAdapter(model=self.model, env=self.env, legs=self._legs, torso=self.setup.torso)

    def _balance_shift(self, leg: str, width: float = 0.04, margin: float = 0.05) -> np.ndarray:
        """Where the body will move (trunk frame) to balance on the other three feet before `leg` lifts:
        the same CoM-inside-the-support-triangle rule as Go1Controller._support_target."""
        g = self._legs
        pts = np.array([g.nominal[l] + np.array([0.0, g.side[l] * width]) for l in g.legs if l != leg])
        c0 = pts.mean(0)
        pts = pts[np.argsort(np.arctan2(pts[:, 1] - c0[1], pts[:, 0] - c0[0]))]
        com = self._com_off.copy()
        for _ in range(3):
            for i in range(len(pts)):
                a, b = pts[i], pts[(i + 1) % len(pts)]
                e = b - a
                n = np.array([-e[1], e[0]]) / (np.linalg.norm(e) + 1e-9)
                d = float(n @ (com - a))
                if d < margin:
                    com = com + n * (margin - d)
        return com - self._com_off

    def push_pose(self, contact_xy, heading: float, leg: str = "FR") -> tuple[float, float, float]:
        """Base pose that puts `leg`'s foot 5 cm behind contact_xy, 5 cm ahead of and 4 cm outside its
        standing spot once the body has shifted onto the other three feet (the Go1's hand-tuned
        FOOT_IN_TRUNK is this rule, with the shift found by trial)."""
        from ..geometry import rot2
        g = self._legs or self.legs_adapter().legs
        foot = g.nominal[leg] + np.array([0.05, 0.04 * g.side[leg]]) + self._balance_shift(leg)
        u = np.array([math.cos(heading), math.sin(heading)])
        base = np.array(contact_xy) - u * 0.05 - rot2(heading) @ foot
        return float(base[0]), float(base[1]), heading

    def sensor(self, name: str) -> np.ndarray:
        return self.env.data.sensor(f"morph_{name}").data.copy()

    def reset(self, seed: int, block_xy_yaw, params: PhysicalParams, base_xy_yaw=(0.0, 0.0, 0.0)) -> None:
        x, y, yaw = base_xy_yaw
        self.env.reset(seed, block_xy_yaw, params)
        d = self.env.data
        d.qpos[self._fq:self._fq + 7] = [x, y, self.setup.stand_height, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        d.qpos[self._q] = self.default
        d.ctrl[:] = self.default
        self.last_act[:] = 0
        self._set_gains(self.setup.kp, 0.0)
        self.mode = "learned"
        mujoco.mj_forward(self.model, d)

    def base(self):
        q = self.env.data.qpos[self._fq:self._fq + 7]
        R = quat_to_mat(q[3:7])
        return q[:3].copy(), yaw_from_mat(R), tilt_from_mat(R)

    def fallen(self) -> bool:
        p, _, tilt = self.base()
        return p[2] < 0.5 * self.setup.stand_height or tilt > math.radians(45)

    def obs(self, cmd: np.ndarray) -> np.ndarray:
        d = self.env.data
        gravity = d.site_xmat[self._imu].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        return np.concatenate([self.sensor("local_linvel"), self.sensor("gyro"), gravity,
                               d.qpos[self._q] - self.default, d.qvel[self._v], self.last_act, cmd])

    def step(self, cmd) -> None:
        if self.mode != "learned":
            self._set_gains(self.setup.kp, 0.0)
            self.mode = "learned"
        a = self.policy(self.obs(np.asarray(cmd, dtype=float)))
        self.last_act = a
        log = self.env.advance(self.default + self.setup.action_scale * a)
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
        """Velocity-command navigation; the learned policy handles legs and balance."""
        for _ in range(int(timeout / self.env.control_dt)):
            p, th, _ = self.base()
            dx, dy = x - p[0], y - p[1]
            ex = math.cos(th) * dx + math.sin(th) * dy
            ey = -math.sin(th) * dx + math.cos(th) * dy
            dist = math.hypot(dx, dy)
            yaw_ref = math.atan2(dy, dx) if dist > 0.25 and abs(wrap_angle(math.atan2(dy, dx) - th)) < 2.0 else yaw
            if dist < tol and abs(wrap_angle(yaw - th)) < yaw_tol:
                self.walk((0.0, 0.0, 0.0), 0.4)
                return True
            self.step((float(np.clip(1.5 * ex, -vmax[0], vmax[0])), float(np.clip(1.5 * ey, -vmax[1], vmax[1])),
                       float(np.clip(2.0 * wrap_angle(yaw_ref - th), -vmax[2], vmax[2]))))
            if self.fallen():
                return False
        return False


def walk_check(policy_dir: Path, seconds: float = 4.0) -> list[dict]:
    """Sim-to-sim check in Morph's scene: stand, track forward/back/sideways/turn commands, stay up."""
    w = LearnedWalker(policy_dir)
    tests = [("stand", (0.0, 0.0, 0.0)), ("forward 0.5", (0.5, 0.0, 0.0)), ("backward 0.3", (-0.3, 0.0, 0.0)),
             ("sideways 0.3", (0.0, 0.3, 0.0)), ("turn left 0.8", (0.0, 0.0, 0.8)), ("turn right 0.8", (0.0, 0.0, -0.8))]
    rows = []
    for name, cmd in tests:
        w.reset(0, (5, 5, 0), PhysicalParams())
        w.walk((0, 0, 0), 1.0)
        v, tilt = [], []
        for _ in range(int(seconds / 0.02)):
            w.step(cmd)
            lv = w.sensor("local_linvel")
            v.append((lv[0], lv[1], w.sensor("gyro")[2]))
            tilt.append(math.degrees(w.base()[2]))
            if w.fallen():
                break
        v = np.array(v[50:]) if len(v) > 60 else np.array(v)
        rows.append({"test": name, "cmd": cmd, "fell": w.fallen(),
                     "mean_vx": round(float(v[:, 0].mean()), 3), "mean_vy": round(float(v[:, 1].mean()), 3),
                     "mean_wz": round(float(v[:, 2].mean()), 3), "max_tilt_deg": round(max(tilt), 1)})
    return rows


if __name__ == "__main__":
    import sys
    for r in walk_check(Path(sys.argv[1])):
        print(r)
