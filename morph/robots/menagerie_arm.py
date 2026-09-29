"""Arms from MuJoCo Menagerie (Franka Emika Panda, UFactory xArm7).

The model files give geometry, inertias, joint limits and servos. This adds the
top-down grasp frame, DLS IK with a posture term (7-DoF redundancy), gripper
open/close, aperture and contact-based grip sensing, and a grasp-axis choice that
respects the wrist's joint range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..geometry import wrap_angle
from ..scene import MENAGERIE, BlockSpec, TIMESTEP, _block_xml, _common_assets, _goal_xml, FLOOR_FRICTION
from ..sim import SimulationEnvironment
from .base import Capabilities


@dataclass
class MenagerieArmConfig:
    name: str
    xml: str  # relative to third_party/mujoco_menagerie
    root_body: str
    license: str
    arm_joints: tuple
    arm_actuators: tuple
    gripper_actuator: str
    ctrl_open: float
    ctrl_close: float
    tcp_body: str
    tcp_pos: tuple
    left_bodies: tuple
    right_bodies: tuple
    aperture_joint: str
    aperture_open: float  # joint value when fully open
    aperture_closed: float
    home_q: tuple
    base_pos: tuple = (0.0, 0.0, 0.0)
    gripper_gain_scale: float = 1.0  # stiffer gripper servo (force limit unchanged)
    color_note: str = ""
    notes: list = field(default_factory=list)


PANDA = MenagerieArmConfig(
    name="franka_panda", xml="franka_emika_panda/panda.xml", root_body="link0", license="Apache-2.0",
    arm_joints=tuple(f"joint{i}" for i in range(1, 8)), arm_actuators=tuple(f"actuator{i}" for i in range(1, 8)),
    gripper_actuator="actuator8", ctrl_open=255.0, ctrl_close=0.0,
    tcp_body="hand", tcp_pos=(0.0, 0.0, 0.1034),
    left_bodies=("left_finger",), right_bodies=("right_finger",),
    aperture_joint="finger_joint1", aperture_open=0.04, aperture_closed=0.0,
    home_q=(0.0, 0.0, 0.0, -1.9, 0.0, 1.9, -0.785),
    base_pos=(-0.05, 0.0, 0.0), gripper_gain_scale=10.0,
    notes=["Menagerie gripper servo gain x10 (squeeze ~1 N/finger -> ~12 N); force limit kept at 100 N.",
           "Body gravity compensation enabled (models the vendor controller)."],
)

XARM7 = MenagerieArmConfig(
    name="ufactory_xarm7", xml="ufactory_xarm7/xarm7.xml", root_body="link_base", license="BSD-3-Clause",
    arm_joints=tuple(f"joint{i}" for i in range(1, 8)), arm_actuators=tuple(f"act{i}" for i in range(1, 8)),
    gripper_actuator="gripper", ctrl_open=0.0, ctrl_close=255.0,
    tcp_body="xarm_gripper_base_link", tcp_pos=(0.0, 0.0, 0.172),
    left_bodies=("left_finger", "left_inner_knuckle", "left_outer_knuckle"),
    right_bodies=("right_finger", "right_inner_knuckle", "right_outer_knuckle"),
    aperture_joint="left_driver_joint", aperture_open=0.0, aperture_closed=0.85,
    home_q=(0.0, -0.247, 0.0, 0.909, 0.0, 1.156, 0.0),
    base_pos=(-0.1, 0.0, 0.0),
    notes=["Body gravity compensation enabled (models the vendor controller)."],
)

MENAGERIE_ARMS = {"franka_panda": PANDA, "ufactory_xarm7": XARM7}


def build_menagerie_arm_model(cfg: MenagerieArmConfig, block: BlockSpec) -> mujoco.MjModel:
    path = MENAGERIE / cfg.xml
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/fetch_assets.sh")
    robot = mujoco.MjSpec.from_file(str(path))
    robot.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT  # match the scene (avoids an attach conflict)
    for body in robot.bodies:  # real arm controllers compensate gravity; do the same here
        body.gravcomp = 1.0
    tcp = robot.body(cfg.tcp_body)
    tcp.add_site(name="tcp", pos=list(cfg.tcp_pos), size=[0.006, 0, 0], rgba=[1, 0, 0, 1], group=4)
    scene = mujoco.MjSpec.from_string(f"""<mujoco model="morph_{cfg.name}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{TIMESTEP}" integrator="implicit" cone="elliptic" impratio="10" noslip_iterations="4"/>
  {_common_assets()}
  <worldbody>
    <light pos="0.6 -0.5 2.6" dir="-0.15 0.2 -1" cutoff="45" exponent="2" castshadow="true" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" friction="{FLOOR_FRICTION} 0.005 0.0001"/>
    <camera name="overview" pos="1.35 -0.9 0.95" xyaxes="0.55 0.83 0 -0.42 0.28 0.86"/>
    <camera name="top" pos="0.45 0.0 1.5" xyaxes="1 0 0 0 1 0"/>
    {_goal_xml()}
    {_block_xml(block)}
    <frame name="mount" pos="{cfg.base_pos[0]} {cfg.base_pos[1]} {cfg.base_pos[2]}"/>
  </worldbody>
</mujoco>""")
    scene.worldbody.frames[0].attach_body(robot.body(cfg.root_body), "", "")
    m = scene.compile()
    a = m.actuator(cfg.gripper_actuator).id
    m.actuator_gainprm[a, 0] *= cfg.gripper_gain_scale
    m.actuator_biasprm[a, 1] *= cfg.gripper_gain_scale
    m.actuator_biasprm[a, 2] *= cfg.gripper_gain_scale
    return m


def _rotvec(R: np.ndarray) -> np.ndarray:
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if abs(s) < 1e-6:  # ~pi: use the diagonal
        axis = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        return axis * ang
    return v / (2 * s) * ang


def topdown_R(yaw: float) -> np.ndarray:
    """Tool frame: +z (approach) points down, +y (finger closing axis) at heading `yaw`."""
    y = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


class MenagerieArmAdapter:
    kind = "fixed_arm"

    def __init__(self, cfg: MenagerieArmConfig, block: BlockSpec | None = None):
        self.cfg = cfg
        self.block = block or BlockSpec()
        self.model = build_menagerie_arm_model(cfg, self.block)
        self.env = SimulationEnvironment(self.model)
        m = self.model
        self.name = cfg.name
        self._qadr = np.array([m.joint(j).qposadr[0] for j in cfg.arm_joints])
        self._dof = np.array([m.joint(j).dofadr[0] for j in cfg.arm_joints])
        self._lo = np.array([m.joint(j).range[0] for j in cfg.arm_joints])
        self._hi = np.array([m.joint(j).range[1] for j in cfg.arm_joints])
        self._act = [m.actuator(a).id for a in cfg.arm_actuators]
        self._grip_act = m.actuator(cfg.gripper_actuator).id
        self._site = m.site("tcp").id
        self._ap = m.joint(cfg.aperture_joint).qposadr[0]
        self._scratch = mujoco.MjData(m)
        self.home_q = np.array(cfg.home_q)
        self._left = {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in cfg.left_bodies}
        self._right = {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in cfg.right_bodies}
        self.finger_geoms = self._left | self._right
        self._block_geom = m.geom("block_geom").id
        self._axis_cache: dict = {}
        self.base_xy = np.array(cfg.base_pos[:2])
        self.q_lo, self.q_hi = self._lo, self._hi
        # Measured: Menagerie joint servos move ~17% of a position lead and ~10-16% of a yaw lead per
        # 20 ms tick (vs ~100% for our generated arms); scale the lead so a task-space action means the same speed.
        self.lead_scale = (5.0, 7.0)
        self._fingers_q = [m.joint(j).qposadr[0] for j in ("finger_joint1", "finger_joint2")] \
            if cfg.name == "franka_panda" else []

    # -------------------------------------------------------------- kinematics
    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = self._scratch
        d.qpos[:] = self.env.data.qpos
        d.qpos[self._qadr] = q
        mujoco.mj_kinematics(self.model, d)
        return d.site_xpos[self._site].copy(), d.site_xmat[self._site].reshape(3, 3).copy()

    def ik(self, tcp: np.ndarray, hand_yaw: float, prefer_q5=None, q0: np.ndarray | None = None,
           iters: int = 60) -> np.ndarray | None:
        """Top-down pose IK by damped least squares with a posture term (redundancy)."""
        m, d = self.model, self._scratch
        q = (self.q() if q0 is None else q0).copy()
        Rd = topdown_R(hand_yaw)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        d.qpos[:] = self.env.data.qpos
        for _ in range(iters):
            d.qpos[self._qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = d.site_xpos[self._site]
            R = d.site_xmat[self._site].reshape(3, 3)
            ep = tcp - p
            er = _rotvec(Rd @ R.T)
            if np.linalg.norm(ep) < 5e-4 and np.linalg.norm(er) < 5e-3:
                return q
            mujoco.mj_jacSite(m, d, jp, jr, self._site)
            J = np.vstack([jp[:, self._dof], jr[:, self._dof]])
            e = np.concatenate([ep, er * 0.5])
            JJt = J @ J.T + 1e-4 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            N = np.eye(len(q)) - J.T @ np.linalg.solve(JJt, J)
            dq += N @ (0.05 * (self.home_q - q))
            q = np.clip(q + np.clip(dq, -0.3, 0.3), self._lo, self._hi)
        d.qpos[self._qadr] = q
        mujoco.mj_kinematics(m, d)
        ok = np.linalg.norm(tcp - d.site_xpos[self._site]) < 2e-3 and \
            np.linalg.norm(_rotvec(Rd @ d.site_xmat[self._site].reshape(3, 3).T)) < 2e-2
        return q if ok else None

    # --------------------------------------------------------------- state io
    def q(self) -> np.ndarray:
        return self.env.data.qpos[self._qadr].copy()

    def tcp(self) -> np.ndarray:
        return self.env.data.site_xpos[self._site].copy()

    def hand_yaw(self) -> float:
        R = self.env.data.site_xmat[self._site].reshape(3, 3)
        return math.atan2(R[1, 1], R[0, 1])

    def aperture(self) -> float:
        c = self.cfg
        v = self.env.data.qpos[self._ap]
        return float(np.clip((v - c.aperture_closed) / (c.aperture_open - c.aperture_closed), 0, 1))

    def gripping(self) -> bool:
        d = self.env.data
        left = right = False
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            if self._block_geom in (g1, g2):
                o = g2 if g1 == self._block_geom else g1
                left |= o in self._left
                right |= o in self._right
        return left and right

    def ctrl(self, q: np.ndarray, close: bool) -> np.ndarray:
        u = self.env.data.ctrl.copy()
        u[self._act] = q
        u[self._grip_act] = self.cfg.ctrl_close if close else self.cfg.ctrl_open
        return u

    def reset(self, seed: int, block_xy_yaw, params, q0: np.ndarray | None = None) -> None:
        q0 = self.home_q if q0 is None else q0
        self.env.reset(seed, block_xy_yaw, params)
        self.env.data.qpos[self._qadr] = q0
        for a in self._fingers_q:  # start with the Panda hand open
            self.env.data.qpos[a] = 0.04
        self.env.data.ctrl[:] = self.ctrl(q0, close=False)
        self._axis_cache.clear()
        mujoco.mj_forward(self.model, self.env.data)

    # ------------------------------------------------------ grasp axis choice
    def choose_grasp_yaw(self, block_xy, block_yaw: float, goal_xy, delta: float, hand_yaw: float) -> float:
        """Pick one of the 4 symmetric grasp axes so that both the grasp and the rotated place pose are
        reachable within the wrist's joint range (Panda joint 7 spans only 332 deg)."""
        key = (round(block_xy[0], 2), round(block_xy[1], 2), round(block_yaw, 1), round(goal_xy[0], 2),
               round(goal_xy[1], 2), round(delta, 1))
        if key in self._axis_cache:
            return self._axis_cache[key]
        from ..arm_skills import TRAVEL_Z
        best, best_cost = None, 1e9
        for k in range(4):
            h0 = block_yaw + k * math.pi / 2
            q_a = self.ik(np.array([block_xy[0], block_xy[1], TRAVEL_Z]), h0, q0=self.home_q)
            q_b = self.ik(np.array([goal_xy[0], goal_xy[1], TRAVEL_Z]), h0 + delta, q0=q_a if q_a is not None else self.home_q)
            if q_a is None or q_b is None:
                continue
            cost = abs(wrap_angle(h0 - hand_yaw)) + 0.3 * float(np.max(np.abs(q_b - q_a)))
            if cost < best_cost:
                best, best_cost = h0, cost
        if best is None:
            best = block_yaw + round(wrap_angle(hand_yaw - block_yaw) / (math.pi / 2)) * math.pi / 2
        self._axis_cache[key] = best
        return best

    def reach_ok(self, x: float, y: float) -> bool:
        from ..arm_skills import GRASP_Z, TRAVEL_Z
        return all(self.ik(np.array([x, y, z]), 0.0, q0=self.home_q) is not None for z in (GRASP_Z, TRAVEL_Z))

    def workspace(self) -> dict:
        rs = np.linspace(0.15, 1.0, 60)
        ok = [r for r in rs if self.reach_ok(self.base_xy[0] + r, self.base_xy[1]) and
              self.reach_ok(self.base_xy[0], self.base_xy[1] + r)]
        return {"type": "annulus", "center": [float(v) for v in self.base_xy],
                "r_min": round(float(min(ok)), 3) if ok else None, "r_max": round(float(max(ok)), 3) if ok else None,
                "valid_heights": [0.04, 0.15], "method": "IK sampling along +x and +y"}

    def capabilities(self) -> Capabilities:
        c = self.cfg
        return Capabilities(
            robot=c.name, embodiment="fixed_arm",
            model_source=f"MuJoCo Menagerie {c.xml} ({c.license})",
            base_mobility="fixed", end_effectors=["parallel_gripper"], contact_modes=["grasp", "push"], prehensile=True,
            object_workspace=self.workspace(), max_payload_kg=1.0, max_contact_force_n=50.0,
            can_set_planar_yaw=True, can_flip_object=False,
            action_primitives=["move_tcp(xyz, hand_yaw) via DLS IK", "open", "close", "pick_place(x,y,yaw)"],
            modeled_limits={"joint_ranges": {j: [round(float(a), 3), round(float(b), 3)] for j, a, b in
                                             zip(c.arm_joints, self._lo, self._hi)},
                            "gripper_max_opening_m": 0.08 if c.name == "franka_panda" else 0.085,
                            "grasp": "top-down only"},
            validation={"ik": "tests/test_menagerie.py (FK round-trip)", "pick_place": "held-out eval"},
            notes=c.notes + ["Top-down grasp cannot flip the block; no regrasp fixture."])
