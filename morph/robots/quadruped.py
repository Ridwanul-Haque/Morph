"""Adapter for the Unitree Go1 (MuJoCo Menagerie model, BSD-3-Clause).

Menagerie provides the model and joint position actuators. Leg IK, standing,
weight shifting and the foot push are implemented here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from ..geometry import quat_to_mat, tilt_from_mat, yaw_from_mat
from ..scene import BlockSpec, build_quadruped_model
from ..sim import SimulationEnvironment
from .base import Capabilities

LEGS = ("FR", "FL", "RR", "RL")  # actuator order in go1.xml
HIP = {"FR": (0.1881, -0.04675), "FL": (0.1881, 0.04675), "RR": (-0.1881, -0.04675), "RL": (-0.1881, 0.04675)}
SIDE = {"FR": -1, "FL": 1, "RR": -1, "RL": 1}
L_ABD, L_THIGH, L_CALF = 0.08, 0.213, 0.213
STAND_HEIGHT = 0.27
HOME_Q = np.array([0.0, 0.9, -1.8] * 4)
SERVO_KP, SERVO_KV = 200.0, 2.0


def leg_ik(leg: str, p: np.ndarray) -> np.ndarray | None:
    """Foot position in the trunk frame -> (abduction, hip, knee). Analytic."""
    hx, hy = HIP[leg]
    x, y, z = p[0] - hx, p[1] - hy, p[2]
    l1 = L_ABD * SIDE[leg]
    d2 = y * y + z * z
    if d2 <= l1 * l1:
        return None
    L = math.sqrt(d2 - l1 * l1)
    q1 = math.atan2(z, y) - math.atan2(-L, l1)
    r2 = x * x + L * L
    c3 = (r2 - L_THIGH**2 - L_CALF**2) / (2 * L_THIGH * L_CALF)
    if abs(c3) > 1:
        return None
    q3 = -math.acos(c3)
    phi = math.atan2(-x, L)
    q2 = phi - math.atan2(L_CALF * math.sin(q3), L_THIGH + L_CALF * math.cos(q3))
    q = np.array([math.remainder(q1, 2 * math.pi), q2, q3])
    lim = [(-0.863, 0.863), (-0.686, 4.501), (-2.818, -0.888)]
    if any(not (lo <= v <= hi) for v, (lo, hi) in zip(q, lim)):
        return None
    return q


def home_feet() -> dict[str, np.ndarray]:
    """Default foot positions (trunk frame) for the standing posture."""
    return {leg: np.array([HIP[leg][0], HIP[leg][1] + L_ABD * SIDE[leg], -STAND_HEIGHT + 0.005]) for leg in LEGS}


@dataclass
class BaseState:
    pos: np.ndarray
    yaw: float
    roll: float
    pitch: float
    tilt: float
    R: np.ndarray


class QuadrupedAdapter:
    kind = "legged"

    def __init__(self, block: BlockSpec | None = None, control_dt: float = 0.02, model: mujoco.MjModel | None = None,
                 env: SimulationEnvironment | None = None):
        """Default: Menagerie Go1 scene. Pass `model`/`env` to drive an existing Go1 scene (same joint,
        site and actuator names), e.g. the learned-walking model; the caller manages the gains then."""
        self.block = block or BlockSpec()
        if model is None:
            self.model = build_quadruped_model(self.block)
            # Menagerie ships kp=100; we use kp=200, kv=2 (same torque limits, 23.7 / 35.55 Nm)
            # and the implicit-fast integrator.
            self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            for i in range(self.model.nu):
                self.model.actuator_gainprm[i, 0] = SERVO_KP
                self.model.actuator_biasprm[i, 1] = -SERVO_KP
                self.model.actuator_biasprm[i, 2] = -SERVO_KV
            self.env = SimulationEnvironment(self.model, control_dt=control_dt)
        else:
            self.model, self.env = model, env
        m = self.model
        self.trunk = m.body("trunk").id
        self._free_q = m.jnt_qposadr[m.body_jntadr[self.trunk]]
        self._free_v = m.jnt_dofadr[m.body_jntadr[self.trunk]]
        self._leg_q = np.array([m.joint(f"{l}_{j}_joint").qposadr[0] for l in LEGS for j in ("hip", "thigh", "calf")])
        self.foot_geom = {l: m.geom(l).id for l in LEGS}
        self.foot_site = {l: m.site(l).id for l in LEGS}
        self._robot_bodies = {b for b in range(m.nbody) if self._is_robot_body(b)}
        self.env.track_extra = lambda: self.env.data.qpos[self._free_q:self._free_q + 2]

    def _is_robot_body(self, b: int) -> bool:
        m = self.model
        while b != 0:
            if b == self.trunk:
                return True
            b = m.body_parentid[b]
        return False

    # ------------------------------------------------------------------ state
    def base(self) -> BaseState:
        d = self.env.data
        q = d.qpos[self._free_q:self._free_q + 7]
        R = quat_to_mat(q[3:7])
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
        return BaseState(q[:3].copy(), yaw_from_mat(R), roll, pitch, tilt_from_mat(R), R)

    def foot_world(self, leg: str) -> np.ndarray:
        return self.env.data.site_xpos[self.foot_site[leg]].copy()

    def com(self) -> np.ndarray:
        return self.env.data.subtree_com[self.trunk].copy()

    def world_to_trunk(self, p: np.ndarray) -> np.ndarray:
        b = self.base()
        return b.R.T @ (p - b.pos)

    def trunk_to_world(self, p: np.ndarray) -> np.ndarray:
        b = self.base()
        return b.R @ p + b.pos

    def heading_frame_to_trunk(self, p_world: np.ndarray, height: float) -> np.ndarray:
        """World point -> trunk frame using yaw only, with z measured from the floor.

        Keeps foot targets consistent while the trunk pitches/rolls slightly.
        """
        b = self.base()
        c, s = math.cos(b.yaw), math.sin(b.yaw)
        dx, dy = p_world[0] - b.pos[0], p_world[1] - b.pos[1]
        return np.array([c * dx + s * dy, -s * dx + c * dy, p_world[2] - height])

    def support_margin(self, stance: list[str]) -> float:
        """Signed distance of the CoM ground projection inside the stance polygon (m)."""
        pts = np.array([self.foot_world(l)[:2] for l in stance])
        c = self.com()[:2]
        centre = pts.mean(0)
        ang = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
        pts = pts[np.argsort(ang)]
        margins = []
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            e = b - a
            n = np.array([-e[1], e[0]]) / (np.linalg.norm(e) + 1e-9)  # inward for CCW order
            margins.append(float(n @ (c - a)))
        return min(margins)

    def robot_floor_contacts(self) -> list[str]:
        """Robot geoms other than feet touching the floor (a fall indicator)."""
        m, d = self.model, self.env.data
        floor = m.geom("floor").id
        feet = set(self.foot_geom.values())
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            if floor not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == floor else c.geom1
            if other in feet:
                continue
            if m.geom_bodyid[other] in self._robot_bodies:
                out.append(m.body(m.geom_bodyid[other]).name)
        return out

    def fallen(self) -> bool:
        """Trunk low/tilted, or trunk/hip/thigh on the floor. The lower calf capsule ends 1 cm
        above the foot sphere and may brush the floor while standing; that is not a fall."""
        b = self.base()
        body_hits = [n for n in self.robot_floor_contacts() if not n.endswith("_calf")]
        return b.pos[2] < 0.15 or b.tilt > math.radians(35) or bool(body_hits)

    # ---------------------------------------------------------------- control
    def ctrl_from_feet(self, feet: dict[str, np.ndarray]) -> np.ndarray | None:
        q = []
        for leg in LEGS:
            ql = leg_ik(leg, feet[leg])
            if ql is None:
                return None
            q.append(ql)
        return np.concatenate(q)

    def reset(self, seed: int, block_xy_yaw, params, base_xy_yaw=(0.0, 0.0, 0.0)) -> None:
        x, y, yaw = base_xy_yaw
        free = np.array([x, y, STAND_HEIGHT + 0.01, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        self.env.reset(seed, block_xy_yaw, params)
        self.env.data.qpos[self._free_q:self._free_q + 7] = free  # Menagerie's trunk freejoint is unnamed
        self.env.data.qpos[self._leg_q] = HOME_Q
        self.env.data.ctrl[:] = HOME_Q
        mujoco.mj_forward(self.model, self.env.data)

    def capabilities(self) -> Capabilities:
        return Capabilities(
            robot="unitree_go1",
            embodiment="legged",
            model_source="MuJoCo Menagerie unitree_go1 (BSD-3-Clause), position actuators kp=100",
            base_mobility="legged_planar",
            end_effectors=["FR_foot", "FL_foot"],
            contact_modes=["push"],
            prehensile=False,
            object_workspace={"type": "floor_reachable_by_walking", "object_height_range_m": [0.02, 0.12]},
            max_payload_kg=0.0,
            max_contact_force_n=15.0,
            can_set_planar_yaw=True,
            can_flip_object=False,
            action_primitives=["stand", "shift_weight(stance)", "place_foot(leg, xyz)", "push(leg, dir, dist)",
                               "reposition(x, y, yaw)"],
            modeled_limits={"joint_torque_nm": {"hip/thigh": 23.7, "knee": 35.55},
                            "foot_contact": "sphere r=0.023 m, friction 0.8"},
            validation={"spike": "scripts/spike_quadruped.py"},
            notes=["No gripper: can only push. Yaw is changed by off-centre pushes.",
                   "Flipping a block onto another face would need a lever/fixture; not implemented."],
        )
