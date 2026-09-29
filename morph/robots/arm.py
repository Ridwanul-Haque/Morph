"""Adapter for the two generated 5-DoF gripper arms (arm A, arm B).

Analytic top-down grasp IK, a Cartesian controller and a scripted pick-and-place
skill (used for demos and to label behaviour-cloning data).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..geometry import quat_to_mat, wrap_angle
from ..scene import ArmGeometry, BlockSpec, build_arm_model
from ..sim import SimulationEnvironment
from .base import Capabilities

ARM_JOINTS = ("j1", "j2", "j3", "j4", "j5")
GRASP_Z = 0.04  # TCP height when grasping the 6 cm tall block
GRIP_LEAD = 0.008  # finger target = measured - lead when closing -> bounded squeeze force
TRAVEL_Z = 0.15
OPEN = 0.065
CLOSED = 0.0


@dataclass
class TcpTarget:
    xyz: np.ndarray
    hand_yaw: float  # world heading of the finger-closing axis
    grip: float  # finger position command (per finger, m from centreline)


class ArmAdapter:
    kind = "fixed_arm"

    def __init__(self, geom: ArmGeometry, block: BlockSpec | None = None):
        self.g = geom
        self.block = block or BlockSpec()
        self.model = build_arm_model(geom, self.block)
        self.env = SimulationEnvironment(self.model)
        self._qadr = [self.model.joint(j).qposadr[0] for j in ARM_JOINTS]
        self._hand = self.model.body("hand").id
        self._fl = self.model.joint("fl").qposadr[0]
        self._fr = self.model.joint("fr").qposadr[0]
        self.name = geom.name
        self.q_lo = np.array([geom.joint_ranges[j][0] for j in ARM_JOINTS])
        self.q_hi = np.array([geom.joint_ranges[j][1] for j in ARM_JOINTS])
        self._block_geom = self.model.geom("block_geom").id
        self.finger_geoms = {self.model.geom("finger_l").id, self.model.geom("finger_r").id}
        self.base_xy = np.array(geom.base_xy)
        self.home_q = self.ik(np.array([0.35, 0.0, 0.25]), 0.0, prefer_q5=0.0)
        assert self.home_q is not None, f"{geom.name}: home pose unreachable"

    # ------------------------------------------------------------- kinematics
    def ik(self, tcp: np.ndarray, hand_yaw: float, prefer_q5: float | None = None) -> np.ndarray | None:
        """Top-down grasp IK. Returns joint vector (j1..j5) or None if unreachable/out of limits."""
        g = self.g
        dx, dy = tcp[0] - g.base_xy[0], tcp[1] - g.base_xy[1]
        r = math.hypot(dx, dy)
        q1 = math.atan2(dy, dx)
        dz = tcp[2] + g.l3 - g.pedestal
        D = (r * r + dz * dz - g.l1**2 - g.l2**2) / (2 * g.l1 * g.l2)
        if abs(D) > 1.0:
            return None
        q3 = math.acos(D)  # elbow up
        q2 = math.atan2(r, dz) - math.atan2(g.l2 * math.sin(q3), g.l1 + g.l2 * math.cos(q3))
        q4 = math.pi - q2 - q3
        # Pitches sum to pi (gripper down): R = Rz(q1)Ry(pi)Rz(q5) = Rz(q1-q5)Ry(pi), so the finger
        # closing axis (hand +y) points at q1 - q5 + pi/2.
        q5 = wrap_angle(q1 + math.pi / 2 - hand_yaw)
        if prefer_q5 is None and hasattr(self, "env"):
            prefer_q5 = float(self.env.data.qpos[self._qadr[4]])
        if prefer_q5 is not None:  # choose the 2*pi branch closest to the preferred value
            q5 = min((q5 + k * 2 * math.pi for k in (-1, 0, 1)), key=lambda v: abs(v - prefer_q5))
        q = np.array([q1, q2, q3, q4, q5])
        for v, j in zip(q, ARM_JOINTS):
            lo, hi = g.joint_ranges[j]
            if not (lo - 1e-6 <= v <= hi + 1e-6):
                return None
        return q

    def reach_ok(self, x: float, y: float) -> bool:
        return all(self.ik(np.array([x, y, z]), 0.0) is not None for z in (GRASP_Z, TRAVEL_Z))

    def workspace(self) -> dict:
        rs = np.linspace(0.05, 1.0, 400)
        ok = [r for r in rs if self.reach_ok(self.g.base_xy[0] + r, self.g.base_xy[1])]
        return {"type": "annulus", "center": [float(v) for v in self.g.base_xy], "r_min": round(float(min(ok)), 3),
                "r_max": round(float(max(ok)), 3), "valid_heights": [GRASP_Z, TRAVEL_Z]}

    # --------------------------------------------------------------- state io
    def q(self) -> np.ndarray:
        return self.env.data.qpos[self._qadr].copy()

    def finger(self) -> float:
        d = self.env.data.qpos
        return float(0.5 * (d[self._fl] + d[self._fr]))

    def tcp(self) -> np.ndarray:
        return self.env.site_pos("tcp")

    def hand_yaw(self) -> float:
        R = quat_to_mat(self.env.data.xquat[self._hand])
        return math.atan2(R[1, 1], R[0, 1])  # heading of local +y (closing axis)

    def aperture(self) -> float:
        return float(np.clip(self.finger() / OPEN, 0, 1))

    def gripping(self) -> bool:
        d = self.env.data
        seen = set()
        for i in range(d.ncon):
            c = d.contact[i]
            if self._block_geom in (c.geom1, c.geom2):
                seen |= {c.geom1, c.geom2} & self.finger_geoms
        return seen == self.finger_geoms

    def ctrl(self, q: np.ndarray, close: bool) -> np.ndarray:
        """Joint targets + finger target. Closing = measured - lead (bounded squeeze force)."""
        f = self.finger()
        g = max(CLOSED, f - GRIP_LEAD) if close else min(OPEN, f + 0.02)
        return np.concatenate([q, [g, g]])

    def choose_grasp_yaw(self, block_xy, block_yaw: float, goal_xy, delta: float, hand_yaw: float) -> float:
        """Continuous wrist roll (+-2pi): simply the symmetric grasp axis nearest the current hand heading."""
        return block_yaw + round(wrap_angle(hand_yaw - block_yaw) / (math.pi / 2)) * math.pi / 2

    def reset(self, seed: int, block_xy_yaw, params, q0: np.ndarray | None = None) -> None:
        q0 = self.home_q if q0 is None else q0
        self.env.reset(seed, block_xy_yaw, params,
                       robot_qpos={j: float(v) for j, v in zip(ARM_JOINTS, q0)} | {"fl": OPEN, "fr": OPEN})
        self.env.data.ctrl[:] = np.concatenate([q0, [OPEN, OPEN]])

    def capabilities(self) -> Capabilities:
        return Capabilities(
            robot=self.g.name,
            embodiment="fixed_arm",
            model_source="generated MJCF (morph/scene.py), link lengths "
                         f"l1={self.g.l1} l2={self.g.l2} l3={self.g.l3} shoulder={self.g.pedestal}",
            base_mobility="fixed",
            end_effectors=["parallel_gripper"],
            contact_modes=["grasp", "push"],
            prehensile=True,
            object_workspace=self.workspace(),
            max_payload_kg=round(2 * 1.5 * self.g.grip_force / 9.81 * 0.5, 2),  # friction-limited, 2x margin
            max_contact_force_n=self.g.grip_force,
            can_set_planar_yaw=True,
            can_flip_object=False,
            action_primitives=["move_tcp(xyz, hand_yaw)", "open", "close", "pick_place(x,y,yaw)"],
            modeled_limits={"joint_ranges": self.g.joint_ranges, "grip_force_n": self.g.grip_force,
                            "gripper_max_opening_m": 2 * self.g.max_opening,
                            "grasp": "top-down only (wrist pitch fixed to vertical)"},
            validation={"ik": "analytic, checked by forward kinematics in tests/test_arm.py",
                        "pick_place": "scripted primitive executed in MuJoCo; see runs/*/source"},
            notes=["Top-down grasp cannot turn the block onto another face: flipping needs a regrasp "
                   "fixture or a side grasp, neither implemented."],
        )
