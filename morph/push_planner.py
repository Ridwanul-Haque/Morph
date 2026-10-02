"""Push planning by simulation.

The planning model is a separate small MuJoCo scene with the same block but nominal
mass/friction (the planner doesn't know the true values) and a foot-sized sphere
moved along the push line. It predicts where each candidate push leaves the block;
the robot executes the best one and the measured result feeds the next plan.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from .geometry import quat_from_yaw, quat_to_mat, rot2, wrap_angle, yaw_from_mat
from .scene import FLOOR_FRICTION, TIMESTEP, BlockSpec, _block_xml

FOOT_R = 0.023


@dataclass
class PushCandidate:
    face: int  # 0:+x 1:+y 2:-x 3:-y (block frame outward normal of the face being pushed)
    offset: float  # along the face tangent (m)
    distance: float  # travel after contact (m)
    direction: float = 0.0  # world heading of the push (filled in)
    contact_xy: tuple = (0.0, 0.0)  # foot centre at first contact (world)
    predicted: tuple = (0.0, 0.0, 0.0)  # block x, y, yaw after the push
    cost: float = 0.0

    def as_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


class PushModel:
    def __init__(self, block: BlockSpec = BlockSpec(), mass: float = 0.2, friction: float = 0.6,
                 push_height: float = 0.035, speed: float = 0.06):
        xml = f"""<mujoco><option timestep="{TIMESTEP}" integrator="implicitfast" cone="elliptic" impratio="10"/>
<worldbody><geom name="floor" type="plane" size="0 0 0.05" friction="{FLOOR_FRICTION} 0.005 0.0001"/>
<body name="goal" mocap="true"/>
{_block_xml(BlockSpec(block.half_x, block.half_y, block.half_z, mass, friction, fixtures_xml=block.fixtures_xml))}
<body name="pusher" mocap="true" pos="5 5 {push_height}"><geom type="sphere" size="{FOOT_R}" friction="0.8 0.02 0.01"/></body>
</worldbody></mujoco>"""
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.block = block
        self.h = push_height
        self.speed = speed
        self._bq = self.m.joint("block_free").qposadr[0]
        self._p = self.m.body("pusher").mocapid[0]

    def geometry(self, bx: float, by: float, byaw: float, face: int, offset: float) -> tuple[np.ndarray, float]:
        """Contact point (foot centre) and push heading for a face/offset."""
        half = self.block.half_x if face in (0, 2) else self.block.half_y
        n_local = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])][face]
        t_local = np.array([-n_local[1], n_local[0]])
        R = rot2(byaw)
        n, t = R @ n_local, R @ t_local
        c = np.array([bx, by]) + n * (half + FOOT_R) + t * offset
        return c, math.atan2(-n[1], -n[0])

    def predict(self, bx: float, by: float, byaw: float, cand: PushCandidate) -> tuple[float, float, float]:
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[self._bq:self._bq + 3] = [bx, by, self.block.half_z]
        d.qpos[self._bq + 3:self._bq + 7] = quat_from_yaw(byaw)
        c, heading = self.geometry(bx, by, byaw, cand.face, cand.offset)
        u = np.array([math.cos(heading), math.sin(heading)])
        start = c - u * 0.03
        total = 0.03 + cand.distance
        n = int(total / self.speed / TIMESTEP)
        for k in range(n + 1):
            p = start + u * total * k / n
            d.mocap_pos[self._p] = [p[0], p[1], self.h]
            mujoco.mj_step(m, d)
        d.mocap_pos[self._p] = [5, 5, self.h]  # withdraw, let it settle
        for _ in range(int(0.4 / TIMESTEP)):
            mujoco.mj_step(m, d)
        q = d.qpos[self._bq:self._bq + 7]
        return float(q[0]), float(q[1]), yaw_from_mat(quat_to_mat(q[3:7]))


def plan_push(model: PushModel, block: tuple[float, float, float], goal: tuple[float, float, float],
              robot_pose: tuple | None = None, pose_fn=None, yaw_weight: float = 0.12, offsets=(-0.03, -0.015, 0.0, 0.015, 0.03),
              distances=(0.04, 0.08, 0.14), yaw_matters: bool = True,
              max_heading_change: float | None = None, pose_ok=None) -> tuple[PushCandidate, list[PushCandidate]]:
    """Evaluate all candidates with the planning model; return the best and the full ranked list.
    pose_ok(x, y, yaw): optional check that the robot can stand at a candidate's push pose (e.g. not on
    top of a fixture in a user's scene)."""
    bx, by, byaw = block
    cands = []
    for face in range(4):
        for off in offsets:
            c0, heading = model.geometry(bx, by, byaw, face, off)
            # skip pushes pointing away from the goal by more than 100 deg when far from it
            to_goal = math.atan2(goal[1] - by, goal[0] - bx)
            if math.hypot(goal[0] - bx, goal[1] - by) > 0.08 and abs(wrap_angle(heading - to_goal)) > math.radians(100):
                continue
            if max_heading_change is not None and robot_pose is not None and \
                    abs(wrap_angle(heading - robot_pose[2])) > max_heading_change:
                continue  # keep repositioning within what the gait does reliably
            if pose_ok is not None and pose_fn is not None and not pose_ok(*pose_fn((float(c0[0]), float(c0[1])), heading)):
                continue
            for dist in distances:
                cand = PushCandidate(face, off, dist, heading, (float(c0[0]), float(c0[1])))
                px, py, pyaw = model.predict(bx, by, byaw, cand)
                pos = math.hypot(px - goal[0], py - goal[1])
                yerr = abs(wrap_angle(pyaw - goal[2])) if yaw_matters else 0.0
                reposition = 0.0
                if robot_pose is not None and pose_fn is not None:  # cost of walking to this push pose
                    tx, ty, th = pose_fn(cand.contact_xy, heading)
                    reposition = 0.03 * abs(wrap_angle(th - robot_pose[2])) + 0.05 * math.hypot(tx - robot_pose[0], ty - robot_pose[1])
                cand.predicted = (px, py, pyaw)
                cand.cost = pos + yaw_weight * yerr + reposition
                cands.append(cand)
    if not cands and max_heading_change is not None:  # nothing within the heading window: relax it
        return plan_push(model, block, goal, robot_pose, pose_fn, yaw_weight, offsets, distances, yaw_matters, None, pose_ok)
    if not cands:
        raise ValueError("no feasible push candidate")
    cands.sort(key=lambda c: c.cost)
    return cands[0], cands
