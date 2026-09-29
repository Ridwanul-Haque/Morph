"""Adapter + controller for the mobile pusher robot.

It moves only through its three force-limited joints (slide x, slide y, hinge yaw),
and the block only moves when the pusher tip hits it.
"""

from __future__ import annotations

import math

import numpy as np

from ..geometry import rot2, wrap_angle
from ..scene import PUSHER_TIP_X, BlockSpec, build_pusher_model
from ..sim import SimulationEnvironment
from .base import Capabilities

SPEED = 0.12  # m/s travel
PUSH_SPEED = 0.06  # m/s while in contact (same as the planning model)
CLEAR_R = 0.20  # detour radius around the block centre


class PusherAdapter:
    kind = "mobile_pusher"

    def __init__(self, block: BlockSpec | None = None):
        self.block = block or BlockSpec()
        self.model = build_pusher_model(self.block)
        self.env = SimulationEnvironment(self.model)
        self._q = [self.model.joint(j).qposadr[0] for j in ("px", "py", "pyaw")]
        self.env.track_extra = lambda: self.env.data.qpos[self._q[:2]]
        self.contacts: list = []
        self.frames: list = []
        self.frame_info: list = []
        self.pushes_done = 0
        self.render_every = 0
        self.render_camera = "overview"
        self.on_tick = None
        self.ticks = 0

    def reset(self, seed: int, block_xy_yaw, params, base=(0.0, -0.6, math.pi / 2)) -> None:
        self.env.reset(seed, block_xy_yaw, params, robot_qpos={"px": base[0], "py": base[1], "pyaw": base[2]})
        self.env.data.ctrl[:] = base
        self.contacts, self.frames, self.frame_info, self.ticks, self.pushes_done = [], [], [], 0, 0

    def pose(self) -> np.ndarray:
        return self.env.data.qpos[self._q].copy()

    def tip(self) -> np.ndarray:
        x, y, th = self.pose()
        return np.array([x, y]) + rot2(th) @ np.array([PUSHER_TIP_X, 0.0])

    def _tick(self, cmd: np.ndarray) -> None:
        log = self.env.advance(cmd)
        for c in log.robot_contacts:
            self.contacts.append({"t": round(c.time, 3), "geom": c.geom, "force_n": round(c.normal_force, 3)})
        self.ticks += 1
        if self.on_tick is not None:
            self.on_tick(self)
        if self.render_every and self.ticks % self.render_every == 0:
            self.frames.append(self.env.render(self.render_camera))
            self.frame_info.append({"t": round(self.env.data.time, 2), "block": self.env.block_pose().as_dict(),
                                    "push": self.pushes_done})

    def move_tip(self, target_xy: np.ndarray, heading: float, speed: float = SPEED, tol: float = 0.004,
                 timeout: float = 8.0) -> bool:
        """Drive the tip to a world point with the given heading (lead control on the base pose)."""
        dt = self.env.control_dt
        for _ in range(int(timeout / dt)):
            x, y, th = self.pose()
            tip = self.tip()
            err = target_xy - tip
            yerr = wrap_angle(heading - th)
            if np.linalg.norm(err) < tol and abs(yerr) < 0.02:
                return True
            step = err * min(1.0, speed * dt * 3 / (np.linalg.norm(err) + 1e-9))
            new_th = th + float(np.clip(yerr, -1.5 * dt * 3, 1.5 * dt * 3))
            new_tip = tip + step
            base = new_tip - rot2(new_th) @ np.array([PUSHER_TIP_X, 0.0])
            self._tick(np.array([base[0], base[1], new_th]))
        return False

    def travel_to(self, target_xy: np.ndarray, heading: float, block_xy: np.ndarray) -> bool:
        """Go around the block (tip stays >= CLEAR_R from its centre) to a pre-contact point."""
        tip = self.tip()
        a0 = math.atan2(tip[1] - block_xy[1], tip[0] - block_xy[0])
        a1 = math.atan2(target_xy[1] - block_xy[1], target_xy[0] - block_xy[0])
        if np.linalg.norm(tip - block_xy) < CLEAR_R:  # back away radially first
            self.move_tip(block_xy + CLEAR_R * np.array([math.cos(a0), math.sin(a0)]), self.pose()[2])
        da = wrap_angle(a1 - a0)
        n = max(1, int(abs(da) / 0.35))
        for k in range(1, n + 1):
            a = a0 + da * k / n
            h = self.pose()[2] + wrap_angle(heading - self.pose()[2]) * k / n
            self.move_tip(block_xy + CLEAR_R * np.array([math.cos(a), math.sin(a)]), h, tol=0.01)
        return self.move_tip(target_xy, heading)

    def push(self, contact_xy, heading: float, distance: float) -> dict:
        u = np.array([math.cos(heading), math.sin(heading)])
        c = np.array(contact_xy)
        n0 = len(self.contacts)
        block = self.env.block_pose()
        self.travel_to(c - u * 0.05, heading, np.array([block.x, block.y]))
        self.move_tip(c + u * distance, heading, speed=PUSH_SPEED, timeout=10.0)
        self.move_tip(c + u * (distance - 0.06), heading)  # retract
        self._settle(0.3)
        self.pushes_done += 1
        tip_c = [x for x in self.contacts[n0:] if x["geom"] == "pusher_tip"]
        return {"tip_contact_ticks": len({x["t"] for x in tip_c}), "max_force_n": max([x["force_n"] for x in tip_c], default=0.0),
                "other_contacts": sorted({x["geom"] for x in self.contacts[n0:] if x["geom"] != "pusher_tip"})}

    def _settle(self, s: float) -> None:
        cmd = self.env.data.ctrl.copy()
        for _ in range(int(s / self.env.control_dt)):
            self._tick(cmd)

    def capabilities(self) -> Capabilities:
        return Capabilities(
            robot="mobile_pusher", embodiment="mobile_pusher",
            model_source="generated MJCF (morph/scene.py): holonomic base, slide x/y + hinge yaw, 0.023 m pusher tip",
            base_mobility="holonomic_planar", end_effectors=["pusher_tip"], contact_modes=["push"], prehensile=False,
            object_workspace={"type": "floor_reachable_by_driving", "object_height_range_m": [0.02, 0.09]},
            max_payload_kg=0.0, max_contact_force_n=30.0, can_set_planar_yaw=True, can_flip_object=False,
            action_primitives=["travel_to(tip xy, heading) around the object", "push(contact, heading, distance)"],
            modeled_limits={"base_force_n": 30, "yaw_torque_nm": 6, "tip_radius_m": 0.023},
            validation={"push": "closed-loop runs in morph/pusher_task.py; held-out eval in artifacts/evidence"},
            notes=["Secondary non-prehensile target. Generic design, not a model of a specific product."])
