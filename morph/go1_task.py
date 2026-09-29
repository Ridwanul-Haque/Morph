"""Go1 push-to-goal using the hand-written crawl gait (kept as the baseline).

Loop: measure block -> plan a push -> walk (turn, forward, turn) to the push pose ->
push with the front-right foot -> measure -> repeat until the goal test passes or the
push budget runs out.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .geometry import rot2, wrap_angle
from .push_planner import PushModel, plan_push
from .robots.go1_control import Go1Controller
from .robots.quadruped import QuadrupedAdapter
from .sim import PhysicalParams, SimulationUnstable
from .task import TaskSpec, evaluate

PUSH_LEG = "FR"
FOOT_IN_TRUNK = np.array([0.24, -0.167])  # pre-contact foot position in the trunk frame


@dataclass
class Go1Result:
    outcome: object
    pushes: list = field(default_factory=list)
    fell: bool = False
    unstable: bool = False
    sim_time: float = 0.0
    wall_s: float = 0.0
    frames: list = field(default_factory=list)
    frame_info: list = field(default_factory=list)
    unintended_contacts: list = field(default_factory=list)
    stop_reason: str = ""


def push_pose(contact_xy, heading: float) -> tuple[float, float, float]:
    u = np.array([math.cos(heading), math.sin(heading)])
    pre = np.array(contact_xy) - u * 0.05
    base = pre - rot2(heading) @ FOOT_IN_TRUNK
    return float(base[0]), float(base[1]), heading


def _turn(ctrl: Go1Controller, yaw: float) -> bool:
    b = ctrl.qa.base()
    d = wrap_angle(yaw - b.yaw)
    if abs(d) > math.radians(100):  # split large in-place turns (validated up to ~115 deg)
        if not ctrl.walk_to(b.pos[0], b.pos[1], b.yaw + d / 2, tol=0.05, yaw_tol=0.1, max_cycles=40) and ctrl.log.fell:
            return False
        b = ctrl.qa.base()
    return ctrl.walk_to(b.pos[0], b.pos[1], yaw, tol=0.05, yaw_tol=0.08, max_cycles=40) or not ctrl.log.fell


def go_to(ctrl: Go1Controller, x: float, y: float, yaw: float) -> bool:
    """Turn to face the target (or its reverse if closer), walk straight, turn to the final heading.

    The crawl gait walks forward/backward and turns reliably but does not side-step, so
    lateral error up to 5 cm is accepted; the push path is world-referenced.
    """
    b = ctrl.qa.base()
    dx, dy = x - b.pos[0], y - b.pos[1]
    dist = math.hypot(dx, dy)
    if dist > 0.06:
        h = math.atan2(dy, dx)
        rel = wrap_angle(h - b.yaw)
        if abs(rel) > math.radians(30):
            h_travel = h if abs(rel) <= math.radians(110) else wrap_angle(h + math.pi)  # walk backwards
            if not _turn(ctrl, h_travel):
                return False
        else:
            h_travel = b.yaw
        if not ctrl.walk_to(x, y, h_travel, tol=0.025, yaw_tol=0.1, max_cycles=60) and ctrl.log.fell:
            return False
    if not _turn(ctrl, yaw):
        return False
    return ctrl.walk_to(x, y, yaw, tol=0.03, yaw_tol=0.06, max_cycles=15) or not ctrl.log.fell


def run_go1_push(spec: TaskSpec, block0, params: PhysicalParams, seed: int, base0=(0.0, 0.0, 0.0),
                 max_pushes: int = 12, render_every: int = 0, planner_params=None, log=print,
                 revise=None, camera: str = "overview", on_tick=None, qa: QuadrupedAdapter | None = None,
                 stop_after_pushes: int | None = None) -> Go1Result:
    t0 = time.time()
    own = qa is None
    qa = qa or QuadrupedAdapter()
    qa.reset(seed, block0, params, base_xy_yaw=base0)
    qa.env.set_goal_marker(spec.gx, spec.gy, spec.gyaw)
    ctrl = Go1Controller(qa, render_every=render_every, camera=camera)
    res = Go1Result(outcome=None)
    frame_info = []

    def tick_hook(c):
        if on_tick is not None:
            on_tick(c)
        if render_every and c.log.ticks % render_every == 0:
            bp = qa.env.block_pose()
            frame_info.append({"t": round(qa.env.data.time, 2), "block": bp.as_dict(), "push": len(res.pushes)})
    ctrl.on_tick = tick_hook
    pm = PushModel(**(planner_params or {}))
    goal = (spec.gx, spec.gy, spec.gyaw)
    tol_p, tol_y = float(spec.goal.pos_tol_m.value), float(spec.goal.yaw_tol_rad.value)
    dist_scale = 1.0
    try:
        ctrl.hold(0.5)
        for i in range(max_pushes):
            bp = qa.env.block_pose()
            pos_err = math.hypot(bp.x - goal[0], bp.y - goal[1])
            yerr = abs(wrap_angle(bp.yaw - goal[2]))
            if pos_err <= tol_p * 0.7 and yerr <= tol_y * 0.7:
                res.stop_reason = "goal reached (with margin)"
                break
            base = qa.base()
            best, ranked = plan_push(pm, (bp.x, bp.y, bp.yaw), goal, robot_pose=(base.pos[0], base.pos[1], base.yaw),
                                     pose_fn=push_pose, distances=tuple(d * dist_scale for d in (0.04, 0.08, 0.14)),
                                     max_heading_change=math.radians(40))
            bx, by, byaw = push_pose(best.contact_xy, best.direction)
            ok = go_to(ctrl, bx, by, byaw)
            if ctrl.log.fell:
                res.fell = True
                res.stop_reason = f"robot fell while repositioning for push {i}"
                break
            # re-measure: walking may have disturbed the block (logged as unintended contact)
            bp2 = qa.env.block_pose()
            moved = math.hypot(bp2.x - bp.x, bp2.y - bp.y)
            if moved > 0.01:
                base = qa.base()
                best, ranked = plan_push(pm, (bp2.x, bp2.y, bp2.yaw), goal, robot_pose=(base.pos[0], base.pos[1], base.yaw),
                                         pose_fn=push_pose, distances=tuple(d * dist_scale for d in (0.04, 0.08, 0.14)),
                                     max_heading_change=math.radians(40))
            before = qa.env.block_pose()
            ev = ctrl.push(PUSH_LEG, np.array(best.contact_xy), best.direction, best.distance)
            after = qa.env.block_pose()
            rec = {"i": i, "plan": best.as_dict(), "reached_push_pose": bool(ok),
                   "block_before": before.as_dict(), "block_after": after.as_dict(),
                   "measured_move_cm": round(math.hypot(after.x - before.x, after.y - before.y) * 100, 2),
                   "measured_yaw_change_deg": round(math.degrees(wrap_angle(after.yaw - before.yaw)), 1),
                   "predicted_vs_measured_cm": round(math.hypot(after.x - best.predicted[0], after.y - best.predicted[1]) * 100, 2),
                   "pos_err_cm": round(math.hypot(after.x - goal[0], after.y - goal[1]) * 100, 2),
                   "yaw_err_deg": round(math.degrees(abs(wrap_angle(after.yaw - goal[2]))), 1),
                   "contact": ev, "disturbed_while_walking_cm": round(moved * 100, 2), "sim_t": round(qa.env.data.time, 1)}
            res.pushes.append(rec)
            log(f"  push {i}: face={best.face} off={best.offset:+.3f} d={best.distance:.2f} -> moved {rec['measured_move_cm']} cm, "
                f"yaw {rec['measured_yaw_change_deg']:+} deg; goal err {rec['pos_err_cm']} cm / {rec['yaw_err_deg']} deg; "
                f"foot contact ticks {ev['foot_contact_ticks']}")
            if ctrl.log.fell:
                res.fell = True
                res.stop_reason = f"robot fell during push {i}"
                break
            if stop_after_pushes is not None and len(res.pushes) >= stop_after_pushes:
                res.stop_reason = f"stopped after {stop_after_pushes} push(es) as requested"
                break
            if revise is not None:
                change = revise(rec, res.pushes)
                if change and "distance_scale" in change:
                    dist_scale = float(change["distance_scale"])
        else:
            res.stop_reason = "push budget exhausted"
        ctrl.hold(0.5)
    except SimulationUnstable as e:
        res.unstable = True
        res.stop_reason = str(e)
    res.sim_time = float(qa.env.data.time)
    res.wall_s = round(time.time() - t0, 1)
    res.frames = ctrl.log.frames
    res.frame_info = frame_info
    res.unintended_contacts = sorted({(c["geom"] or c["body"]) for c in ctrl.log.block_contacts if c["geom"] != PUSH_LEG})
    res.outcome = evaluate(spec, qa.env.block_pose(), qa.env.block_speed(), released=True)
    if res.fell:
        res.outcome.reasons.append("robot fell")
        res.outcome.success = False
    if own:
        qa.env.close()
    return res
