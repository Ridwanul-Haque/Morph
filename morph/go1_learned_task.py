"""Go1 push-to-goal using the learned walking policy.

Loop: measure block -> plan a push -> walk around the block to the push pose (learned
policy) -> switch to the three-leg foot push -> measure -> repeat.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .geometry import wrap_angle
from .go1_task import PUSH_LEG, push_pose
from .push_planner import PushModel, plan_push
from .robots.go1_control import Go1Controller
from .robots.go1_learned import PUSH_KP, PUSH_KV, LearnedGo1
from .robots.quadruped import QuadrupedAdapter
from .sim import PhysicalParams, SimulationUnstable
from .task import TaskSpec, evaluate

CLEAR_R = 0.55  # keep the trunk this far from the block centre while walking around it


@dataclass
class LearnedGo1Result:
    outcome: object
    pushes: list = field(default_factory=list)
    fell: bool = False
    unstable: bool = False
    sim_time: float = 0.0
    wall_s: float = 0.0
    walk_s: float = 0.0
    frames: list = field(default_factory=list)
    frame_info: list = field(default_factory=list)
    unintended_contacts: list = field(default_factory=list)
    stop_reason: str = ""


def _route(robot: LearnedGo1, target_xy: np.ndarray, block_xy: np.ndarray) -> list[np.ndarray]:
    """Waypoints on a circle around the block if the straight path would pass too close to it."""
    p, _, _ = robot.base()
    a, b = p[:2], target_xy
    ab = b - a
    t = float(np.clip(np.dot(block_xy - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1))
    if np.linalg.norm(a + t * ab - block_xy) > CLEAR_R * 0.9:
        return [b]
    a0 = math.atan2(a[1] - block_xy[1], a[0] - block_xy[0])
    a1 = math.atan2(b[1] - block_xy[1], b[0] - block_xy[0])
    da = wrap_angle(a1 - a0)
    n = max(1, int(abs(da) / 0.6))
    pts = [block_xy + CLEAR_R * np.array([math.cos(a0 + da * k / n), math.sin(a0 + da * k / n)]) for k in range(1, n)]
    return pts + [b]


def run_learned_go1_push(spec: TaskSpec, block0, params: PhysicalParams, seed: int, base0=(0.0, -0.7, math.pi / 2),
                         max_pushes: int = 14, render_every: int = 0, camera: str = "track", log=print,
                         robot: LearnedGo1 | None = None, on_tick=None) -> LearnedGo1Result:
    t0 = time.time()
    rb = robot or LearnedGo1()
    rb.reset(seed, block0, params, base_xy_yaw=base0)
    rb.env.set_goal_marker(spec.gx, spec.gy, spec.gyaw)
    qa = QuadrupedAdapter(model=rb.model, env=rb.env)
    res = LearnedGo1Result(outcome=None)
    contacts = []

    def tick(r, steplog):
        for c in steplog.robot_contacts:
            contacts.append((c.geom or c.body, c.normal_force))
        if on_tick is not None:
            on_tick(r)
        if render_every and r.ticks % render_every == 0:
            res.frames.append(r.env.render(camera))
            res.frame_info.append({"t": round(r.env.data.time, 2), "block": r.env.block_pose().as_dict(),
                                   "push": len(res.pushes), "mode": "learned walk"})
    rb.on_tick = tick
    PUSH_H = 0.025  # contact low on the face: the 5x5x6 cm block tips if pushed above ~4 cm
    pm = PushModel(push_height=PUSH_H)
    h = rb.block.half_x
    offsets = (-0.6 * h, -0.3 * h, 0.0, 0.3 * h, 0.6 * h)  # keep the contact on the block face
    goal = (spec.gx, spec.gy, spec.gyaw)
    tol_p, tol_y = float(spec.goal.pos_tol_m.value), float(spec.goal.yaw_tol_rad.value)
    try:
        rb.walk((0, 0, 0), 0.5)
        for i in range(max_pushes):
            bp = rb.env.block_pose()
            if bp.tilt > math.radians(10):  # no push primitive can right a tipped block: stop and say so
                res.stop_reason = f"block tipped over ({math.degrees(bp.tilt):.0f} deg); no recovery skill"
                break
            if math.hypot(bp.x - goal[0], bp.y - goal[1]) <= 0.9 * tol_p and abs(wrap_angle(bp.yaw - goal[2])) <= 0.9 * tol_y:
                res.stop_reason = "goal reached with margin"
                break
            p, th, _ = rb.base()
            best, _ = plan_push(pm, (bp.x, bp.y, bp.yaw), goal, robot_pose=(p[0], p[1], th), pose_fn=push_pose,
                                offsets=offsets)
            tx, ty, tth = push_pose(best.contact_xy, best.direction)
            tw = time.time()
            ok = True
            for wp in _route(rb, np.array([tx, ty]), np.array([bp.x, bp.y]))[:-1]:
                ok &= rb.go_to(wp[0], wp[1], math.atan2(ty - wp[1], tx - wp[0]), tol=0.08, yaw_tol=0.6, timeout=10)
            ok &= rb.go_to(tx, ty, tth, tol=0.025, yaw_tol=0.05, timeout=15)
            for _retry in range(2):  # re-approach: pushing from >3 cm off the planned pose tipped the block
                p, th, _ = rb.base()
                if math.hypot(p[0] - tx, p[1] - ty) < 0.03 or rb.fallen():
                    break
                ok &= rb.go_to(tx, ty, tth, tol=0.02, yaw_tol=0.05, timeout=8)
            res.walk_s += time.time() - tw
            if rb.fallen():
                res.fell, res.stop_reason = True, f"robot fell while walking to push {i}"
                break
            # re-plan from the (possibly nudged) block with the robot now in place
            bp2 = rb.env.block_pose()
            p, th, _ = rb.base()
            best, _ = plan_push(pm, (bp2.x, bp2.y, bp2.yaw), goal, robot_pose=(p[0], p[1], th), pose_fn=push_pose,
                                offsets=offsets, max_heading_change=math.radians(15))
            # hand over to the three-leg foot push (stiffer gains)
            rb._set_gains(PUSH_KP, PUSH_KV)
            rb.mode = "push"
            ctrl = Go1Controller(qa, render_every=render_every, camera=camera)

            def ptick(c, _i=i):
                if on_tick is not None:
                    on_tick(rb)
                if render_every and c.log.ticks % render_every == 0:
                    res.frame_info.append({"t": round(rb.env.data.time, 2), "block": rb.env.block_pose().as_dict(),
                                           "push": _i, "mode": "foot push"})
            ctrl.on_tick = ptick
            before = rb.env.block_pose()
            ev = ctrl.push(PUSH_LEG, np.array(best.contact_xy), best.direction, best.distance, push_height=PUSH_H)
            res.frames.extend(ctrl.log.frames)
            contacts.extend((c["geom"] or c["body"], c["force_n"]) for c in ctrl.log.block_contacts)
            after = rb.env.block_pose()
            rec = {"i": i, "plan": best.as_dict(), "reached_push_pose": bool(ok),
                   "base_error_cm": round(math.hypot(p[0] - tx, p[1] - ty) * 100, 1),
                   "block_before": before.as_dict(), "block_after": after.as_dict(),
                   "measured_move_cm": round(math.hypot(after.x - before.x, after.y - before.y) * 100, 2),
                   "measured_yaw_change_deg": round(math.degrees(wrap_angle(after.yaw - before.yaw)), 1),
                   "pos_err_cm": round(math.hypot(after.x - goal[0], after.y - goal[1]) * 100, 2),
                   "yaw_err_deg": round(math.degrees(abs(wrap_angle(after.yaw - goal[2]))), 1),
                   "block_tilt_deg": round(math.degrees(after.tilt), 1),
                   "contact": ev, "sim_t": round(rb.env.data.time, 1)}
            res.pushes.append(rec)
            log(f"  push {i}: walked to pose (err {rec['base_error_cm']} cm) -> moved {rec['measured_move_cm']} cm, "
                f"yaw {rec['measured_yaw_change_deg']:+} deg | goal err {rec['pos_err_cm']} cm / {rec['yaw_err_deg']} deg | "
                f"foot contact ticks {ev['foot_contact_ticks']} | block tilt {rec['block_tilt_deg']} deg")
            if ctrl.log.fell or rb.fallen():
                res.fell, res.stop_reason = True, f"robot fell during push {i}"
                break
            rb.last_act[:] = 0
            rb.walk((0, 0, 0), 0.6)  # learned policy takes back over and re-balances
            if rb.fallen():
                res.fell, res.stop_reason = True, f"robot fell after push {i}"
                break
        else:
            res.stop_reason = "push budget exhausted"
        if not res.fell:
            rb.walk((0, 0, 0), 0.5)
    except SimulationUnstable as e:
        res.unstable, res.stop_reason = True, str(e)
    res.sim_time = float(rb.env.data.time)
    res.wall_s = round(time.time() - t0, 1)
    res.unintended_contacts = sorted({g for g, _ in contacts if g != PUSH_LEG})
    res.outcome = evaluate(spec, rb.env.block_pose(), rb.env.block_speed(), released=True)
    if res.fell:
        res.outcome.success = False
        res.outcome.reasons.append("robot fell")
    return res
