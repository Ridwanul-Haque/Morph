"""Closed-loop push-to-goal for the mobile pusher.

Loop: measure block -> plan a push with the planning model -> drive around the block
to the pre-contact point -> push -> measure -> repeat. An optional `revise` callback
(Nemotron or rule) can adjust the planner's parameters after a bad push.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .geometry import wrap_angle
from .push_planner import PushModel, plan_push
from .robots.pusher import PusherAdapter
from .sim import PhysicalParams, SimulationUnstable
from .task import TaskSpec, evaluate

PUSH_REVISION_SPACE = {
    "distance_scale": {"range": [0.3, 1.5]},
    "yaw_weight": {"range": [0.02, 0.5]},
    "offset_max_m": {"range": [0.0, 0.035]},
}


@dataclass
class PushRunResult:
    outcome: object
    pushes: list = field(default_factory=list)
    revisions: list = field(default_factory=list)
    unstable: bool = False
    sim_time: float = 0.0
    wall_s: float = 0.0
    frames: list = field(default_factory=list)
    frame_info: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    stop_reason: str = ""
    other_contacts: list = field(default_factory=list)


def _fixed_direction_push(pm: PushModel, block, goal, dists):
    """Baseline: push through the face whose normal best opposes the goal direction, centred, no yaw reasoning."""
    bx, by, byaw = block
    to_goal = math.atan2(goal[1] - by, goal[0] - bx)
    best = None
    for face in range(4):
        c, heading = pm.geometry(bx, by, byaw, face, 0.0)
        if best is None or abs(wrap_angle(heading - to_goal)) < abs(wrap_angle(best[1] - to_goal)):
            best = (face, heading, c)
    from .push_planner import PushCandidate
    remaining = math.hypot(goal[0] - bx, goal[1] - by)
    d = min(max(dists), max(min(dists), remaining))
    cand = PushCandidate(best[0], 0.0, d, best[1], (float(best[2][0]), float(best[2][1])))
    cand.predicted = pm.predict(bx, by, byaw, cand)
    return cand


def run_pusher(spec: TaskSpec, block0, params: PhysicalParams, seed: int, base0=(0.3, -0.6, math.pi / 2),
               max_pushes: int = 12, render_every: int = 0, revise=None, log=print, mode: str = "mpc",
               camera: str = "overview", on_tick=None, adapter: PusherAdapter | None = None) -> PushRunResult:
    """mode: "mpc" (Morph: re-plan after every measured push), "fixed_direction" (baseline),
    "open_loop" (baseline: plan the whole sequence once on the nominal model, never re-measure)."""
    t0 = time.time()
    rb = adapter or PusherAdapter()
    rb.render_every = render_every
    rb.render_camera = camera
    rb.on_tick = on_tick
    rb.reset(seed, block0, params, base=base0)
    rb.env.set_goal_marker(spec.gx, spec.gy, spec.gyaw)
    pm = PushModel()  # nominal mass 0.2 kg / friction 0.6; the true values are randomized
    goal = (spec.gx, spec.gy, spec.gyaw)
    tol_p, tol_y = float(spec.goal.pos_tol_m.value), float(spec.goal.yaw_tol_rad.value)
    knobs = {"distance_scale": 1.0, "yaw_weight": 0.12, "offset_max_m": 0.6 * rb.block.half_x}  # stay on the face
    res = PushRunResult(outcome=None)
    open_plan = []
    if mode == "open_loop":  # roll the nominal model forward to build the whole plan up front
        sim_b = (block0[0], block0[1], block0[2])
        for _ in range(max_pushes):
            if math.hypot(sim_b[0] - goal[0], sim_b[1] - goal[1]) <= 0.6 * tol_p and abs(wrap_angle(sim_b[2] - goal[2])) <= 0.6 * tol_y:
                break
            om = knobs["offset_max_m"]
            c, _ = plan_push(pm, sim_b, goal, yaw_weight=knobs["yaw_weight"], offsets=(-om, -om / 2, 0.0, om / 2, om))
            open_plan.append(c)
            sim_b = c.predicted
    try:
        for i in range(max_pushes):
            bp = rb.env.block_pose()
            pe, ye = math.hypot(bp.x - goal[0], bp.y - goal[1]), abs(wrap_angle(bp.yaw - goal[2]))
            if mode == "open_loop":
                if i >= len(open_plan):
                    res.stop_reason = "open-loop plan finished"
                    break
                best = open_plan[i]
            else:
                if pe <= 0.6 * tol_p and ye <= 0.6 * tol_y:
                    res.stop_reason = "goal reached with margin"
                    break
                om = knobs["offset_max_m"]
                offs = tuple(sorted({-om, -om / 2, 0.0, om / 2, om}))
                dists = tuple(d * knobs["distance_scale"] for d in (0.03, 0.06, 0.12))
                if mode == "fixed_direction":
                    if pe <= 0.6 * tol_p:
                        res.stop_reason = "position reached (baseline ignores yaw)"
                        break
                    best = _fixed_direction_push(pm, (bp.x, bp.y, bp.yaw), goal, dists)
                else:
                    best, _ = plan_push(pm, (bp.x, bp.y, bp.yaw), goal, yaw_weight=knobs["yaw_weight"], offsets=offs, distances=dists)
            ev = rb.push(best.contact_xy, best.direction, best.distance)
            af = rb.env.block_pose()
            rec = {"i": i, "plan": best.as_dict(), "knobs": dict(knobs), "block_before": bp.as_dict(), "block_after": af.as_dict(),
                   "measured_move_cm": round(math.hypot(af.x - bp.x, af.y - bp.y) * 100, 2),
                   "measured_yaw_change_deg": round(math.degrees(wrap_angle(af.yaw - bp.yaw)), 1),
                   "model_error_cm": round(math.hypot(af.x - best.predicted[0], af.y - best.predicted[1]) * 100, 2),
                   "model_yaw_error_deg": round(math.degrees(abs(wrap_angle(af.yaw - best.predicted[2]))), 1),
                   "pos_err_cm": round(math.hypot(af.x - goal[0], af.y - goal[1]) * 100, 2),
                   "yaw_err_deg": round(math.degrees(abs(wrap_angle(af.yaw - goal[2]))), 1),
                   "contact": ev, "sim_t": round(rb.env.data.time, 2)}
            res.pushes.append(rec)
            log(f"  push {i}: face={best.face} off={best.offset:+.3f} d={best.distance:.3f} -> moved {rec['measured_move_cm']} cm "
                f"yaw {rec['measured_yaw_change_deg']:+} deg | goal err {rec['pos_err_cm']} cm {rec['yaw_err_deg']} deg | "
                f"model err {rec['model_error_cm']} cm | tip contact ticks {ev['tip_contact_ticks']}")
            if revise is not None:
                change = revise(rec, res.pushes, PUSH_REVISION_SPACE, dict(knobs))
                if change:
                    knobs.update(change["apply"])
                    res.revisions.append({"after_push": i, **change})
        else:
            res.stop_reason = "push budget exhausted"
    except SimulationUnstable as e:
        res.unstable = True
        res.stop_reason = str(e)
    res.sim_time = float(rb.env.data.time)
    res.wall_s = round(time.time() - t0, 1)
    res.frames = rb.frames
    res.frame_info = rb.frame_info
    res.other_contacts = sorted({c["geom"] for c in rb.contacts if c["geom"] != "pusher_tip"})
    res.outcome = evaluate(spec, rb.env.block_pose(), rb.env.block_speed(), released=True)
    if adapter is None:
        rb.env.close()
    return res


def rule_revise(rec: dict, history: list, space: dict, knobs: dict) -> dict | None:
    """No-LLM baseline revision: if the last push overshot or barely moved, rescale the push distance."""
    if len(history) < 2:
        return None
    prev = history[-2]
    if rec["pos_err_cm"] > prev["pos_err_cm"] + 0.5:  # got worse -> shorter pushes
        return {"apply": {"distance_scale": max(space["distance_scale"]["range"][0], knobs["distance_scale"] * 0.6)},
                "method": "rule", "reason": "error increased after the last push"}
    return None
