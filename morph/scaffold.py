"""Practice starts near success, with no working program to copy (a reverse curriculum, Florensa et al. 2017).

When Nemotron's programs didn't work, learning starts from nothing: every practice run begins with the robot at home and
the object on the floor, and a robot that has to drive, reach, grip, lift and carry almost never stumbles on the end by
chance (the Stretch never got near the parcel; the toss never got a grip), so it has nothing to learn from. Morph can
build the situations on the way itself, from its own tools: the grip its grasp planning found, its IK, and the physics to
check them. It runs that route once (pick with the planned grip, carry to the goal, lower, let go) and keeps real states
along it: the object really in the fingers after the grip, nothing overlapping. A wheeled base is first set down beside
the object (and at points on the way there). Practice then starts a share of its runs from those states, late ones first
(already holding it over the bin), earlier ones in later rounds (learn_task._easy_first). These are practice starts only:
every try that counts starts from the real beginning.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from .goals import GoalSpec, lets_go, placement
from .task_model import contact_held, load_task


def scaffold_starts(task_path: Path, pj, goal: GoalSpec, log=print, every: int = 4, share: float = 0.4) -> dict | None:
    """Writes starts.npz (qpos, ctrl, t) from Morph's own route and points the task at it. None if this robot has no
    gripping hand or the goal doesn't need the object held."""
    from .body_programs import (BodyRunner, ContactPlanner, ProgramError, calibrate_grip, check_body_program, needs_hold,
                                run_body_program)
    from .programs import scene_names
    task = load_task(task_path)
    try:
        rn = BodyRunner(task)
    except Exception:  # noqa: BLE001
        return None
    hand = next((n for n, p in rn.parts.items() if p.can_grip), None)
    if hand is None or not needs_hold(goal, pj.scene):
        # no gripping hand Morph can work (or nothing to hold): a wheeled base can still be set down along the drive up
        # to the object, its furthest-reaching part beside it (the Stretch never got within 50 cm of the parcel)
        return _drive_only(task_path, rn, pj.scene, share, log) if rn.ix.base >= 0 else None
    if not task.get("gripper", {}).get("calibrated"):  # the same firm-enough grip the learner will have
        k = calibrate_grip(rn, pj.scene, log=lambda s: None)
        t = json.loads(Path(task_path).read_text())
        t["gripper"]["gain_scale"] = float(t["gripper"].get("gain_scale", 1.0)) * k
        t["gripper"]["calibrated"] = True
        Path(task_path).write_text(json.dumps(t, indent=1))
        task = load_task(task_path)
        rn = BodyRunner(task)
    scene = pj.scene
    p = placement(goal, scene)
    top = max([f.top_z for f in scene.fixtures.values()] + [2 * scene.block.half_z])
    names = scene_names(scene, top + 0.12, 0.01, (p["x"], p["y"]) if p.get("has_place") else None,
                        p["support_z"] + scene.block.half_z)
    route = [{"do": "pick", "part": hand}, {"do": "move", "part": hand, "z": "carry_z", "speed": "normal"}]
    if p.get("has_place"):
        route += [{"do": "move", "part": hand, "x": "goal_x", "y": "goal_y", "z": "carry_z", "speed": "normal"},
                  {"do": "move", "part": hand, "x": "goal_x", "y": "goal_y", "z": "goal_z + 0.03", "speed": "slow"}]
        if lets_go(goal) and not any(c.when == "release" for c in goal.conditions):
            route += [{"do": "grip", "how": "open"}]
    base_path = _drive_up(rn, scene, hand) if rn.ix.base >= 0 else [None]
    if base_path is None:
        return None
    if base_path[-1] is not None:  # the route (and its grasp planning, which resets the robot per try) starts there
        rn.task = dict(rn.task, mount=list(base_path[-1]))
    planner = ContactPlanner(rn, scene, log=lambda s: None)
    try:
        steps = check_body_program(route, rn.parts, [], set(names), task.get("fixtures"), planner)
    except ProgramError as e:
        log(f"  (no practice starts from Morph's own route: {e})")
        return None
    from .project import _params
    qs, cs, ts = [], [], []
    m = rn.m
    # the drive there, for a wheeled base: the robot set down at points along the way (each a start of its own)
    for k, pose in enumerate(base_path[:-1]):
        rn.reset(0, scene.start, _params(0))
        _put_base(rn, pose)
        qs.append(rn.d.qpos.copy())
        cs.append(rn.ctrl.copy())
        ts.append(k)
    rn.reset(0, scene.start, _params(0))
    rec, marks = [], []
    run = run_body_program(rn, steps, goal, scene, names=names, record=rec,
                           on_step=lambda i, d_: marks.append((len(rec), steps[i - 1])))
    grip_at = next((n for n, st in marks if st["do"] == "grip" and st["how"] == "close"), None)
    open_at = next((n for n, st in marks if st["do"] == "grip" and st["how"] == "open" and n > (grip_at or 0)), None)
    scratch = mujoco.MjData(m)
    kept_held = 0
    for k in range(0, len(rec), every):
        q, _, c = rec[k]
        if grip_at is not None and k > grip_at + 25 and (open_at is None or k < open_at):
            scratch.qpos[:] = q
            mujoco.mj_forward(m, scratch)
            if contact_held(m, scratch, rn.ix) < 0.5:
                continue  # it slipped here: not a state to practise from
            kept_held += 1
        qs.append(q.copy())
        cs.append(c.copy())
        ts.append(len(base_path) + k)
    if not qs:
        return None
    np.savez(Path(task_path).parent / "starts.npz", qpos=np.array(qs), ctrl=np.array(cs), t=np.array(ts, dtype=np.int32))
    t = json.loads(Path(task_path).read_text())
    t["starts"] = {"file": "starts.npz", "share": share, "scaffold": True,
                   "from": "Morph's own route: " + ("driving up, " if len(base_path) > 1 else "")
                           + "its planned grip, carrying it to the goal" + (" and letting go" if open_at else "")}
    Path(task_path).write_text(json.dumps(t, indent=1))
    info = {"starts": len(qs), "holding": kept_held, "route_reached_goal": bool(run["success"]),
            "drive": len(base_path) - 1}
    log(f"  practice starts near success: {len(qs)} moments along Morph's own route"
        + (f" ({info['drive']} on the drive up)" if info["drive"] else "")
        + f", {kept_held} with the {scene.object_name} in its grip; "
        + ("the route itself reaches the goal" if run["success"] else "the learner works out the rest"))
    return info


def _drive_only(task_path: Path, rn, scene, share: float, log) -> dict | None:
    from .project import _params
    if not rn.parts:
        return None
    part = max(rn.parts, key=lambda n: (len(rn.parts[n].joints), n))  # its most capable end part (a gripper's tips)
    path = _drive_up(rn, scene, part, n_path=6)
    if path is None:
        return None
    qs, cs = [], []
    for pose in path:
        rn.reset(0, scene.start, _params(0))
        _put_base(rn, pose)
        qs.append(rn.d.qpos.copy())
        cs.append(rn.ctrl.copy())
    # and, beside it, that part lowered to just above the object (whatever of its joints Morph's IK can drive)
    p_ = rn.parts[part]
    top = float(rn.d.xpos[rn.ix.block][2] + scene.block.half_z)
    q, err, _ = rn.ik(p_, np.array([*rn.d.xpos[rn.ix.block][:2], top + 0.04]))
    if err < 0.08:
        rn.command(p_, q)
        for _ in range(int(1.0 / rn.env.control_dt)):  # let the joints get there (the base stays put)
            rn.env.advance(rn.ctrl)
        if np.linalg.norm(rn.d.xpos[rn.ix.block][:2] - np.array(scene.start[:2])) < 0.01:  # it didn't knock it
            qs.append(rn.d.qpos.copy())
            cs.append(rn.ctrl.copy())
    np.savez(Path(task_path).parent / "starts.npz", qpos=np.array(qs), ctrl=np.array(cs),
             t=np.arange(len(qs), dtype=np.int32))
    t = json.loads(Path(task_path).read_text())
    t["starts"] = {"file": "starts.npz", "share": share, "scaffold": True,
                   "from": f"Morph setting the base down along the drive up to the {scene.object_name}"}
    Path(task_path).write_text(json.dumps(t, indent=1))
    log(f"  practice starts near success: the base set down at {len(qs)} points on the way to the {scene.object_name}, "
        f"the last with its {part} beside it (Morph can't work this robot's gripper itself, so the rest is learned)")
    return {"starts": len(qs), "holding": 0, "route_reached_goal": False, "drive": len(qs) - 1}


def _put_base(rn, pose) -> None:
    x, y, yaw = pose
    m, d, ix = rn.m, rn.d, rn.ix
    d.qpos[ix.base_q:ix.base_q + 7] = [x, y, rn.task.get("base_z", 0.0), math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)


def _drive_up(rn, scene, hand: str, n_path: int = 4) -> list | None:
    """A wheeled base: where to set it down so its hand is right beside the object (its hand where it is at home, moved
    over the object), facing whichever way needs the least turning and keeps clear of the fixtures, and points on the
    straight way there from where it starts. None if it can't get its hand to the object anywhere."""
    if rn.task.get("base_z", 0.0) > 0.1:
        return None  # legs: not a base to set down beside things
    m, d, ix = rn.m, rn.d, rn.ix
    rn.reset(0, scene.start, None)
    base = d.xpos[ix.base].copy()
    yaw0 = float(rn.task["mount"][2])
    off = rn.tip(rn.parts[hand]) - base  # the hand relative to the base, at home, in the world (base turned yaw0)
    c, s = math.cos(-yaw0), math.sin(-yaw0)
    local = np.array([c * off[0] - s * off[1], s * off[0] + c * off[1]])
    obj = np.array(scene.start[:2], dtype=float)
    best = None
    for k in range(24):
        yaw = yaw0 + k * math.tau / 24
        c, s = math.cos(yaw), math.sin(yaw)
        xy = obj - np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]])
        if math.hypot(*(xy - obj)) < 0.12 or any(
                f.lo[0] - 0.25 < xy[0] < f.hi[0] + 0.25 and f.lo[1] - 0.25 < xy[1] < f.hi[1] + 0.25
                for f in scene.fixtures.values()):
            continue
        cost = math.hypot(*(xy - base[:2])) + 0.2 * abs(math.remainder(yaw - yaw0, math.tau))
        if best is None or cost < best[0]:
            best = (cost, float(xy[0]), float(xy[1]), yaw)
    if best is None:
        return None
    _, x1, y1, yaw1 = best
    path = []
    for i in range(n_path + 1):
        f = i / n_path
        path.append((base[0] + (x1 - base[0]) * f, base[1] + (y1 - base[1]) * f,
                     yaw0 + math.remainder(yaw1 - yaw0, math.tau) * f))
    return path
