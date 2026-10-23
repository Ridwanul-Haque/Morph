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
    base_path = _drive_up_to_grip(rn, scene, hand, log) if rn.ix.base >= 0 else [None]
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
    geom = _route_grip(rn, hand, rec, grip_at, open_at, scratch)
    ready = _about_to_grip(rn, hand, rec, grip_at, geom, scratch)
    kept_held, flags = 0, [False] * len(qs)  # (the drive up: not about to grip)
    for k in range(len(rec)):
        if k % every and k not in ready:  # the moments just before its grip: every one of them
            continue
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
        flags.append(k in ready)
    if not qs:
        return None
    np.savez(Path(task_path).parent / "starts.npz", qpos=np.array(qs), ctrl=np.array(cs), t=np.array(ts, dtype=np.int32),
             ready=np.array(flags, dtype=bool))
    t = json.loads(Path(task_path).read_text())
    t["starts"] = {"file": "starts.npz", "share": share, "scaffold": True,
                   "from": "Morph's own route: " + ("driving up, " if len(base_path) > 1 else "")
                           + "its planned grip, carrying it to the goal" + (" and letting go" if open_at else "")}
    if any(flags):
        t["starts"]["ready"] = READY_SHARE
    if geom and not t.get("grasp"):  # the learner aims for the grip Morph's own route held it with (train_task)
        t["grasp"] = geom
    from .plan_check import require_grip
    note = require_grip(t)
    Path(task_path).write_text(json.dumps(t, indent=1))
    info = {"starts": len(qs), "holding": kept_held, "route_reached_goal": bool(run["success"]),
            "drive": len(base_path) - 1, "ready": int(sum(flags)), "grasp": bool(t.get("grasp"))}
    log(f"  practice starts near success: {len(qs)} moments along Morph's own route"
        + (f" ({info['drive']} on the drive up)" if info["drive"] else "")
        + f", {kept_held} with the {scene.object_name} in its grip"
        + (f", {info['ready']} with its open hand where Morph grips it (a {READY_SHARE:.0%} share of practice starts "
           "there every round, to practise closing)" if info["ready"] else "") + "; "
        + ("the route itself reaches the goal" if run["success"] else "the learner works out the rest"))
    if geom and t.get("grasp") is geom:
        log("  the learner aims for the grip Morph's own route used")
    if note:
        log(f"  {note}")
    return info


READY_SHARE = 0.3  # of the practice runs that start from Morph's route, how many start just before its grip
READY_GAP = 0.04  # m: the hand's point this close to where it grips the object, the gripper still open


def _route_grip(rn, hand: str, rec: list, grip_at, open_at, scratch) -> dict | None:
    """How Morph's own route held the object (contact_plan.grasp_geometry), measured once it really had it."""
    from .contact_plan import grasp_geometry
    if grip_at is None:
        return None
    end = open_at if open_at is not None else len(rec)
    for k in range(grip_at + 25, min(end, len(rec))):
        rn.d.qpos[:] = rec[k][0]
        rn.d.qvel[:] = 0.0
        mujoco.mj_forward(rn.m, rn.d)
        if contact_held(rn.m, rn.d, rn.ix) > 0.5:
            g = grasp_geometry(rn, hand)
            return g if g.get("fingers") and g.get("close") is not None else None
    return None


def _about_to_grip(rn, hand: str, rec: list, grip_at, geom, scratch) -> set[int]:
    """The route's moments just before it closes the gripper: the open hand where its grip will be. Practice starts
    there too, every round, so closing on the object gets practised on its own (the Stretch learned to drive up and
    rest its hand on top of the parcel, and never once had its fingers either side of it with the gripper open)."""
    if grip_at is None or not geom:
        return set()
    part = rn.parts[hand]
    point = np.array(geom["point"])
    ix = rn.ix
    out = set()
    for k in range(grip_at, -1, -1):
        if ix.grip_act >= 0 and ix.grip_close != ix.grip_open:  # (grip_at is where the closing has finished)
            if (rec[k][2][ix.grip_act] - ix.grip_open) / (ix.grip_close - ix.grip_open) > 0.2:
                continue
        scratch.qpos[:] = rec[k][0]
        mujoco.mj_kinematics(rn.m, scratch)
        R = scratch.xmat[rn.ix.block].reshape(3, 3)
        gap = np.linalg.norm(R.T @ (rn.tip(part, scratch) - scratch.xpos[rn.ix.block]) - point)
        if gap > READY_GAP:
            break
        out.add(k)
    return out


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


def _grip_spots(rn, scene, hand: str) -> list[tuple[float, tuple]]:
    """A wheeled base: where to set it down to grip the object from each of its sides, the hand level and pointing at
    it (a Stretch can't point its gripper down, and with its base still its hand only reaches along its arm's line, so
    a spot chosen to put its folded hand over the object left every grip out of reach). Per side: (IK error, pose),
    best first. The base's heading is tried all round; its position then follows from where the hand lands."""
    m, d, ix = rn.m, rn.d, rn.ix
    part = rn.parts[hand]
    rn.reset(0, scene.start, None)
    ox, oy, oyaw = (float(v) for v in scene.start)
    oz = float(d.xpos[ix.block][2])
    b = scene.block
    obj = np.array([ox, oy, oz])
    from .contact_plan import level_tilt
    pz = level_tilt(rn, part)
    c = math.sqrt(max(1.0 - pz * pz, 0.0))
    out = []
    for side in (0, 90, 180, 270):
        a = oyaw + math.radians(side)
        u = np.array([c * math.cos(a), c * math.sin(a), pz])
        across = np.array([-math.sin(a), math.cos(a), 0.0])
        ext = abs(b.half_x * math.cos(math.radians(side))) + abs(b.half_y * math.sin(math.radians(side)))
        flat = np.array([math.cos(a), math.sin(a), 0.0])
        pre, grab = obj - (ext + 0.08) * flat, obj + 0.3 * ext * flat  # as the side grasps go (contact_plan)
        best = None
        for k in range(24):
            yaw = k * math.tau / 24
            x, y = ox, oy
            ok = True
            for _ in range(4):  # move the base by however far the hand misses, and solve again
                _put_base(rn, (x, y, yaw))
                q, pe, ae = rn.ik(part, pre, point=u, across=across, q_start=rn.q(part), iters=120)
                rn.scratch.qpos[:] = rn.d.qpos
                rn._set(rn.scratch, part, q)
                mujoco.mj_kinematics(m, rn.scratch)
                miss = pre - rn.tip(part, rn.scratch)
                if np.hypot(miss[0], miss[1]) < 0.003:
                    break
                x, y = x + miss[0], y + miss[1]
            if math.hypot(x - ox, y - oy) < 0.12 or any(
                    f.lo[0] - 0.25 < x < f.hi[0] + 0.25 and f.lo[1] - 0.25 < y < f.hi[1] + 0.25 for f in scene.fixtures.values()):
                ok = False
            if not ok:
                continue
            _put_base(rn, (x, y, yaw))
            _, pe1, ae1 = rn.ik(part, pre, point=u, across=across, q_start=rn.q(part), iters=150)
            _, pe2, ae2 = rn.ik(part, grab, point=u, across=across, q_start=rn.q(part), iters=150)
            err = max(pe1, pe2) + 0.05 * max(ae1, ae2)
            if best is None or err < best[0]:
                best = (err, (x, y, yaw))
        if best is not None:
            out.append(best)
    rn.reset(0, scene.start, None)
    return sorted(out, key=lambda e: e[0])


def _drive_up_to_grip(rn, scene, hand: str, log=print, n_path: int = 4) -> list | None:
    """Where to set a wheeled base down for the pick: the spot from which Morph's grasp planning grips the object best,
    of those where the hand can reach it from one side (_grip_spots); else beside it as _drive_up does. With the points
    on the straight way there from where it starts."""
    from .contact_plan import plan_grasp
    home = list(rn.task["mount"])
    spots = [pose for err, pose in _grip_spots(rn, scene, hand) if err < 0.02]
    best = None
    for pose in spots:
        rn.task = dict(rn.task, mount=list(pose))
        g = plan_grasp(rn, scene, hand)
        if g and (best is None or g["score"] > best[0]):
            best = (g["score"], pose, g["max"])
    rn.task = dict(rn.task, mount=home)
    if best is None or best[0] <= 0:
        low = _lowest_grip(rn, scene, hand)
        top = 2 * scene.block.half_z
        log(f"  Morph couldn't grip the {scene.object_name} from anywhere it can set the base down"
            + (f": the point between its fingertips gets no lower than {low * 100:.1f} cm, and the {scene.object_name} is "
               f"only {top * 100:.1f} cm tall, so its fingers can't get around it" if low is not None and low > 0.6 * top
               else ""))
        return _drive_up(rn, scene, hand, n_path)
    rn.reset(0, scene.start, None)
    base, yaw0 = rn.d.xpos[rn.ix.base].copy(), float(home[2])
    x1, y1, yaw1 = best[1]
    log(f"  set down to grip it: the base {math.hypot(x1 - base[0], y1 - base[1]):.2f} m from where it starts; "
        f"Morph's grip held {best[0]:.0f}/{best[2]:.0f} in its grasp planning there")
    return [(base[0] + (x1 - base[0]) * f, base[1] + (y1 - base[1]) * f,
             yaw0 + math.remainder(yaw1 - yaw0, math.tau) * f) for f in (i / n_path for i in range(n_path + 1))]


def _lowest_grip(rn, scene, hand: str) -> float | None:
    """How low the point between the fingertips gets, set down beside the object and reaching for the floor."""
    spots = _grip_spots(rn, scene, hand)
    if not spots:
        return None
    part, low = rn.parts[hand], None
    for _, pose in spots:
        _put_base(rn, pose)
        target = np.array([*rn.d.xpos[rn.ix.block][:2], 0.0])
        q, _, _ = rn.ik(part, target, q_start=rn.q(part), iters=150)
        rn.scratch.qpos[:] = rn.d.qpos
        rn._set(rn.scratch, part, q)
        mujoco.mj_kinematics(rn.m, rn.scratch)
        z = float(rn.tip(part, rn.scratch)[2])
        low = z if low is None else min(low, z)
    rn.reset(0, scene.start, None)
    return low


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
