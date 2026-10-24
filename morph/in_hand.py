"""Goals that start with the object in the hand (GoalSpec.start == "in_hand"): spinning a pencil, rolling a ball in
the palm. Morph puts it there before every try instead of on the floor.

  setup(task, goal)   turns a hand robot palm up and raises it (mount_pose), then drops the object onto the palm or
                      the fingers and closes some of them, trying grips until one holds it: the robot alone carries it,
                      it has come to rest, it already meets the goal's "the whole time" conditions (no thumb on it,
                      for "without the thumb"), and it keeps doing so for 3 s untouched in both the accurate and the
                      learner's physics. The grip found becomes the task's start (start_state) and its resting command
                      (q0), so a policy that does nothing keeps holding it; the same grip with the object laid a little
                      differently gives more starting poses (start_states), for practice and for the final tries.
"""

from __future__ import annotations

import itertools
import math

import mujoco
import numpy as np

from .fingers import hands

SETTLE_S = 1.5
CLOSURES = (0.3, 0.5, 0.7)


class NoGrip(ValueError):
    pass


def _home_data(m: mujoco.MjModel, home: dict, prefix: str = "") -> mujoco.MjData:
    d = mujoco.MjData(m)
    for n, v in home.items():
        try:
            d.qpos[m.jnt_qposadr[m.joint(prefix + n).id]] = v
        except KeyError:
            pass
    mujoco.mj_kinematics(m, d)
    return d


def _curled(m: mujoco.MjModel, d: mujoco.MjData, chains) -> np.ndarray:
    """The joint angles with every finger joint at the end of its range farther from where it is now (bent)."""
    q = d.qpos.copy()
    for chain in chains:
        for b in chain:
            for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
                if m.jnt_limited[j] and int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE):
                    lo, hi = m.jnt_range[j]
                    a = m.jnt_qposadr[j]
                    q[a] = hi if abs(hi - q[a]) > abs(lo - q[a]) else lo
    return q


def palm_up_pose(robot_xml: str, home: dict, at_xy) -> dict:
    """Where to bolt a hand so its fingers curl upwards, 30 cm above at_xy: {"pos", "quat"} for the task's mount."""
    m = mujoco.MjModel.from_xml_path(robot_xml)
    h = hands(m)
    if not h:
        raise NoGrip("it has no hand with fingers")
    d = _home_data(m, home)
    chains = list(h[0]["fingers"].values())
    tips0 = np.array([d.xpos[c[-1]] for c in chains])
    d2 = mujoco.MjData(m)
    d2.qpos[:] = _curled(m, d, chains)
    mujoco.mj_kinematics(m, d2)
    tips1 = np.array([d2.xpos[c[-1]] for c in chains])
    palm = d.xpos[h[0]["palm"]].copy()
    f = tips0.mean(0) - palm  # the way the fingers point
    f /= np.linalg.norm(f) + 1e-9
    n = (tips1 - tips0).mean(0)  # the way they curl, leaving out that they also pull back towards the palm
    n -= (n @ f) * f
    n /= np.linalg.norm(n) + 1e-9
    R = np.column_stack([f, np.cross(n, f), n]).T  # fingers -> +x, curl -> +z (up)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())
    t = np.array([at_xy[0], at_xy[1], 0.3]) - R @ palm
    low = min((R @ d.geom_xpos[g] + t)[2] - m.geom_rbound[g] for g in range(m.ngeom) if m.geom_bodyid[g] > 0)
    t[2] += max(0.0, 0.1 - low)
    return {"pos": [float(v) for v in t], "quat": [float(v) for v in quat]}


def _actuator_joints(m: mujoco.MjModel, a: int) -> list[int]:
    if int(m.actuator_trntype[a]) in (int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)):
        return [int(m.actuator_trnid[a, 0])]
    if int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_TENDON):  # Shadow's coupled finger joints
        t = int(m.actuator_trnid[a, 0])
        return [int(m.wrap_objid[w]) for w in range(m.tendon_adr[t], m.tendon_adr[t] + m.tendon_num[t])
                if int(m.wrap_type[w]) == int(mujoco.mjtWrap.mjWRAP_JOINT)]
    return []


def _finger_actuators(m: mujoco.MjModel, fingers: dict) -> dict[str, list[int]]:
    body_finger = {b: lab for lab, chain in fingers.items() for b in chain}
    out: dict[str, list[int]] = {}
    for a in range(m.nu):
        labs = {body_finger.get(int(m.jnt_bodyid[j])) for j in _actuator_joints(m, a)} - {None}
        if len(labs) == 1:
            out.setdefault(labs.pop(), []).append(a)
    return out


def _closed(m: mujoco.MjModel, a: int, now: float) -> float:
    """Which end of an actuator's range bends its finger: the one farther from where it is. Sideways joints that sit
    mid-range (spreading the fingers) stay where they are."""
    lo, hi = m.actuator_ctrlrange[a]
    if abs(abs(hi - now) - abs(lo - now)) < 0.2 * (hi - lo):
        return now
    return float(hi if abs(hi - now) > abs(lo - now) else lo)


def setup(task: dict, goal, log=print) -> dict:
    """Mutates task (see the module's docstring). Raises NoGrip with the reason when no grip holds it."""
    from .goal_terms import MOTION, WINDOW_FUNCS, condition_violation, uses
    from .task_model import build_task_model, contact_held, contact_touch, indices, measure
    if not hands(mujoco.MjModel.from_xml_path(task["robot_xml"])):
        raise NoGrip(f"{task['robot']} has no hand with fingers to hold it in")
    if task.get("floating"):
        raise NoGrip("starting with the object in the hand is only set up for hands bolted in place")
    task["mount_pose"] = palm_up_pose(task["robot_xml"], task.get("home") or {}, task["start"][:2])
    task["gravcomp"] = True
    m = build_task_model(task, accurate=True)
    ix = indices(m, task)
    h = hands(m)[0]
    acts = _finger_actuators(m, h["fingers"])
    q0 = np.clip(np.array(task["q0"], dtype=float), m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
    d = mujoco.MjData(m)
    d.qpos[ix.home_adr] = ix.home_val
    mujoco.mj_forward(m, d)
    palm = d.xpos[h["palm"]].copy()
    tips = np.array([d.xpos[c[-1]] for c in h["fingers"].values()]).mean(0)
    b = task["block"]
    half = (b["half_x"], b["half_y"], b["half_z"])
    lift = min(b["half_x"], b["half_y"], b["half_z"]) + 0.015
    spots = [("over the palm", 0.5 * (palm + tips)), ("over the fingers", 0.25 * palm + 0.75 * tips)]
    across = math.atan2(*(tips - palm)[:2][::-1]) + math.pi / 2  # its long side across the fingers
    always = [c for c in task.get("conditions", []) if c.get("when") == "always"  # at rest as a try starts (not over time)
              and not (c.get("expr") and uses(c["expr"]) & (MOTION | WINDOW_FUNCS))]
    labels = list(h["fingers"])
    sets = [labels] + [[x for x in labels if x != leave] for leave in labels]
    # the fingers the request says it starts on ("starts between the index and middle fingers"): laid over them, those
    # fingers closed (alone or with one more), and a grip only counts as that start if they all touch it
    want = [f for f in (getattr(goal, "start_fingers", None) or []) if f in labels]
    if not want:  # the request didn't say: where the goal first needs it ("at some point it's between the index and
        # middle fingers"), so practice starts there instead of in a fist it has to work loose by trial and error
        want, why = fingers_needed_first(task.get("conditions", []), labels)
        if want:
            log(f"  it starts on the {' and '.join(want)} fingers: the goal first needs it there ({why})")
    on_want = []
    if want:
        mid = np.array([d.xpos[h["fingers"][f][-1]] for f in want]).mean(0)
        spots = [(f"over the {' and '.join(want)} finger{'s' if len(want) > 1 else ''}", 0.25 * palm + 0.75 * mid),
                 (f"at the {' and '.join(want)} fingertip{'s' if len(want) > 1 else ''}", 0.1 * palm + 0.9 * mid)] + spots
        sets = [want] + [want + [x] for x in labels if x not in want] + [labels]
        on_want = [{"kind": "expression", "expr": f"touching('{f}') == 1", "text": f"the {f} finger on it"} for f in want]

    def drop(p, yaw, used, c):
        """Lay it there, close those fingers that far, let it settle: the grip's commands, or None if it blew up."""
        mujoco.mj_resetData(m, d)
        d.qpos[ix.home_adr] = ix.home_val
        d.qpos[ix.block_q:ix.block_q + 7] = [p[0], p[1], p[2] + lift, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        ctrl = q0.copy()
        for lab in used:
            for a in acts.get(lab, []):
                ctrl[a] = q0[a] + c * (_closed(m, a, q0[a]) - q0[a])
        d.ctrl[:] = ctrl
        try:
            for _ in range(int(SETTLE_S / m.opt.timestep)):
                mujoco.mj_step(m, d)
        except Exception:  # noqa: BLE001  (an unstable grip)
            return None
        return ctrl

    cands, tried, why = [], 0, {}
    for (spot, p), yaw, used, c in itertools.product(spots, (across, across - math.pi / 2), sets, CLOSURES):
        tried += 1
        ctrl = drop(p, yaw, used, c)
        if ctrl is None:
            why["the simulation became unstable"] = why.get("the simulation became unstable", 0) + 1
            continue
        touch = contact_touch(m, d, ix)
        held_by_robot = _only_robot_touches(m, d, ix) and d.xpos[ix.block][2] > 0.05
        speed = float(np.linalg.norm(d.qvel[ix.block_v:ix.block_v + 6]))
        if not held_by_robot or speed > 0.05 or not np.isfinite(d.qpos).all():
            k = ("it fell" if d.xpos[ix.block][2] < 0.05 else "it rested on something besides the hand"
                 if not _only_robot_touches(m, d, ix) else "it was still moving")
            why[k] = why.get(k, 0) + 1
            continue
        v = measure(np, d.xpos, d.xmat, d.qvel, ix, task, held=contact_held(m, d, ix), ctrl=d.ctrl, qpos=d.qpos,
                    touch=touch)
        bad = [c_["text"] or c_["expr"] or c_["kind"] for c_ in always
               if not float(condition_violation(np, {**c_, "when": "end"}, v, task.get("fixtures", {}), half)) <= 1e-6]
        missing = [c_["text"] for c_ in on_want
                   if not float(condition_violation(np, {**c_, "when": "end"}, v, task.get("fixtures", {}), half)) <= 1e-6]
        others = int(touch["fingers_touching"]) - (len(want) - len(missing))
        # the fingers asked for first (and as few others as can be), else as many fingers on it as possible
        score = (not bad, not missing, -others if want else touch["fingers_touching"], -abs(c - 0.5), -speed)
        cands.append({"score": score, "qpos": d.qpos.copy(), "ctrl": ctrl.copy(), "bad": bad, "p": p, "yaw": yaw,
                      "used": used, "c": c, "how": f"{spot}, {', '.join(used)} closed {c:.0%}", "missing": missing,
                      "on": want})
    if not cands:
        raise NoGrip(f"none of {tried} grips held the {goal.object} (" + ", ".join(f"{k}: {n}" for k, n in why.items()) + ")")
    cands.sort(key=lambda x: x["score"], reverse=True)
    if cands[0]["bad"]:
        raise NoGrip(f"it can hold it, but not while meeting the goal the whole time ({'; '.join(cands[0]['bad'])})")
    # a grip that has settled can still slip a moment later (the LinkerHand's first pick dropped the pen within 3 s
    # doing nothing): each is held, untouched, for HOLD_S in both the accurate and the learner's physics
    fast = build_task_model({**task, "physics": task.get("physics") or {"impratio": 1.0}})
    held_for = []
    best = None
    if want and cands[0]["missing"]:
        log(f"  (no grip Morph tried had the {' and '.join(want)} fingers all on the {goal.object}: starting from the "
            f"closest, {cands[0]['how']}, without {'; '.join(cands[0]['missing'])})")
    for cand in [x for x in cands if not x["bad"] and x["missing"] == cands[0]["missing"]][:MAX_CHECKED]:
        # (the asked-for fingers have to stay on it, untouched, as the goal's whole-time conditions do)
        secs = min(_holds_for(mm, task, cand["qpos"], cand["ctrl"], always + [c_ for c_ in on_want if c_["text"]
                                                                             not in cand["missing"]]) for mm in (m, fast))
        held_for.append(secs)
        cand["held_s"] = secs
        if secs >= HOLD_S:
            best = cand
            break
    note = ""
    if best is None:  # none holds the whole time: the one that holds longest (practice has to learn to keep it)
        best = max((x for x in cands if "held_s" in x), key=lambda x: x["held_s"])
        note = f"; untouched it slips after {best['held_s']:.1f} s, so practice has to learn to keep it"
    # more starts like it (a grasp cache, as in-hand rotation work uses): the same fingers and closure, the object
    # laid a little differently. Practice starts from any of them, so it can't learn one lucky resting pose
    starts = [best]
    if best.get("held_s", 0) >= HOLD_S:
        rng = np.random.default_rng(0)
        for _ in range(VARIATIONS):
            dp = np.array([*rng.uniform(-0.005, 0.005, 2), 0.0])
            yaw = best["yaw"] + rng.uniform(-0.3, 0.3)
            ctrl = drop(best["p"] + dp, yaw, best["used"], best["c"])
            if ctrl is None or not (_only_robot_touches(m, d, ix) and d.xpos[ix.block][2] > 0.05):
                continue
            qpos = d.qpos.copy()
            keep = always + [c_ for c_ in on_want if c_["text"] not in best.get("missing", [])]
            if min(_holds_for(mm, task, qpos, ctrl, keep) for mm in (m, fast)) >= HOLD_S:
                starts.append({"qpos": qpos, "ctrl": ctrl, "how": best["how"] + " (laid a little differently)"})
    _, qpos, ctrl, how = None, best["qpos"], best["ctrl"], best["how"]
    task["start_state"] = {"qpos": [float(x) for x in qpos], "ctrl": [float(x) for x in ctrl], "how": how}
    task["start_states"] = [{"qpos": [float(x) for x in st["qpos"]], "ctrl": [float(x) for x in st["ctrl"]], "how": st["how"]}
                            for st in starts]
    task["q0"] = [float(x) for x in ctrl]  # doing nothing keeps the grip
    task["home"] = {m.joint(j).name.removeprefix("robot_"): float(qpos[m.jnt_qposadr[j]])
                    for j in range(m.njnt) if m.joint(j).name.startswith("robot_")
                    and int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))}
    bq = qpos[ix.block_q:ix.block_q + 7]
    task["start"] = [float(bq[0]), float(bq[1]), float(2 * math.atan2(bq[6], bq[3]))]
    log(f"  in the hand to start: {how} ({tried} grips tried, {len(held_for)} held untouched for {HOLD_S:g} s; "
        f"{len(starts)} starting pose{'s' if len(starts) > 1 else ''}{note})")
    return task["start_state"]


HOLD_S = 3.0  # how long a grip must hold the object, untouched, to be a start
def _touches(expr: str) -> dict[str, bool]:
    """The fingers a condition needs on the object (True) or off it (False): touching('index') == 1, ... == 0, in its
    and-ed parts (not under "or" or "not")."""
    import ast
    from .goal_terms import normalize
    try:
        node = ast.parse(normalize(expr), mode="eval").body
    except SyntaxError:
        return {}
    parts = node.values if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And) else [node]
    out = {}
    for n in parts:
        if not (isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.left, ast.Call)
                and getattr(n.left.func, "id", "") == "touching" and n.left.args
                and isinstance(n.left.args[0], ast.Constant) and isinstance(n.comparators[0], ast.Constant)):
            continue
        f, op, v = str(n.left.args[0].value), n.ops[0], n.comparators[0].value
        if isinstance(op, ast.Eq):
            out[f] = v >= 0.5
        elif isinstance(op, ast.Gt) and 0 <= v < 1 or isinstance(op, ast.GtE) and 0 < v <= 1:
            out[f] = True
        elif isinstance(op, (ast.Lt, ast.LtE)) and v <= 0.5:
            out[f] = False
    return out


def fingers_needed_first(conditions: list[dict], labels: list[str]) -> tuple[list[str], str]:
    """The fingers of the first "at some point" condition that needs certain fingers on the object, less any a "the
    whole time" condition keeps off it: ([], "") if none. Not the end's: starting there, holding still would pass it."""
    from .goal_sources import finger_labels
    off = set(finger_labels(f for c in conditions if c.get("when") == "always" and c.get("expr")
                            for f, on in _touches(c["expr"]).items() if not on))
    for c in conditions:
        if c.get("when") != "ever" or not c.get("expr"):
            continue
        on = [f for f in finger_labels(f for f, v in _touches(c["expr"]).items() if v) if f in labels and f not in off]
        if on:
            return on, c.get("text") or c["expr"]
    return [], ""


MAX_CHECKED = 12  # grips held that long, best-scored first, before settling for the one that lasted longest
VARIATIONS = 8  # the chosen grip, with the object laid slightly differently


def _holds_for(m: mujoco.MjModel, task: dict, qpos, ctrl, always: list, seconds: float = HOLD_S) -> float:
    """How long the hand keeps the object, doing nothing from this state: off the ground, carried by the robot alone,
    and the goal's "the whole time" conditions kept (contact flickers under 0.1 s don't count)."""
    from .goal_terms import FLICKER_TICKS, condition_violation
    from .task_model import contact_held, contact_touch, indices, measure
    ix = indices(m, task)
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    d.ctrl[:] = ctrl
    b = task["block"]
    half = (b["half_x"], b["half_y"], b["half_z"])
    every = max(1, int(round(0.02 / m.opt.timestep)))
    streak = 0
    try:
        for k in range(int(seconds / m.opt.timestep)):
            mujoco.mj_step(m, d)
            if k % every:
                continue
            ok = d.xpos[ix.block][2] > 0.05 and _only_robot_touches(m, d, ix) and np.isfinite(d.qpos).all()
            if ok and always:
                v = measure(np, d.xpos, d.xmat, d.qvel, ix, task, held=contact_held(m, d, ix), ctrl=d.ctrl,
                            qpos=d.qpos, touch=contact_touch(m, d, ix))
                ok = all(float(condition_violation(np, {**c, "when": "end"}, v, task.get("fixtures", {}), half)) <= 1e-6
                         for c in always)
            streak = 0 if ok else streak + 1
            if streak >= FLICKER_TICKS:
                return (k - (FLICKER_TICKS - 1) * every) * m.opt.timestep
    except Exception:  # noqa: BLE001
        return 0.0
    return seconds


def _only_robot_touches(m: mujoco.MjModel, d: mujoco.MjData, ix) -> bool:
    robot, block = set(ix.robot_geoms.tolist()), set(ix.block_geoms.tolist())
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist > 0.002:
            continue
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in block and g2 not in robot) or (g2 in block and g1 not in robot):
            return False
    return True
