"""Robot programs: Nemotron writes them from building blocks, Morph runs them in simulation, checks the
result against the goal checklist, and tells Nemotron what actually happened so it can fix them.

Learning a task from scratch means discovering the whole plan by trial and error, which mostly fails for
long tasks (reach, grip, lift, carry, let go). Here the plan comes from the model as a short program, and
the building blocks underneath are things Morph already does reliably (inverse kinematics, the gripper),
so a program either works or fails for a reason that can be measured and said in words: "the hand
stopped 6 cm short: out of reach", "it closed on nothing: the fingers were 3 cm above the parcel".

Building blocks (arms with a gripper; the hand always points down in this version):
  move_hand(x, y, z, yaw, speed, release_at)   straight line to the point between the fingertips
  open_hand() / close_hand()                    close waits until the fingers stop
  wait(seconds)
Numbers can be formulas over where things are when the step starts (obj_x, obj_y, obj_z, obj_top, ...),
so a program follows the object if it isn't exactly where the scene put it.

Programs are data (JSON steps), never code: Morph only runs these building blocks with checked arguments.
"""

from __future__ import annotations

import ast
import hashlib
import re
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from .goals import GoalSpec, check
from .goal_terms import History

SPEEDS = {"slow": 0.003, "normal": 0.008, "fast": 0.03}  # metres per 20 ms control tick: 0.15, 0.4, 1.5 m/s
OPS = {"move_hand", "open_hand", "close_hand", "wait", "throw_to", "repeat_if"}
NAMES = {"obj_x", "obj_y", "obj_z", "obj_top", "obj_bottom", "obj_yaw_deg", "hand_x", "hand_y", "hand_z"}
MAX_STEPS = 24
MAX_TIME = 30.0  # seconds of simulation per program run

BLOCKS_DOC = (
    "move_hand(x, y, z, yaw, speed, release_at): the hand's grasp point (between the fingertips) moves in a straight "
    "line to (x, y, z), pointing straight down (leave one out to keep the hand where it is along it). yaw: \"keep\" (default), \"object\" (lined up with the object's sides, "
    "so the fingers close across it), or a number in degrees (0 = fingers close along east-west). speed: \"slow\" "
    "(0.15 m/s, for touching and placing), \"normal\" (0.4 m/s) or \"fast\" (1.5 m/s, for swings). release_at "
    "(optional, 0..1): open the hand at that fraction of the way, while still moving: that is how it throws. "
    "throw_to(x, y, from_x, from_y): throws what the hand is holding so it lands at (x, y): Morph does the backswing, "
    "the fast swing and lets go on the way, and corrects the aim from where practice throws land; from_x, from_y "
    "(optional) is where the hand should be when it lets go (e.g. 1 m away), else about 40 cm before the target. Use "
    "it for any throw or toss; pick the object up first. "
    "open_hand(), close_hand() (waits until the fingers stop), wait(seconds). move_hand also takes until: \"touch\" "
    "(stop as soon as the fingers touch the object). repeat_if(condition, back_to, times): measure now and, if the "
    "condition holds, go back to step back_to (at most times, 1..3), e.g. {\"do\": \"repeat_if\", \"condition\": "
    "\"held == 0\", \"back_to\": 2} after lifting grasps again if it slipped (held is 1 only while the robot alone "
    "carries it, off whatever it rested on; conditions use the goal formula names: held, lifted, x, y, "
    "z, bottom, tilt_deg, speed, dist_to(<fixture>)). Every number can instead be a formula "
    "(a string) over obj_x, obj_y, obj_z (object centre), obj_top, obj_bottom, obj_yaw_deg, hand_x, hand_y, hand_z, "
    "and the scene's own names (each fixture's <name>_x, <name>_y, <name>_top, <name>_floor; carry_z; lowest_z), "
    "with + - * / abs min max hypot sqrt, read when that step starts, e.g. {\"x\": \"obj_x\", \"z\": \"obj_top + 0.05\"} or "
    "{\"x\": \"bin_x\", \"z\": \"bin_floor + obj_z - obj_bottom\"}."
)


class ProgramError(ValueError):
    pass


# ------------------------------------------------------------------ checking a program
def scene_names(scene, carry_z: float, lowest_z: float, goal_xy=None) -> dict:
    """Numbers a formula can use by name: where each fixture is (centre x, y; its top; for containers the floor
    things rest on inside), the comfortable carry height and the lowest the hand can go."""
    out = {"carry_z": float(carry_z), "lowest_z": float(lowest_z)}
    if goal_xy is not None:  # where the goal puts the object (if it has a place)
        out["goal_x"], out["goal_y"] = float(goal_xy[0]), float(goal_xy[1])
    for n, f in scene.fixtures.items():
        key = "".join(ch if ch.isalnum() else "_" for ch in n.lower())
        out[f"{key}_x"] = float((f.lo[0] + f.hi[0]) / 2)
        out[f"{key}_y"] = float((f.lo[1] + f.hi[1]) / 2)
        out[f"{key}_top"] = float(f.top_z)
        out[f"{key}_floor"] = float(f.rest_z)
    return out


def _formula(v, what: str, known: set = frozenset()):
    if isinstance(v, dict) and len(v) == 1:  # {"formula": "obj_x"}: the same thing, written another way
        v = next(iter(v.values()))
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if not isinstance(v, str):
        raise ProgramError(f"{what}: {v!r} isn't a number or a formula")
    try:
        tree = ast.parse(v, mode="eval")
    except SyntaxError:
        raise ProgramError(f"{what}: can't read {v!r}") from None
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id not in NAMES | set(known) | {"abs", "min", "max", "hypot", "sqrt"}:
            raise ProgramError(f"{what}: unknown name {n.id!r} (known: {', '.join(sorted(NAMES | set(known)))})")
        if not isinstance(n, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub,
                              ast.UAdd, ast.Call, ast.Name, ast.Constant, ast.Load)):
            raise ProgramError(f"{what}: {type(n).__name__} isn't allowed in {v!r}")
        if isinstance(n, ast.Call) and not (isinstance(n.func, ast.Name) and n.func.id in ("abs", "min", "max", "hypot", "sqrt")):
            raise ProgramError(f"{what}: only abs, min, max, hypot and sqrt can be called")
    return v


def normalize_step(st):
    """The same step written another common way: arguments nested under "args" (a dict, or a list / a word for one
    argument such as grip's "close"), "action"/"op" for "do", and dotted names like where_it_must_end.x."""
    if not isinstance(st, dict):
        return st
    st = dict(st)
    if "do" not in st:
        st["do"] = st.pop("op", None) or st.pop("action", None) or st.pop("block", None)
    args = st.pop("args", None) or st.pop("arguments", None) or st.pop("params", None)
    if isinstance(args, dict):
        st = {**args, **st}
    elif isinstance(args, (list, tuple)) and args:
        st.setdefault("how", args[0])
    elif isinstance(args, str):
        st.setdefault("how", args)
    for k, v in list(st.items()):
        if isinstance(v, str):
            v = re.sub(r"\bwhere_it_must_end\.(x|y)\b", r"goal_\1", v)
            st[k] = re.sub(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b", r"\1_\2", v)  # bin.x -> bin_x, bin.top -> bin_top
    return st


def check_program(steps: list, known: set = frozenset(), fixtures=None) -> list[dict]:
    """The program in Morph's form; ProgramError says what's wrong, in words Nemotron can act on."""
    if not isinstance(steps, list) or not steps:
        raise ProgramError("the program has no steps")
    if len(steps) > MAX_STEPS:
        raise ProgramError(f"too many steps ({len(steps)}; at most {MAX_STEPS})")
    out = []
    for i, st in enumerate(steps, 1):
        st = normalize_step(st)
        if not isinstance(st, dict):
            raise ProgramError(f"step {i} isn't an object")
        op = st.get("do") or st.get("op")
        if op not in OPS:
            raise ProgramError(f"step {i}: unknown building block {op!r} (known: {', '.join(sorted(OPS))})")
        s = {"do": op}
        if op == "move_hand":
            if not any(k in st for k in ("x", "y", "z")):
                raise ProgramError(f"step {i}: move_hand needs at least one of x, y, z")
            for k in ("x", "y", "z"):  # one left out: stay where the hand is along it ("just lift to z = 0.2")
                s[k] = _formula(st[k], f"step {i} {k}", known) if k in st else f"hand_{k}"
            yaw = st.get("yaw", "keep")
            s["yaw"] = yaw if yaw in ("keep", "object") else _formula(yaw, f"step {i} yaw", known)
            s["speed"] = st.get("speed", "normal")
            if s["speed"] not in SPEEDS:
                raise ProgramError(f"step {i}: speed must be slow, normal or fast")
            if st.get("until") not in (None, "touch"):
                raise ProgramError(f"step {i}: until can only be \"touch\"")
            if st.get("until"):
                s["until"] = "touch"
            if st.get("release_at") is not None:
                r = float(st["release_at"])
                if not 0.0 <= r <= 1.0:
                    raise ProgramError(f"step {i}: release_at must be between 0 and 1")
                s["release_at"] = r
        elif op == "wait":
            s["seconds"] = min(max(float(st.get("seconds", 0.5)), 0.0), 5.0)
        elif op == "repeat_if":
            from .body_programs import check_repeat
            s.update(check_repeat(st, i, fixtures))
        elif op == "throw_to":
            for k in ("x", "y"):
                if k not in st:
                    raise ProgramError(f"step {i}: throw_to needs {k}")
            out += _throw_moves(st, i, known)
            continue
        out.append(s)
    return out


def _throw_moves(st: dict, i: int, known) -> list[dict]:
    """throw_to as moves: back to a wind-up point behind the release point (away from the target), then a fast
    swing towards the target that lets go half-way (the aim tuner adjusts where and when)."""
    tx, ty = _formula(st["x"], f"step {i} x", known), _formula(st["y"], f"step {i} y", known)
    if "from_x" in st and "from_y" in st:  # let go around (from_x, from_y): wind up 25 cm further back
        rx, ry = _formula(st["from_x"], f"step {i} from_x", known), _formula(st["from_y"], f"step {i} from_y", known)
        rel = f"(({rx}) - ({tx}))", f"(({ry}) - ({ty}))"
        n = f"max(0.05, hypot({rel[0]}, {rel[1]}))"
        wx, wy = f"({rx}) + 0.25 * {rel[0]} / {n}", f"({ry}) + 0.25 * {rel[1]} / {n}"
        # the swing ends 30 cm past the release point, towards the target: it flies the rest of the way
        ex, ey = f"({rx}) - 0.3 * {rel[0]} / {n}", f"({ry}) - 0.3 * {rel[1]} / {n}"
        return [{"do": "move_hand", "x": wx, "y": wy, "z": 0.25, "yaw": "keep", "speed": "normal", "throw": i},
                {"do": "move_hand", "x": ex, "y": ey, "z": 0.35, "yaw": "keep", "speed": "fast", "release_at": 0.45,
                 "throw": i}]
    else:  # from where the hand is: wind up 65 cm back from the target, along the line to it
        rel = f"(hand_x - ({tx}))", f"(hand_y - ({ty}))"
        n = f"max(0.05, hypot({rel[0]}, {rel[1]}))"
        wx, wy = f"({tx}) + 0.65 * {rel[0]} / {n}", f"({ty}) + 0.65 * {rel[1]} / {n}"
    ex, ey = tx, ty
    return [{"do": "move_hand", "x": wx, "y": wy, "z": 0.25, "yaw": "keep", "speed": "normal", "throw": i},
            {"do": "move_hand", "x": ex, "y": ey, "z": 0.35, "yaw": "keep", "speed": "fast", "release_at": 0.5,
             "throw": i}]


def describe_step(s: dict) -> str:
    if s["do"] == "move_hand" and "throw" in s:
        return "throw: wind up" if "release_at" not in s else f"throw: swing and let go {s['release_at']:.0%} of the way"
    if s["do"] == "move_hand":
        extra = f", {s['speed']}" + (f", let go {s['release_at']:.0%} of the way" if "release_at" in s else "")
        yaw = "" if s["yaw"] == "keep" else f", turned {'to the object' if s['yaw'] == 'object' else str(s['yaw']) + '°'}"
        return f"move the hand to ({s['x']}, {s['y']}, {s['z']}){yaw}{extra}"
    if s["do"] == "wait":
        return f"wait {s['seconds']:.1f} s"
    if s["do"] == "repeat_if":
        return f"if {s['condition']}: back to step {s['back_to']} (at most {s['times']}x)"
    return s["do"].replace("_", " ")


# ------------------------------------------------------------------ running a program
class _Values:
    def __init__(self, ad, half_z: float, extra: dict | None = None):
        p, tcp = ad.env.block_pose(), ad.tcp()
        self.v = dict(extra or {}) | {"obj_x": p.x, "obj_y": p.y, "obj_z": p.z, "obj_top": p.z + half_z, "obj_bottom": p.z - half_z,
                  "obj_yaw_deg": math.degrees(p.yaw), "hand_x": tcp[0], "hand_y": tcp[1], "hand_z": tcp[2]}

    def __call__(self, f) -> float:
        if isinstance(f, (int, float)):
            return float(f)
        def ev(n):
            if isinstance(n, ast.Expression):
                return ev(n.body)
            if isinstance(n, ast.Constant):
                return float(n.value)
            if isinstance(n, ast.Name):
                return self.v[n.id]
            if isinstance(n, ast.UnaryOp):
                return -ev(n.operand) if isinstance(n.op, ast.USub) else ev(n.operand)
            if isinstance(n, ast.BinOp):
                a, b = ev(n.left), ev(n.right)
                return {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b, ast.Div: a / b if b else 0.0}[type(n.op)]
            args = [ev(x) for x in n.args]
            return {"abs": lambda: abs(args[0]), "min": lambda: min(args), "max": lambda: max(args),
                    "hypot": lambda: math.hypot(*args), "sqrt": lambda: math.sqrt(max(args[0], 0.0))}[n.func.id]()
        return ev(ast.parse(f, mode="eval"))


def _touching(ad, names: dict) -> list[str]:
    """What the hand is touching right now (fixture names, the object, the floor)."""
    m, d = ad.model, ad.env.data
    hand = ad.finger_geoms
    out = set()
    for i in range(d.ncon):
        g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
        if g1 in hand or g2 in hand:
            o = g2 if g1 in hand else g1
            out.add(names.get(m.geom_bodyid[o], "floor" if m.geom_bodyid[o] == 0 else None))
    return sorted(x for x in out if x)


def run_program(ad, steps: list[dict], goal: GoalSpec, scene, settle_s: float = 1.5, on_step=None,
                names: dict | None = None, record: list | None = None) -> dict:
    """Run a checked program on an arm (an auto_arm adapter whose scene is the user's). Returns success, the
    checklist results and, per step, what was measured: the words Nemotron gets to fix the program with.
    record: a list to append (qpos, qvel, ctrl) to every control tick (a demonstration for learning from)."""
    from .arm_skills import released
    from .goals import _measure
    from .goal_terms import fixtures_geometry
    from .robots.arm import GRASP_Z
    env, m, d = ad.env, ad.model, ad.env.data
    b = scene.block
    bodies = {m.body(n).id: n for n in scene.fixtures if _has_body(m, n)}
    bodies[m.body("block").id] = scene.object_name
    hist = History([c.model_dump() for c in goal.conditions], fixtures_geometry(scene), (b.half_x, b.half_y, b.half_z))
    floor_z = GRASP_Z - 0.015
    state = {"close": False, "q": ad.q(), "t": 0.0, "touched": set()}

    def tick():
        u = ad.ctrl(state["q"], close=state["close"])
        env.advance(u)
        if record is not None:
            record.append((d.qpos.copy(), d.qvel.copy(), u.copy()))
        state["t"] += env.control_dt
        state["touched"] |= set(_touching(ad, bodies))
        p = env.block_pose()
        v = _measure(goal, scene, m, d, p, released(ad))[0]
        # held, for "when it lets go": the hand is closed on the object. Letting go is the moment the program
        # opens the hand while holding it, not a finger brushing it after it has landed.
        near = np.linalg.norm(ad.tcp() - np.array([p.x, p.y, p.z])) < 0.035
        v["held"] = 1.0 if state["close"] and (ad.gripping() or near) else 0.0
        hist.update(v)
        state["v"] = v

    report = []
    jumps: dict = {}
    nxt = 0
    while nxt < len(steps):
        i, st = nxt + 1, steps[nxt]
        nxt += 1
        if st["do"] == "repeat_if":  # reactive: measure now, and go back if the condition holds
            from .goal_terms import fixtures_geometry as _fg, parse, violation
            if "v" not in state:
                tick()
            holds = float(violation(np, parse(st["condition"], _fg(scene)), state["v"])) <= 1e-6
            if holds and jumps.get(i, 0) < st["times"]:
                jumps[i] = jumps.get(i, 0) + 1
                nxt = st["back_to"] - 1
                report.append(f"{i}. {describe_step(st)}: it holds, going back to step {st['back_to']} (time {jumps[i]})")
            else:
                report.append(f"{i}. {describe_step(st)}: " + (f"it holds, but it was already repeated {st['times']} times: "
                                                               "carrying on" if holds else "it doesn't hold: carrying on"))
            continue
        if state["t"] > MAX_TIME:
            report.append(f"{i}. not run: out of time ({MAX_TIME:.0f} s of simulation)")
            break
        val = _Values(ad, b.half_z, names)
        before = env.block_pose()
        state["touched"] = set()
        if on_step:
            on_step(i, describe_step(st))
        note = ""
        if st["do"] == "move_hand":
            target = np.array([val(st["x"]), val(st["y"]), val(st["z"])])
            low = target[2] < floor_z
            target[2] = max(target[2], floor_z)
            y0 = ad.hand_yaw()
            if st["yaw"] == "keep":
                yaw = y0
            elif st["yaw"] == "object":  # the object side nearest the hand's current turn
                oy = before.yaw
                yaw = oy + round((y0 - oy) / (math.pi / 2)) * (math.pi / 2)
            else:
                yaw = math.radians(val(st["yaw"]))
            dyaw = math.remainder(yaw - y0, math.tau)
            p0 = ad.tcp()
            dist = float(np.linalg.norm(target - p0))
            n = max(1, math.ceil(max(dist / SPEEDS[st["speed"]], abs(dyaw) / 0.06)))
            stuck, touched = None, None
            for k in range(n):
                f = (k + 1) / n
                p = p0 + (target - p0) * f
                q = ad.ik(p, y0 + dyaw * f, q0=state["q"])
                if q is None and hasattr(ad, "ik_reconfigure"):
                    q = ad.ik_reconfigure(p, y0 + dyaw * f)
                if q is None:
                    stuck = p
                    break
                state["q"] = q
                if "release_at" in st and f >= st["release_at"] and state["close"]:
                    state["close"] = False
                    note += f" (let go at {f:.0%}, hand moving at {SPEEDS[st['speed']] / env.control_dt:.1f} m/s)"
                tick()
                if st.get("until") == "touch" and scene.object_name in _touching(ad, bodies):
                    touched = f
                    break
            for _ in range(25):  # let the servos catch up (0.5 s at most)
                if touched is not None or np.linalg.norm(ad.tcp() - target) < 0.005:
                    break
                tick()
            err = float(np.linalg.norm(ad.tcp() - target))
            if touched is not None:
                note = f"touched the {scene.object_name} {touched:.0%} of the way there (stopped)" + note
            elif stuck is not None:
                note = (f"stopped {np.linalg.norm(target - ad.tcp()) * 100:.0f} cm short: the hand can't reach "
                        f"({stuck[0]:.2f}, {stuck[1]:.2f}, {stuck[2]:.2f}) pointing down from where the arm stands" + note)
            else:
                note = (f"reached it (within {err * 100:.1f} cm)" if err < 0.015 else
                        f"ended {err * 100:.1f} cm from it (blocked or too fast)") + note
            if low:
                note += f"; z was below the lowest the hand can go ({floor_z:.3f} m: the fingertips would hit the floor)"
        elif st["do"] in ("open_hand", "close_hand"):
            state["close"] = st["do"] == "close_hand"
            last, gripped = ad.aperture(), []
            for k in range(40):  # until the fingers stop (0.8 s at most)
                tick()
                gripped = (gripped + [ad.gripping()])[-8:]
                a = ad.aperture()
                if k > 5 and abs(a - last) < 0.002:
                    break
                last = a
            if state["close"]:
                # both fingers touching at some moment just now (pad contacts flicker), or the fingers stopped
                # part-way with the object between them
                p_now = env.block_pose()
                between = np.linalg.norm(ad.tcp() - np.array([p_now.x, p_now.y, p_now.z])) < 0.03
                holding = any(gripped) or (between and ad.aperture() > 0.08)
                note = ("it's holding the object (both fingers on it)" if holding else
                        f"the fingers closed on nothing (aperture {ad.aperture():.0%}); the object is "
                        f"{np.linalg.norm(ad.tcp() - np.array([before.x, before.y, before.z])) * 100:.0f} cm from the "
                        f"grasp point")
            else:
                note = "the hand is open" + ("" if released(ad) else ", but still touching the object")
        else:
            for _ in range(int(st["seconds"] / env.control_dt)):
                tick()
            note = "waited"
        after = env.block_pose()
        moved = math.hypot(after.x - before.x, after.y - before.y)
        bottom = max(0.0, after.z - (abs(np.cos(after.tilt)) * b.half_z + abs(np.sin(after.tilt)) * b.half_y))
        obj = f"; the {scene.object_name} is at ({after.x:.2f}, {after.y:.2f}), bottom {bottom * 100:.0f} cm up"
        if moved > 0.01:
            obj += f" (moved {moved * 100:.0f} cm)"
        if math.degrees(after.tilt) > 15 and getattr(b, "shape", "box") != "sphere":
            obj += f", tipped {math.degrees(after.tilt):.0f}°"
        hit = sorted(state["touched"] - {scene.object_name})
        if hit:
            obj += f"; the hand touched: {', '.join(hit)}"
        report.append(f"{i}. {describe_step(st)}: {note}{obj}")
    for _ in range(int(settle_s / env.control_dt)):  # let it settle where it ends up
        tick()
    pose = env.block_pose()
    res = check(goal, scene, m, d, pose, env.block_speed(), released(ad), history=hist)
    from .goals import placement
    pl = placement(goal, scene)
    if pl.get("has_place"):  # where it ended up, against where it had to end up, in words
        dx, dy = pose.x - pl["x"], pose.y - pl["y"]
        miss = math.hypot(dx, dy)
        if miss > 0.02:
            ns = f"{abs(dy) * 100:.0f} cm {'north' if dy > 0 else 'south'}" if abs(dy) > 0.01 else ""
            ew = f"{abs(dx) * 100:.0f} cm {'east' if dx > 0 else 'west'}" if abs(dx) > 0.01 else ""
            report.append(f"in the end the {scene.object_name} is at ({pose.x:.2f}, {pose.y:.2f}): {miss * 100:.0f} cm from "
                          f"where it had to end ({', '.join(x for x in (ns, ew) if x)} of it)")
    return {"success": res["success"], "results": res["results"], "reasons": res["reasons"], "report": report,
            "sim_s": round(state["t"], 1), "miss_m": math.hypot(pose.x - pl["x"], pose.y - pl["y"]) if pl.get("has_place") else None,
            "miss_vec": [pose.x - pl["x"], pose.y - pl["y"]] if pl.get("has_place") else None}


def _robust(ad, trial, goal, scene, names, starts) -> tuple[int, dict]:
    """How many of the starts it works from (stops at the first failure), and the last run."""
    from .project import _params
    ok, run = 0, None
    for i, start in enumerate(starts):
        ad.reset(i, start, _params(i))
        run = run_program(ad, trial, goal, scene, names=names)
        if not run["success"]:
            break
        ok += 1
    return ok, run


def tune(ad, steps: list[dict], goal: GoalSpec, scene, names: dict, params, log=print) -> tuple[list[dict], dict] | None:
    """The plan is right but the aim is off (a throw that lands short): try the same program with the moment
    it lets go moved earlier or later, and the swing slower or faster, and keep what works, or else the
    closest miss. Each run takes well under a second."""
    from .project import _params
    idx = [i for i, s in enumerate(steps) if "release_at" in s]
    if not idx:
        return None
    from .project import _start_poses
    starts = _start_poses(scene, 3)  # the same small variations the trials use
    best = None
    k = idx[-1]
    for r in (0.5, 0.6, 0.4, 0.7, 0.3, 0.8):  # when to let go along the swing
        for lift in (0.0, 0.1, 0.2):  # a swing that ends higher throws in a higher arc (over a container's walls)
            shift = np.zeros(2)
            for _ in range(4):  # correct the aim from where the last throw landed, as a person would
                trial = [dict(s) for s in steps]
                trial[k]["release_at"], trial[k]["speed"] = r, "fast"
                trial[k]["x"] = f"({trial[k]['x']}) + {shift[0]:.4f}"
                trial[k]["y"] = f"({trial[k]['y']}) + {shift[1]:.4f}"
                if lift:
                    trial[k]["z"] = f"({trial[k]['z']}) + {lift}"
                ad.reset(0, scene.start, params)
                run = run_program(ad, trial, goal, scene, names=names)
                if run["success"]:  # works from the scene's start: does it hold up when the object is a bit off?
                    n_ok = sum(_robust(ad, trial, goal, scene, names, [st0])[0] for st0 in starts[1:])
                    if n_ok == 0:
                        run = dict(run, success=False, robust=1)
                key = (run["success"], run.get("robust", 0), -(run["miss_m"] if run["miss_m"] is not None else 9.0))
                if best is None or key > best[0]:
                    best = (key, trial, run)
                if run["success"]:
                    log(f"  Morph tuned the throw: let go {r:.0%} of the way" + (f", swing ending {lift * 100:.0f} cm higher" if lift else "")
                        + (f", aim moved {np.linalg.norm(shift) * 100:.0f} cm" if np.linalg.norm(shift) > 0.005 else ""))
                    return trial, run
                if run.get("miss_vec") is None or run.get("robust"):
                    break  # it lands from the scene's start; only the variations miss: aim stays
                shift = shift - np.array(run["miss_vec"])
    return None if best is None else (best[1], best[2])


def _fx(scene) -> dict:
    from .goal_terms import fixtures_geometry
    return fixtures_geometry(scene)


def _has_body(m, name: str) -> bool:
    try:
        m.body(name)
        return True
    except KeyError:
        return False


def feedback(run: dict) -> dict:
    """What Nemotron is told about a run, to fix the program with."""
    return {"what_happened_step_by_step": run["report"],
            "goal_checklist": [f"{'met' if r['ok'] else 'NOT met'}: {r['condition']} ({r['measured']})" for r in run["results"]],
            "other_problems": [r for r in run["reasons"] if not any(r.startswith(x["condition"]) for x in run["results"])]}


# ------------------------------------------------------------------ write, run, fix
PROGRAM_CACHE = Path(__file__).resolve().parents[1] / "runs" / "programs"


def _cache_path(name: str, goal: GoalSpec, scene, ad) -> Path:
    """Where a working program for this robot, goal, scene and mount is kept (programs are data, cheap to keep)."""
    key = json.dumps([name, [c.model_dump() for c in goal.conditions], str(getattr(scene, "path", "")), list(scene.start),
                      [round(float(v), 3) for v in ad.cfg.base_pos[:2]], round(float(getattr(ad.cfg, "base_yaw", 0.0)), 3)],
                     sort_keys=True, default=str)
    return PROGRAM_CACHE / f"{name}__{hashlib.sha1(key.encode()).hexdigest()[:12]}.json"


def solve(pj, goal: GoalSpec, robot_path: str, name: str, llm, log=print, emit=lambda **e: None, n: int = 5,
          attempts: int = 5, stand_at=None) -> dict:
    """Nemotron writes a program; Morph runs it and tells it what happened; repeat until the checklist is met
    (at most `attempts` programs). A program that works is then run on n slightly different starts."""
    from .goals import explain, placement, release_spot
    from .project import _params, _start_poses
    from .robots.arm import GRASP_Z
    from .robots.auto_arm import NotAGraspingArm, build_auto_arm
    from .skill_steps import robot_brief
    from .introspect import profile_robot
    scene, b = pj.scene, pj.scene.block
    p = placement(goal, scene)
    spot = release_spot(goal, scene)
    throwing = any(c.when == "release" for c in goal.conditions)
    if throwing:  # it flies there: the arm only has to reach the object (and where it lets go), not the target
        min_carry, task_xy = None, [scene.start[:2]] + ([spot] if spot else [])
    else:
        min_carry = p["clear_z"] + 0.02 + 2 * b.half_z - 0.015 if p["needs_lift"] else None
        task_xy = [scene.start[:2], (p["x"], p["y"])] + ([spot] if spot else [])
    emit(type="stage", robot=name, stage="setup", text="setting the arm up for Nemotron's program: mount, servos, grip")
    try:
        ad, rep = build_auto_arm(robot_path, name, block=b, log=log, task_xy=task_xy, min_carry_z=min_carry,
                                 start=scene.start, keepout=[(f.lo, f.hi) for f in scene.fixtures.values()], mount=stand_at)
    except NotAGraspingArm as e:
        return {"success": 0, "n": 0, "setup_error": str(e)}
    prof = profile_robot(robot_path)
    facts = {"arm_base_xy_m": [round(float(v), 3) for v in ad.cfg.base_pos[:2]],
             "arm_base_facing_deg": round(math.degrees(getattr(ad.cfg, "base_yaw", 0.0)), 1),
             "arm_reach_m": rep.get("reach_m"), "lowest_hand_z_m": round(GRASP_Z - 0.015, 3),
             "comfortable_carry_height_m": rep.get("travel_z"),
             "gripper_opening_m": round(float(prof.gripper.get("max_opening_m", 0.0)), 3),
             "object_half_size_m": [b.half_x, b.half_y, b.half_z], "object_start": [round(float(v), 3) for v in scene.start],
             "grasp_tip": "to grip it: open the hand, move above it with yaw 'object', come down slowly to z = obj_z, close",
             "where_it_must_end": {k: round(float(p[k]), 3) for k in ("x", "y", "support_z")} if p.get("has_place") else None,
             "where_to_let_go": [round(v, 3) for v in spot] if spot else None}
    checklist = [f"{c.describe()}: {explain(c, scene)}" for c in goal.conditions]
    names = scene_names(scene, rep.get("travel_z") or 0.15, GRASP_Z - 0.015,
                        (p["x"], p["y"]) if p.get("has_place") else None)
    facts["names_for_formulas"] = {k: round(v, 3) for k, v in names.items()}
    tries, steps, ok_run = [], None, None
    a, format_fixes = 0, 0
    cache = _cache_path(name, goal, scene, ad)
    if cache.exists():  # a program that worked here before (same robot, goal, scene and mount): try it first, for free
        try:
            cached = check_program(json.loads(cache.read_text())["program"], set(names), _fx(scene))
            ad.reset(0, scene.start, _params(0))
            run = run_program(ad, cached, goal, scene, names=names)
            if run["success"]:
                log("  using the program that worked last time (no new Nemotron call)")
                emit(type="program", robot=name, attempt=1, steps=[describe_step(x) for x in cached],
                     rationale="Nemotron's program that worked here last time (kept, so no new call)")
                emit(type="program_result", robot=name, attempt=1, success=True, report=run["report"], reasons=[])
                steps, ok_run = cached, run
                attempts = 0
        except (ProgramError, KeyError, ValueError):
            pass
    while a < attempts:
        emit(type="stage", robot=name, stage="trials", text=f"Nemotron is writing program {a + 1}")
        try:
            prop, rec = llm.write_program(robot_brief(prof), checklist, scene.geometry(), facts, BLOCKS_DOC, tries)
        except Exception as e:  # noqa: BLE001  (a cut-off or garbled reply: ask again, it's not a wrong idea)
            log(f"  Nemotron's reply couldn't be read ({e}); asking again")
            format_fixes += 1
            if format_fixes > 3:
                break
            continue
        try:
            steps = check_program(prop.steps, set(names), _fx(scene))
        except ProgramError as e:  # a slip in the writing, not a wrong idea: doesn't use up a try (up to 3 times)
            tries.append({"program": prop.steps, "problem": f"Morph couldn't run it: {e}"})
            log(f"  program {a + 1}: can't run it: {e}")
            format_fixes += 1
            if format_fixes > 3:
                a += 1
            continue
        log(f"  program {a + 1} ({rec.latency_s}s): " + "; ".join(describe_step(s) for s in steps))
        emit(type="program", robot=name, attempt=a + 1, steps=[describe_step(s) for s in steps], rationale=prop.rationale)
        ad.reset(0, scene.start, _params(0))
        run = run_program(ad, steps, goal, scene, names=names,
                          on_step=lambda i, d, k=a: emit(type="caption", robot=name, text=f"Program {k + 1}, step {i}: {d}"))
        tries.append({"program": steps, "rationale": prop.rationale, "result": feedback(run)})
        emit(type="program_result", robot=name, attempt=a + 1, success=run["success"], report=run["report"],
             reasons=run["reasons"])
        log(f"  program {a + 1}: {'reached the goal' if run['success'] else 'failed: ' + '; '.join(run['reasons'][:2])}")
        if not run["success"] and any("release_at" in st for st in steps):
            emit(type="caption", robot=name, text="Morph is fine-tuning the aim: when to let go, how fast to swing")
            tuned = tune(ad, steps, goal, scene, names, _params(0), log)
            if tuned is not None:
                t_steps, t_run = tuned
                if t_run["success"]:
                    steps, run = t_steps, t_run
                    tries[-1]["tuned_by_morph"] = [describe_step(st) for st in steps if "release_at" in st]
                    emit(type="program_result", robot=name, attempt=a + 1, success=True, report=run["report"],
                         reasons=[], tuned=True)
                else:
                    tries[-1]["closest_tuning"] = {"program": t_steps, "result": feedback(t_run)}
        if run["success"]:  # works from the scene's start: does it hold up when the object starts a bit off?
            starts = _start_poses(scene, 3)[1:]
            n_ok, r2 = 0, None
            for j, st0 in enumerate(starts, 1):
                ad.reset(j, st0, _params(j))
                r = run_program(ad, steps, goal, scene, names=names)
                n_ok += r["success"]
                r2 = r if not r["success"] else r2
            if n_ok == 0:  # only from the exact start: a fluke, not a program (most starts: keep it; trials tell)
                log(f"  program {a + 1} works from the scene's start but not when the {scene.object_name} starts "
                    "a little off")
                if any("release_at" in st for st in steps):
                    tuned = tune(ad, steps, goal, scene, names, _params(0), log)
                    if tuned is not None and tuned[1]["success"]:
                        steps, run = tuned
                    else:
                        run = dict(run, success=False)
                else:
                    run = dict(run, success=False)
                if not run["success"]:
                    tries[-1]["result"]["other_problems"].append(
                        f"it worked from the scene's start, but failed when the {scene.object_name} started about 2 cm away "
                        f"or turned 10°: {'; '.join(r2['reasons'][:2])}. Make it follow the object (obj_x, obj_y, ...)")
        if run["success"]:
            ok_run = run
            break
        a += 1
    out = {"program_attempts": tries, "setup": rep}
    if ok_run is None:
        out |= {"success": 0, "n": 0, "why_not": f"Nemotron's programs didn't reach the goal in {a} tries"}
        return out
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"robot": name, "goal": goal.describe(), "program": steps}, indent=1))
    emit(type="stage", robot=name, stage="trials", text="trying the program from slightly different starts")
    trials, demos = [], []
    for i, start in enumerate(_start_poses(scene, n)):
        emit(type="trial_start", robot=name, i=i, n=n)
        ad.reset(i, start, _params(i))
        rec: list = []
        r = run_program(ad, steps, goal, scene, names=names, record=rec)
        if r["success"]:
            demos.append(rec)
        trials.append({"success": r["success"], "conditions": r["results"], "reasons": r["reasons"]})
        emit(type="trial", robot=name, i=i, n=n, success=r["success"], conditions=r["results"], reasons=r["reasons"])
        log(f"  trial {i + 1}: {'success' if r['success'] else 'fail: ' + '; '.join(r['reasons'][:2])}")
    ok = sum(t["success"] for t in trials)
    log(f"{name}: {ok}/{n} with Nemotron's program")
    return out | {"success": ok, "n": n, "trials": trials, "program": steps, "demos": demos, "adapter": ad}
