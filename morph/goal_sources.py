"""Where a goal comes from: a user's demonstration, a description, or both.

Demonstrations are recordings of what happened to the object, not of a robot's joints, so they
work whatever robot (or person) did the task:

  object path   .npz with `pos` (T x 3) and `quat` (T x 4, w x y z), optional `t`;
                .csv with columns x, y, z and either qw, qx, qy, qz or yaw (degrees);
                .json {"pos": [...], "quat": [...]}
  joint log     .npz with `qpos` (T x nq) and `model` (path to the MJCF it was recorded in):
                replayed in MuJoCo to recover the object's path (`object` names the body, else
                the project's object name is used)

Descriptions go to Nemotron (morph.reasoning.goal_from_request) when a key is set; the rule parser
here handles plain sentences ("put the parcel into the bin facing north") without one, and is the
fallback when Nemotron's answer doesn't validate against the scene.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import mujoco
import numpy as np

from .geometry import quat_to_mat, tilt_from_mat, wrap_angle, yaw_from_mat
from .goals import Condition, GoalSpec, measure_relations, validate
from .user_scene import UserScene

HEADINGS = {"east": 0.0, "north": 90.0, "west": 180.0, "south": -90.0}


# ------------------------------------------------------------------ demonstrations
def load_demo(path: str | Path, object_name: str | None = None) -> dict:
    """-> {"pos": T x 3, "quat": T x 4, "source": str}"""
    path = Path(path)
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        if "qpos" in z.files:
            model = Path(str(z["model"]))
            if not model.is_absolute():
                model = (path.parent / model).resolve()
            obj = str(z["object"]) if "object" in z.files else object_name
            return replay_joint_log(model, z["qpos"], obj) | {"source": f"joint log replayed in {model.name}"}
        pos, quat = z["pos"], z["quat"]
    elif path.suffix == ".json":
        j = json.loads(path.read_text())
        pos, quat = np.array(j["pos"]), np.array(j["quat"])
    elif path.suffix == ".csv":
        rows = list(csv.DictReader(path.open()))
        pos = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
        if "qw" in rows[0]:
            quat = np.array([[float(r[k]) for k in ("qw", "qx", "qy", "qz")] for r in rows])
        else:
            quat = np.array([[math.cos(math.radians(float(r["yaw"])) / 2), 0, 0, math.sin(math.radians(float(r["yaw"])) / 2)]
                             for r in rows])
    else:
        raise ValueError(f"unsupported demonstration format {path.suffix} (use .npz, .csv or .json)")
    return {"pos": np.asarray(pos, float), "quat": np.asarray(quat, float), "source": f"object path from {path.name}"}


def replay_joint_log(model_path: Path, qpos: np.ndarray, object_name: str) -> dict:
    """Kinematic replay of a recorded qpos log; reads the object body's pose at every frame."""
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    b = m.body(object_name).id
    pos, quat = [], []
    for q in np.asarray(qpos):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        pos.append(d.xpos[b].copy())
        quat.append(d.xquat[b].copy())
    return {"pos": np.array(pos), "quat": np.array(quat)}


def summarize_demo(demo: dict, scene: UserScene) -> dict:
    pos, quat = demo["pos"], demo["quat"]
    R0, R1 = quat_to_mat(quat[0]), quat_to_mat(quat[-1])
    yaw0, yaw1 = yaw_from_mat(R0), yaw_from_mat(R1)
    return {"source": demo["source"], "frames": len(pos),
            "start": {"xyz": np.round(pos[0], 3).tolist(), "yaw_deg": round(math.degrees(yaw0), 1)},
            "end": {"xyz": np.round(pos[-1], 3).tolist(), "yaw_deg": round(math.degrees(yaw1), 1),
                    "tilt_deg": round(math.degrees(tilt_from_mat(R1)), 1)},
            "turned_deg": round(math.degrees(wrap_angle(yaw1 - yaw0)), 1),
            "moved_cm": round(float(np.linalg.norm(pos[-1][:2] - pos[0][:2])) * 100, 1),
            "max_height_cm": round(float(pos[:, 2].max() - scene.block.half_z) * 100, 1),
            "end_relations": [c.describe() for c in measure_relations(scene, *pos[-1], yaw1, tilt_from_mat(R1))]}


def goal_from_demo(demo: dict, scene: UserScene) -> GoalSpec:
    """What the demonstration achieved, as conditions: where the object came to rest relative to the
    scene, its heading if the demo deliberately turned it, and upright if it stayed upright."""
    pos, quat = demo["pos"], demo["quat"]
    R0, R1 = quat_to_mat(quat[0]), quat_to_mat(quat[-1])
    x, y, z = pos[-1]
    yaw0, yaw1, tilt = yaw_from_mat(R0), yaw_from_mat(R1), tilt_from_mat(R1)
    conds = measure_relations(scene, x, y, z, yaw1, tilt)
    notes = []
    if not any(c.kind in ("inside", "on_top_of") for c in conds):
        conds.append(Condition(kind="at_position", x=round(float(x), 3), y=round(float(y), 3), tol=0.05, source="demo"))
    if abs(math.degrees(wrap_angle(yaw1 - yaw0))) > 15:
        conds.append(Condition(kind="heading", yaw_deg=round(math.degrees(yaw1), 1), tol=15, source="demo"))
    else:
        notes.append(f"the demo turned the object only {math.degrees(wrap_angle(yaw1 - yaw0)):.0f} deg: heading left free")
    if tilt < math.radians(10):
        conds.append(Condition(kind="upright", tol=10, source="demo"))
    return GoalSpec(object=scene.object_name, conditions=conds, notes=notes)


# ------------------------------------------------------------------ descriptions (rule parser)
def _fixture_mentions(text: str, scene: UserScene) -> list[tuple[int, str]]:
    out = []
    for name in scene.fixtures:
        for alias in {name.lower(), name.lower().replace("_", " ")}:
            for mt in re.finditer(r"\b" + re.escape(alias) + r"\b", text):
                out.append((mt.start(), name))
    return sorted(set(out))


def goal_from_description_rules(text: str, scene: UserScene) -> GoalSpec:
    t = text.lower()
    conds, notes = [], []
    for at, name in _fixture_mentions(t, scene):
        before = t[max(0, at - 25):at]
        f = scene.fixtures[name]
        if re.search(r"\b(off|out of|away from|far from|out from|remove\w* from|take\w* from)\b(\s+(the|of the))?\s*$", before):
            if re.search(r"\b(away|far) from\b(\s+the)?\s*$", before):
                m_cm = re.search(r"(\d+(?:\.\d+)?)\s*cm", t)
                conds.append(Condition(kind="away_from", target=name, tol=float(m_cm.group(1)) / 100 if m_cm else 0.3,
                                       source="description"))
            else:  # off the shelf / out of the bin: not where it is now
                kind = "inside" if f.kind == "container" else "on_top_of" if f.kind == "surface" else "touching"
                conds.append(Condition(kind=kind, target=name, negate=True, source="description"))
            continue
        if re.search(r"\b(in|into|inside)\b(\s+the)?\s*$", before):
            kind = "inside"
        elif re.search(r"\b(on|onto|on top of|upon)\b(\s+the)?\s*$", before):
            kind = "on_top_of"
            if f.kind == "container":
                kind = "inside"
                notes.append(f"'on the {name}': {name} is a container, read as inside")
        elif re.search(r"\b(against|touching|touch|up to|flush with)\b(\s+the)?\s*$", before):
            kind = "touching"
        elif m_side := re.search(r"\b(north|south|east|west)\s+(?:side\s+)?of\b(\s+the)?\s*$", before):
            conds.append(Condition(kind="beside", target=name, side=m_side.group(1), source="description"))
            continue
        elif re.search(r"\b(near|close to|next to|beside|by)\b(\s+the)?\s*$", before):
            conds.append(Condition(kind="near", target=name, tol=0.1, source="description"))
            continue
        else:  # "to the bin" / "to the shelf": what that kind of fixture is for
            kind = {"container": "inside", "surface": "on_top_of", "obstacle": "touching"}[f.kind]
            notes.append(f"'{name}' with no preposition: read as {kind} (it is a {f.kind})")
        conds.append(Condition(kind=kind, target=name, source="description"))
    m = re.search(r"\bat\s*\(?\s*(-?\d*\.?\d+)\s*(?:m)?\s*,\s*(-?\d*\.?\d+)", t)
    if m:
        tol = 0.05
        mt = re.search(r"within\s*(\d+(?:\.\d+)?)\s*cm", t)
        if mt:
            tol = float(mt.group(1)) / 100
        conds.append(Condition(kind="at_position", x=float(m.group(1)), y=float(m.group(2)), tol=tol, source="description"))
    m = re.search(r"\b(facing|faces|face|pointing|points|point|arrow (?:pointing )?(?:to(?:wards)?)?)\s*(?:to(?:wards)?\s+)?(?:the\s+)?"
                  r"(north|south|east|west)\b", t) or \
        re.search(r"\b(north|south|east|west)\s*(?:-|\s)?facing\b", t)
    if m:
        word = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)
        conds.append(Condition(kind="heading", yaw_deg=HEADINGS[word], tol=15, source="description"))
    else:
        m = re.search(r"\b(?:facing|heading|yaw(?: of)?|rotated to)\s*(-?\d+(?:\.\d+)?)\s*(?:deg|degrees|°)", t)
        if m:
            conds.append(Condition(kind="heading", yaw_deg=float(m.group(1)), tol=15, source="description"))
        else:
            m = re.search(r"\b(?:turn|rotate)\w*\s+(?:it\s+)?(?:by\s+)?(-?\d+(?:\.\d+)?)\s*(?:deg|degrees|°)", t)
            if m:
                yaw = math.degrees(scene.start[2]) + float(m.group(1)) * (-1 if "clockwise" in t and "counter" not in t else 1)
                conds.append(Condition(kind="heading", yaw_deg=round(math.degrees(wrap_angle(math.radians(yaw))), 1),
                                       tol=15, source="description"))
    # throwing / dropping: where it is when the robot lets go
    m = re.search(r"\b(throw|toss|lob|fling|drop)\w*\b.*?\bfrom\s+(\d+(?:\.\d+)?)\s*(cm|m)\b(?:\s+(?:to the\s+)?"
                  r"(north|south|east|west|up|high|above))?", t)
    if m:
        dist = float(m.group(2)) / (100 if m.group(3) == "cm" else 1)
        side = m.group(4)
        place = next((c for c in conds if c.kind in ("inside", "on_top_of", "touching") and not c.negate), None)
        f = scene.fixtures.get(place.target) if place else None
        if side in ("up", "high", "above") or m.group(1).startswith("drop"):
            top = f.top_z if f is not None else 0.0
            expr, words = f"bottom > {top + dist:.3f}", f"let go with its bottom at least {dist * 100:.0f} cm above the {place.target if place else 'floor'}"
        elif f is not None and side:
            expr = {"north": f"y > {f.hi[1] + dist:.3f}", "south": f"y < {f.lo[1] - dist:.3f}",
                    "east": f"x > {f.hi[0] + dist:.3f}", "west": f"x < {f.lo[0] - dist:.3f}"}[side]
            words = f"let go at least {dist:g} m {side} of the {place.target}"
        elif f is not None:
            expr, words = f"dist_to('{place.target}') >= {dist:.3f}", f"let go at least {dist:g} m from the {place.target}"
        else:
            expr, words = f"travelled < {dist:.3f}", f"let go within {dist:g} m of where it started"
        conds.append(Condition(kind="expression", expr=expr, text=words, when="release", source="description"))
        # "from 1 m north, away from the bin" describes where the throw starts, not where it ends
        conds = [c for c in conds if not (c.kind in ("away_from", "near", "beside") and place is not None
                                          and c.target == place.target)]
    # a throw with no distance given: it has to fly, so it's let go moving fast, away from where it lands
    if not m and re.search(r"\b(throw|toss|lob|fling|chuck)\w*\b", t):
        place = next((c for c in conds if c.kind in ("inside", "on_top_of", "touching") and not c.negate), None)
        away = f" and dist_to('{place.target}') > 0.25" if place else " and travelled < 0.3"
        conds.append(Condition(kind="expression", expr="speed > 1.0" + away, when="release", source="description",
                               text="let go while it's moving fast" + (f", at least 25 cm from the {place.target}" if place else "")))
    # how it should lie
    if re.search(r"\b(upside[- ]down|flip\w*|turn\w*\s+(?:\w+\s+){0,2}over)\b", t):
        conds.append(Condition(kind="orientation", up_face="bottom", tol=20, source="description"))
    elif re.search(r"\b(on its side|(?:knock|tip|push)\w*\s+(?:\w+\s+){0,2}over|topple\w*|lay\s+(?:\w+\s+){0,2}down|lying down)\b", t):
        conds.append(Condition(kind="orientation", up_face="side", tol=20, source="description"))
    elif re.search(r"\b(upright|right way up|standing up|stand it up|don'?t (let it )?(tip|fall|drop|knock)|carefully|gently|fragile)\b", t):
        conds.append(Condition(kind="upright", tol=10, source="description"))
    # lifting / holding
    m = re.search(r"\b(lift|raise|pick up|hold|hold up|keep)\w*\b.*?(?:(\d+(?:\.\d+)?)\s*(cm|m)\b)?", t)
    if m and re.search(r"\b(lift|raise|hold|in the air|off the (floor|ground))\w*\b", t) and not any(
            c.kind in ("inside", "on_top_of") and not c.negate for c in conds):
        h = re.search(r"(\d+(?:\.\d+)?)\s*(cm|m)\b", t)
        z = (float(h.group(1)) / (100 if h.group(2) == "cm" else 1)) if h else 0.15
        hold = bool(re.search(r"\b(hold|keep it (up|in the air)|in the air)\b", t))
        conds.append(Condition(kind="height", min_z=z, hold=hold, source="description"))
    # moving it a distance
    m = re.search(r"\b(?:move|push|slide|shift|drag)\w*\s+(?:it|the \w+)?\s*(?:by\s+)?(\d+(?:\.\d+)?)\s*(cm|m)\b"
                  r"(?:\s+(?:to the\s+|towards?\s+(?:the\s+)?)?(north|south|east|west))?", t)
    if m and not any(c.kind in ("at_position", "inside", "on_top_of", "touching") and not c.negate for c in conds):
        conds.append(Condition(kind="moved", tol=float(m.group(1)) / (100 if m.group(2) == "cm" else 1),
                               side=m.group(3), source="description"))
    if re.search(IN_HAND, t) and re.search(r"\b(never|not|n't|without)\s+(?:\w+\s+){0,2}(fall\w*|drop\w*|slip\w*)\s+(out|from)\b|"
                                         r"\b(stay|stays|keep|keeps|remain\w*)\s+(?:\w+\s+){0,2}in the (hand|fingers)\b", t):
        conds.append(Condition(kind="expression", expr="in_hand == 1", when="always", source="description",
                               text="it never falls out of the hand"))
    if re.search(r"\b(spin|twirl|twiddle|rotate|roll)\w*\b", t) and re.search(IN_HAND, t):
        conds.append(Condition(kind="expression", expr="spin_dps > 180", when="last", seconds=2.0,
                               text="spinning at over half a turn per second", source="description"))
        if re.search(r"\b(same (direction|way)|round and round|full (circle|turn)s?|(\d+|one|two|three) (full )?(turns?|"
                     r"circles?|times round))\b", t):  # one way, not shaken back and forth: how far it turned in all
            conds.append(Condition(kind="expression", expr="abs(change(yaw_deg, 2)) >= 360", when="end",
                                   text="turned at least a full turn about the vertical in the last 2 s",
                                   source="description"))
    m = re.search(r"\b(?:without|not|no|except|but not|other than)\s+(?:using\s+|with\s+|for\s+)?(?:the\s+|its\s+|a\s+)?"
                  r"(thumb|index|middle|ring|little|pinky)\b", t)
    if m:
        finger = {"pinky": "little"}.get(m.group(1), m.group(1))
        conds.append(Condition(kind="expression", expr=f"touching('{finger}') == 0", when="always",
                               text=f"the {finger} never touches it", source="description"))
    if not conds:
        notes.append("no condition found in the description (try Nemotron for free-form requests)")
    start = "in_hand" if re.search(IN_HAND, t) else "scene"
    return GoalSpec(object=scene.object_name, conditions=conds, notes=notes, start=start,
                    start_fingers=start_fingers(t) if start == "in_hand" else [])


FINGER_WORDS = r"thumb|index|middle|ring|little|pinky"


def start_fingers(t: str) -> list[str]:
    """The fingers a request says it starts on: "starts between the index and middle fingers" -> [index, middle]."""
    m = re.search(r"\bstart\w*\b[^.]*?\b(?:between|on|in|with|across)\s+(?:the\s+|its\s+)?((?:(?:" + FINGER_WORDS
                  + r")(?:\s+fingers?)?(?:\s*,\s*|\s+and\s+(?:the\s+)?|\s+)?)+)", t)
    return finger_labels(re.findall(FINGER_WORDS, m.group(1))) if m else []


def finger_labels(names) -> list[str]:
    """Finger names as Morph labels them (pinky -> little), each once, in order."""
    out = []
    for n in names:
        n = {"pinky": "little"}.get(str(n).lower().strip(), str(n).lower().strip())
        if n in ("thumb", "index", "middle", "ring", "little") and n not in out:
            out.append(n)
    return out


# the request is about handling it in the hand, so it starts there (spinning a pen, rolling a ball in the palm)
IN_HAND = (r"\b(in|within|between|across|over|around|through|along)\s+(the\s+|its\s+|one\s+|a\s+)?(fingers?|palm|hand)\b"
           r"|\bin-hand\b|\bfinger\s?tips?\b")


def merge(demo_goal: GoalSpec | None, text_goal: GoalSpec | None) -> GoalSpec:
    """Demo + description: the description wins where they talk about the same thing (place, heading)."""
    if demo_goal is None:
        return text_goal
    if text_goal is None:
        return demo_goal
    placey = {"at_position", "inside", "on_top_of", "touching"}
    said = {c.kind for c in text_goal.conditions if c.source == "description"}
    keep = []
    for c in demo_goal.conditions:
        if c.kind in placey and said & placey:
            continue
        if c.kind in said:
            continue
        keep.append(c)
    extra = [c for c in text_goal.conditions if c.source == "description"]
    notes = demo_goal.notes + text_goal.notes + ([f"description overrides the demo for: {', '.join(sorted(said))}"] if said else [])
    return GoalSpec(object=demo_goal.object, conditions=keep + extra, notes=notes, start=text_goal.start)


def check_goal(goal: GoalSpec, scene: UserScene) -> GoalSpec:
    errs = validate(goal, scene)
    if errs:
        raise ValueError("; ".join(errs))
    return goal
