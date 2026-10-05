"""Programs for any robot body: building blocks that move a named part of the robot (a fingertip, a tool flange,
a gripper, a foot of a robot bolted down), so Nemotron can write a recipe for robots the arm programs
(programs.py: arms with a parallel gripper, hand pointing down) don't cover: arms without a gripper, hands, tilted
or sideways hands (flipping, side grasps, pouring), several limbs.

They run on the learner's own model (task_model.py: the same robot, mount, servos and scene the policy trains in),
so a recipe that works is recorded there and the learner copies it without any translation (learn_task.demo_to_task
with ad=None).

Building blocks (data, never code; Morph checks every argument):
  move(part, x, y, z, point, turn, speed, until)   the part's tip in a straight line to (x, y, z)
  push(part, direction, distance, speed)           slide the part along the floor plane, keeping its height
  grip("close" | "open")                           the gripper, if it has one; waits until the fingers stop
  joints({name: angle}, seconds)                   set joints directly (radians)
  home(seconds)                                    back to the starting pose
  wait(seconds)
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .goal_terms import History
from .goals import GoalSpec, check
from .programs import SPEEDS, ProgramError, _formula, scene_names
from .task_model import CTRL_DT, build_task_model, contact_held, indices, is_position

BODY_OPS = {"move", "push", "grip", "joints", "home", "wait", "repeat_if"}
POINTS = {"down": (0, 0, -1), "up": (0, 0, 1), "north": (0, 1, 0), "south": (0, -1, 0), "east": (1, 0, 0),
          "west": (-1, 0, 0)}
MAX_STEPS = 24
MAX_TIME = 40.0
CACHE = Path(__file__).resolve().parents[1] / "runs" / "programs"

BLOCKS_DOC = (
    "move(part, x, y, z, point, turn, speed, until): the tip of one of the robot's parts (see facts.parts) moves in a "
    "straight line to (x, y, z) (leave one out to keep it). point (optional): which way the part's tip points while "
    "it moves: \"down\", \"up\", \"north\", \"south\", \"east\", \"west\", \"object\" (towards the object's centre), "
    "\"keep\" (default), or [dx, dy, dz]. turn (optional, parts that grip): \"object\" lines the fingers up across "
    "the object's sides, or a number in degrees (0 = fingers close along east-west). speed: \"slow\" (0.15 m/s, for "
    "touching and placing), \"normal\" (0.4 m/s), \"fast\" (1.5 m/s). until (optional): \"touch\" stops the move as "
    "soon as the part touches the object. "
    "push(part, direction, distance, speed): slide the part horizontally by distance (m) towards direction (degrees: "
    "0 = east, 90 = north), keeping its height and pointing; use it to push or sweep the object (put the part against "
    "the far side of the object first, low on its face). "
    "grip(\"close\") / grip(\"open\"): only for parts listed with can_grip; close waits until the fingers stop. "
    "joints({joint: radians}, seconds): set some joints directly. home(seconds): back to the starting pose. "
    "wait(seconds). repeat_if(condition, back_to, times): measure now and, if condition holds, go back to step back_to "
    "(at most times, 1..3): e.g. {\"do\": \"repeat_if\", \"condition\": \"held == 0\", \"back_to\": 2} after lifting grasps "
    "again if it slipped out (held is 1 only while the robot alone carries it, off whatever it rested on; conditions use the goal formula names: held, lifted, x, y, z, bottom, tilt_deg, speed, touch_dist, "
    "dist_to(<fixture>)). Every number can instead be a formula (a string) over obj_x, obj_y, obj_z (object centre), "
    "obj_top, obj_bottom, obj_yaw_deg, each part's <part>_x, <part>_y, <part>_z, and the scene's names (each "
    "fixture's <name>_x, <name>_y, <name>_top, <name>_floor; carry_z; lowest_z), with + - * / abs min max hypot sqrt, "
    "read when that step starts."
)


# ------------------------------------------------------------------ the robot's parts
@dataclass
class Part:
    name: str
    body: int  # the body whose frame carries the tip point (the wrist, for a hand)
    offset: np.ndarray  # the tip point in that body's frame
    joints: list[int]  # the joints that move it (its chain), each driven by a position servo
    acts: list[int]  # the servo per joint
    gears: list[float]
    axis: np.ndarray  # which way it points, in the body's frame (body origin -> tip)
    fingers: tuple | None = None  # a hand: its two fingertip bodies and their tip offsets (for "turn")
    bodies: set = field(default_factory=set)  # every body of the part (what can touch)
    can_grip: bool = False


def _servo_joints(m: mujoco.MjModel) -> dict[int, tuple[int, float]]:
    """joint -> (its position servo, gear)."""
    out = {}
    for a in range(m.nu):
        if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and is_position(m, a):
            j = int(m.actuator_trnid[a, 0])
            if int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                out[j] = (a, float(m.actuator_gear[a, 0]) or 1.0)
    return out


def _chain_joints(m: mujoco.MjModel, body: int, servo: dict) -> list[int]:
    out = []
    b = body
    while b > 0:
        for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
            if j in servo:
                out.append(j)
        b = int(m.body_parentid[b])
    return out[::-1]


def _subtree(m: mujoco.MjModel, root: int) -> set:
    out = {root}
    for b in range(root + 1, m.nbody):
        p = b
        while p > 0 and p not in out:
            p = int(m.body_parentid[p])
        if p in out:
            out.add(b)
    return out


def _tip(m, d, body: int, offset) -> np.ndarray:
    return d.xpos[body] + d.xmat[body].reshape(3, 3) @ offset


def robot_parts(m: mujoco.MjModel, d: mujoco.MjData, ix, task: dict) -> dict[str, Part]:
    """Every end part of the robot (fingertips, tool, feet: indices().tips) by a plain name, plus "hand": the point
    between the gripper's fingers, if it has one. d must be at the home pose."""
    servo = _servo_joints(m)
    parts: dict[str, Part] = {}
    grip_bodies: set = set()
    g = task.get("gripper") or {}
    if ix.grip_act >= 0:  # the gripper's fingers: the tips that move when the gripper motor moves
        d2 = mujoco.MjData(m)
        d2.qpos[:] = d.qpos
        mujoco.mj_forward(m, d2)
        before = {int(b): _tip(m, d2, int(b), ix.tip_off[k]).copy() for k, b in enumerate(ix.tips)}
        d2.ctrl[:] = d.ctrl
        d2.ctrl[ix.grip_act] = ix.grip_close
        for _ in range(int(0.6 / m.opt.timestep)):
            mujoco.mj_step(m, d2)
        moved = [int(b) for k, b in enumerate(ix.tips) if np.linalg.norm(_tip(m, d2, int(b), ix.tip_off[k]) - before[int(b)]) > 0.004]
        tips = {int(b): ix.tip_off[k] for k, b in enumerate(ix.tips)}
        if len(moved) == 1:  # a single jaw closing against a fixed one (the part it hangs from): that is the other finger
            jaw = int(m.body_parentid[moved[0]])
            tips[jaw] = _far_point(m, d, jaw)
            moved.append(jaw)
        if len(moved) >= 2:
            pts = {b: _tip(m, d, b, tips[b]) for b in moved}
            a, b2 = max(((x, y) for x in moved for y in moved if x < y), key=lambda p: np.linalg.norm(pts[p[0]] - pts[p[1]]))
            wrist = _common_ancestor(m, [a, b2])
            if wrist in (a, b2):  # a fixed jaw is the wrist itself: the hand hangs from it
                wrist = a if wrist == a else b2
            R, p0 = d.xmat[wrist].reshape(3, 3), d.xpos[wrist]
            mid = (pts[a] + pts[b2]) / 2
            off = R.T @ (mid - p0)
            js = _chain_joints(m, wrist, servo)
            grip_bodies = _subtree(m, wrist)
            parts["hand"] = Part("hand", wrist, off, js, [servo[j][0] for j in js], [servo[j][1] for j in js],
                                 off / (np.linalg.norm(off) + 1e-9), ((a, tips[a]), (b2, tips[b2])), grip_bodies,
                                 can_grip=True)
    for k, b in enumerate(ix.tips):
        b = int(b)
        if b in grip_bodies:
            continue  # a finger: moved with the hand
        name = m.body(b).name.removeprefix("robot_")
        js = _chain_joints(m, b, servo)
        if not js:
            continue
        off = np.array(ix.tip_off[k])
        pivot = d.xanchor[m.body_jntadr[b]] if m.body_jntnum[b] else d.xpos[int(m.body_parentid[b])]
        far = _tip(m, d, b, off)
        if m.body_jntnum[b] and int(m.jnt_type[m.body_jntadr[b]]) == int(mujoco.mjtJoint.mjJNT_HINGE):
            jax = d.xaxis[m.body_jntadr[b]]
            v = far - pivot
            if abs(v @ jax) > 0.7 * np.linalg.norm(v):  # a part that spins about its own length (a tool flange, a
                # wrist): its tip is the middle of its end face, on the axis, not a point on the rim
                far = pivot + jax * (v @ jax)
                off = d.xmat[b].reshape(3, 3).T @ (far - d.xpos[b])
        ax = d.xmat[b].reshape(3, 3).T @ (far - pivot)
        parts[name] = Part(name, b, np.array(off), js, [servo[j][0] for j in js], [servo[j][1] for j in js],
                           ax / (np.linalg.norm(ax) + 1e-9), None, _subtree(m, b))
    return parts


def _far_point(m, d, b: int) -> np.ndarray:
    """A body's tip in its own frame: the corner of its shapes farthest from the joint it turns about."""
    pivot = d.xanchor[m.body_jntadr[b]] if m.body_jntnum[b] else d.xpos[int(m.body_parentid[b])]
    best, far = d.xpos[b].copy(), -1.0
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b:
            continue
        c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        for k in np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c:
            w = p + R @ k
            if np.linalg.norm(w - pivot) > far:
                best, far = w, float(np.linalg.norm(w - pivot))
    return d.xmat[b].reshape(3, 3).T @ (best - d.xpos[b])


def _common_ancestor(m, bodies: list[int]) -> int:
    def path(b):
        out = []
        while b > 0:
            out.append(b)
            b = int(m.body_parentid[b])
        return out
    common = set(path(bodies[0]))
    for b in bodies[1:]:
        common &= set(path(b))
    return max(common, key=lambda b: len(path(b))) if common else bodies[0]


# ------------------------------------------------------------------ moving a part (inverse kinematics on the real model)
class BodyRunner:
    """The task's robot in its scene (accurate physics), with its parts and a damped-least-squares solver that puts
    a part's tip at a point, optionally pointing a given way and with the fingers turned."""

    def __init__(self, task: dict):
        from .sim import SimulationEnvironment
        self.task = task
        self.m = build_task_model(task, accurate=True)
        self.ix = indices(self.m, task)
        self.env = SimulationEnvironment(self.m, control_dt=CTRL_DT)
        self.d = self.env.data
        self.scratch = mujoco.MjData(self.m)
        self.reset(0, task["start"], None)
        self.parts = robot_parts(self.m, self.d, self.ix, task)
        self.home_q = self.d.qpos.copy()

    def reset(self, seed: int, start, params) -> None:
        from .sim import PhysicalParams
        m, ix, task = self.m, self.ix, self.task
        self.env.reset(seed, start, params or PhysicalParams())
        d = self.d = self.env.data
        d.qpos[ix.home_adr] = ix.home_val
        if ix.base >= 0:
            mx, my, myaw = task["mount"]
            d.qpos[ix.base_q:ix.base_q + 7] = [mx, my, task.get("base_z", 0.4), math.cos(myaw / 2), 0, 0, math.sin(myaw / 2)]
        self.ctrl = np.clip(ix.q0, ix.act_lo, ix.act_hi).copy()
        d.ctrl[:] = self.ctrl
        mujoco.mj_forward(m, d)
        t = task["target"]
        self.env.set_goal_marker(t["x"], t["y"], t["yaw"])

    def tip(self, part: Part, d=None) -> np.ndarray:
        d = d if d is not None else self.d
        return _tip(self.m, d, part.body, part.offset)

    def pointing(self, part: Part, d=None) -> np.ndarray:
        d = d if d is not None else self.d
        return d.xmat[part.body].reshape(3, 3) @ part.axis

    def finger_axis(self, part: Part, d=None) -> np.ndarray | None:
        if not part.fingers:
            return None
        d = d if d is not None else self.d
        (a, oa), (b, ob) = part.fingers
        v = _tip(self.m, d, b, ob) - _tip(self.m, d, a, oa)
        return v / (np.linalg.norm(v) + 1e-9)

    def ik(self, part: Part, target, point=None, across=None, q_start=None, iters: int = 80):
        """Joint values (part.joints order) that put the tip at target, pointing along `point` (unit vector) and with
        the fingers along `across` (unit, horizontal), as far as the body allows. Returns (q, position error m,
        pointing error rad)."""
        m, d2 = self.m, self.scratch
        d2.qpos[:] = self.d.qpos
        adr = [m.jnt_qposadr[j] for j in part.joints]
        dof = [m.jnt_dofadr[j] for j in part.joints]
        if q_start is not None:
            d2.qpos[adr] = q_start
        lo = np.array([m.jnt_range[j][0] if m.jnt_limited[j] else -1e9 for j in part.joints])
        hi = np.array([m.jnt_range[j][1] if m.jnt_limited[j] else 1e9 for j in part.joints])
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        target = np.asarray(target, float)
        pe, ae = 9.0, 0.0
        for _ in range(iters):
            mujoco.mj_kinematics(m, d2)
            mujoco.mj_comPos(m, d2)
            p = self.tip(part, d2)
            mujoco.mj_jac(m, d2, jp, jr, p, part.body)
            rows, errs = [jp[:, dof]], [target - p]
            pe = float(np.linalg.norm(target - p))
            ae = 0.0
            if point is not None:
                a = self.pointing(part, d2)
                rows.append(0.25 * (-_skew(a) @ jr[:, dof]))
                errs.append(0.25 * (point - a))
                ae = math.acos(float(np.clip(a @ point, -1, 1)))
            if across is not None and part.fingers:
                f = self.finger_axis(part, d2)
                f = f if f @ across >= 0 else -f  # either finger can be on either side
                rows.append(0.15 * (-_skew(f) @ jr[:, dof]))
                errs.append(0.15 * (across - f))
            J, e = np.vstack(rows), np.concatenate(errs)
            if pe < 0.002 and ae < 0.03 and (across is None or np.linalg.norm(errs[-1]) < 0.01):
                break
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(len(e)), e)
            dq = np.clip(dq, -0.2, 0.2)
            d2.qpos[adr] = np.clip(d2.qpos[adr] + dq, lo, hi)
        return d2.qpos[adr].copy(), pe, ae

    def reconfigure(self, part: Part, target, point=None, across=None, tries: int = 12, seed: int = 0):
        """Joint values for the end pose found from other starting arrangements (home, random), or None."""
        rng = np.random.default_rng(seed)
        m = self.m
        lo = np.array([m.jnt_range[j][0] if m.jnt_limited[j] else -math.pi for j in part.joints])
        hi = np.array([m.jnt_range[j][1] if m.jnt_limited[j] else math.pi for j in part.joints])
        home = self.home_q[[m.jnt_qposadr[j] for j in part.joints]] if hasattr(self, "home_q") else (lo + hi) / 2
        best, key = None, None
        for k in range(tries):
            q0 = home if k == 0 else rng.uniform(lo, hi)
            q, pe, ae = self.ik(part, target, point, across, q0, iters=150)
            if pe < 0.01 and ae < 0.1:
                cur = self.d.qpos[[m.jnt_qposadr[j] for j in part.joints]]
                score = float(np.max(np.abs(q - cur)))  # the nearest arrangement that works
                if key is None or score < key:
                    best, key = q, score
        return best

    def touches_object(self) -> bool:
        """Is any part of the robot touching the object right now?"""
        m, d = self.m, self.d
        robot, blk = set(self.ix.robot_geoms.tolist()), set(self.ix.block_geoms.tolist())
        for i in range(d.ncon):
            g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
            if (g1 in blk and g2 in robot) or (g2 in blk and g1 in robot):
                return True
        return False

    def command(self, part: Part, q) -> None:
        for a, gr, v in zip(part.acts, part.gears, q):
            self.ctrl[a] = np.clip(gr * v, self.ix.act_lo[a], self.ix.act_hi[a])


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


# ------------------------------------------------------------------ checking a program
def check_body_program(steps: list, parts: dict, joints: list[str], known: set = frozenset(), fixtures=None) -> list[dict]:
    """The program in Morph's form; ProgramError says what's wrong in words Nemotron can act on."""
    if not isinstance(steps, list) or not steps:
        raise ProgramError("the program has no steps")
    if len(steps) > MAX_STEPS:
        raise ProgramError(f"too many steps ({len(steps)}; at most {MAX_STEPS})")
    names = set(known) | {f"{p}_{k}" for p in parts for k in "xyz"}
    from .programs import normalize_step
    out = []
    for i, st in enumerate(steps, 1):
        st = normalize_step(st)
        if not isinstance(st, dict):
            raise ProgramError(f"step {i} isn't an object")
        op = st.get("do") or st.get("op")
        if op not in BODY_OPS:
            raise ProgramError(f"step {i}: unknown building block {op!r} (known: {', '.join(sorted(BODY_OPS))})")
        s = {"do": op}
        if op in ("move", "push"):
            part = st.get("part", "hand" if "hand" in parts else None)
            if part not in parts:
                raise ProgramError(f"step {i}: part must be one of {', '.join(parts)} (got {part!r})")
            s["part"] = part
            s["speed"] = st.get("speed", "slow" if op == "push" else "normal")
            if s["speed"] not in SPEEDS:
                raise ProgramError(f"step {i}: speed must be slow, normal or fast")
        if op == "move":
            if not any(k in st for k in ("x", "y", "z")):
                raise ProgramError(f"step {i}: move needs at least one of x, y, z, written in the step itself: "
                                   f"{{\"do\": \"move\", \"part\": \"hand\", \"x\": ..., \"z\": ...}}")
            for k in ("x", "y", "z"):
                s[k] = _formula(st[k], f"step {i} {k}", names) if k in st else f"{s['part']}_{k}"
            pt = st.get("point", "keep")
            if isinstance(pt, list):
                if len(pt) != 3 or not all(isinstance(v, (int, float)) for v in pt) or not any(pt):
                    raise ProgramError(f"step {i}: point as a direction needs three numbers [dx, dy, dz]")
            elif pt not in set(POINTS) | {"keep", "object"}:
                raise ProgramError(f"step {i}: point must be down, up, north, south, east, west, object, keep or [dx, dy, dz]")
            s["point"] = pt
            turn = st.get("turn", "keep")
            if turn not in ("keep", "object"):
                turn = _formula(turn, f"step {i} turn", names)
            if turn != "keep" and not parts[s["part"]].fingers:
                raise ProgramError(f"step {i}: turn only applies to a part that grips ({s['part']} doesn't)")
            s["turn"] = turn
            if st.get("until") not in (None, "touch"):
                raise ProgramError(f"step {i}: until can only be \"touch\"")
            if st.get("until"):
                s["until"] = "touch"
        elif op == "push":
            s["direction"] = _formula(st.get("direction", 0.0), f"step {i} direction", names)
            s["distance"] = _formula(st.get("distance", 0.1), f"step {i} distance", names)
        elif op == "grip":
            how = st.get("how") or st.get("action") or st.get("state") or st.get("mode")
            if how is None:
                how = next((v for v in st.values() if v in ("close", "open")), None)
            if how is None and isinstance(st.get("close"), bool):
                how = "close" if st["close"] else "open"
            if how not in ("close", "open"):
                raise ProgramError(f"step {i}: grip needs \"close\" or \"open\", written {{\"do\": \"grip\", \"how\": \"close\"}}")
            if not any(p.can_grip for p in parts.values()):
                raise ProgramError(f"step {i}: this robot has no gripper")
            s["how"] = how
        elif op == "joints":
            vals = st.get("joints") or st.get("values") or {k: v for k, v in st.items() if k not in ("do", "op", "seconds")}
            if not isinstance(vals, dict) or not vals:
                raise ProgramError(f"step {i}: joints needs {{joint: radians}}")
            bad = [k for k in vals if k not in joints]
            if bad:
                raise ProgramError(f"step {i}: unknown joint(s) {', '.join(bad)} (known: {', '.join(joints)})")
            s["joints"] = {k: float(v) for k, v in vals.items()}
            s["seconds"] = min(max(float(st.get("seconds", 1.0)), 0.2), 5.0)
        elif op in ("home", "wait"):
            s["seconds"] = min(max(float(st.get("seconds", 1.0 if op == "home" else 0.5)), 0.0), 5.0)
        elif op == "repeat_if":
            s.update(check_repeat(st, i, fixtures))
        out.append(s)
    return out


def check_repeat(st: dict, i: int, fixtures) -> dict:
    """repeat_if(condition, back_to, times): a reactive step. If the condition (a goal_terms formula over what is
    measured right now: held, lifted, z, x, y, tilt_deg, speed, touch_dist, ...) holds, go back to step back_to, at most
    `times` times (e.g. it slipped out: grasp again)."""
    from .goal_terms import parse
    cond = st.get("condition") or st.get("if")
    try:
        parse(str(cond), fixtures or {})
    except Exception as e:  # noqa: BLE001
        raise ProgramError(f"step {i}: repeat_if condition {cond!r} can't be measured: {e}") from None
    back = st.get("back_to", st.get("step"))
    if not isinstance(back, int) or not 1 <= back < i:
        raise ProgramError(f"step {i}: repeat_if needs back_to: an earlier step's number (1..{i - 1})")
    return {"condition": str(cond), "back_to": back, "times": int(min(max(int(st.get("times", 2)), 1), 3))}


def describe(s: dict) -> str:
    if s["do"] == "move":
        bits = [f"move the {s['part']} to ({s['x']}, {s['y']}, {s['z']})"]
        if s["point"] != "keep":
            bits.append(f"pointing {s['point']}")
        if s["turn"] != "keep":
            bits.append("fingers across the object" if s["turn"] == "object" else f"turned {s['turn']}°")
        bits.append(s["speed"])
        if s.get("until"):
            bits.append("until it touches")
        return ", ".join(bits)
    if s["do"] == "push":
        return f"push with the {s['part']} {s['distance']} m towards {s['direction']}°, {s['speed']}"
    if s["do"] == "grip":
        return f"{s['how']} the gripper"
    if s["do"] == "joints":
        return "set " + ", ".join(f"{k} to {v:.2f}" for k, v in s["joints"].items())
    if s["do"] == "repeat_if":
        return f"if {s['condition']}: back to step {s['back_to']} (at most {s['times']}x)"
    return f"{s['do']} {s.get('seconds', 0):.1f} s"


# ------------------------------------------------------------------ running a program
class _Values:
    def __init__(self, rn: BodyRunner, half_z: float, extra: dict):
        p = rn.env.block_pose()
        self.v = dict(extra) | {"obj_x": p.x, "obj_y": p.y, "obj_z": p.z, "obj_top": p.z + half_z, "obj_bottom": p.z - half_z,
                                "obj_yaw_deg": math.degrees(p.yaw)}
        for n, part in rn.parts.items():
            t = rn.tip(part)
            self.v |= {f"{n}_x": t[0], f"{n}_y": t[1], f"{n}_z": t[2]}

    def __call__(self, f) -> float:
        from .programs import _Values
        return _Values.__call__(self, f)  # the same safe evaluator, over these names


def _direction(spec, rn: BodyRunner, part: Part, val) -> np.ndarray | None:
    if spec == "keep":
        return None
    if spec == "object":
        p = rn.env.block_pose()
        v = np.array([p.x, p.y, p.z]) - rn.tip(part)
    elif isinstance(spec, list):
        v = np.array(spec, float)
    else:
        v = np.array(POINTS[spec], float)
    return v / (np.linalg.norm(v) + 1e-9)


def _touching(rn: BodyRunner, bodies: set) -> set:
    m, d = rn.m, rn.d
    blk = set(rn.ix.block_geoms.tolist())
    out = set()
    for i in range(d.ncon):
        g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
        b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
        if b1 in bodies or b2 in bodies:
            o = g2 if b1 in bodies else g1
            ob = int(m.geom_bodyid[o])
            if ob in bodies:
                continue
            out.add("object" if o in blk else "floor" if ob == 0 else
                    "robot" if m.body(ob).name.startswith("robot_") else m.body(ob).name)
    return out


def run_body_program(rn: BodyRunner, steps: list[dict], goal: GoalSpec, scene, settle_s: float = 1.5, on_step=None,
                     names: dict | None = None, record: list | None = None) -> dict:
    """Run a checked program; returns success, the checklist results and, per step, what was measured."""
    from .task_model import measure
    m, ix = rn.m, rn.ix
    b = scene.block
    hist = History([c.model_dump() for c in goal.conditions], rn.task.get("fixtures", {}), (b.half_x, b.half_y, b.half_z))
    state = {"t": 0.0, "touched": set()}
    names = names or {}
    grippers = [p for p in rn.parts.values() if p.can_grip]

    def tick():
        rn.env.advance(rn.ctrl)
        d = rn.d
        if record is not None:
            record.append((d.qpos.copy(), d.qvel.copy(), rn.ctrl.copy()))
        state["t"] += CTRL_DT
        for p in rn.parts.values():
            state["touched"] |= _touching(rn, p.bodies)
        v = measure(np, d.xpos, d.xmat, d.qvel, ix, rn.task, held=contact_held(m, d, ix), ctrl=d.ctrl)
        hist.update(v)

    def fell() -> bool:
        if ix.base < 0:
            return False
        return rn.d.xmat[ix.base].reshape(3, 3)[2, 2] < 0.5

    report = []
    q_part = {n: rn.d.qpos[[m.jnt_qposadr[j] for j in p.joints]].copy() for n, p in rn.parts.items()}
    jumps: dict = {}
    nxt = 0
    while nxt < len(steps):
        i, st = nxt + 1, steps[nxt]
        nxt += 1
        if st["do"] == "repeat_if":  # reactive: measure now, and go back if the condition holds
            from .goal_terms import parse, violation
            v = measure(np, rn.d.xpos, rn.d.xmat, rn.d.qvel, ix, rn.task, held=contact_held(m, rn.d, ix), ctrl=rn.d.ctrl)
            holds = float(violation(np, parse(st["condition"], rn.task.get("fixtures", {})), v)) <= 1e-6
            if holds and jumps.get(i, 0) < st["times"]:
                jumps[i] = jumps.get(i, 0) + 1
                nxt = st["back_to"] - 1
                report.append(f"{i}. {describe(st)}: it holds, going back to step {st['back_to']} (time {jumps[i]})")
            else:
                report.append(f"{i}. {describe(st)}: " + ("it holds, but it was already repeated "
                                                          f"{st['times']} times: carrying on" if holds else "it doesn't hold: carrying on"))
            continue
        if state["t"] > MAX_TIME:
            report.append(f"{i}. not run: out of time ({MAX_TIME:.0f} s of simulation)")
            break
        val = _Values(rn, b.half_z, names)
        before = rn.env.block_pose()
        state["touched"] = set()
        if on_step:
            on_step(i, describe(st))
        note = ""
        if st["do"] in ("move", "push"):
            part = rn.parts[st["part"]]
            p0 = rn.tip(part)
            if st["do"] == "move":
                target = np.array([val(st["x"]), val(st["y"]), val(st["z"])])
                point = _direction(st["point"], rn, part, val)
                across = None
                if st["turn"] == "object":
                    yaw = before.yaw
                    f = rn.finger_axis(part)
                    cands = [np.array([math.cos(yaw + k * math.pi / 2), math.sin(yaw + k * math.pi / 2), 0.0]) for k in range(2)]
                    across = max(cands, key=lambda c: abs(c @ f)) if f is not None else cands[0]
                elif st["turn"] != "keep":
                    a = math.radians(val(st["turn"]))
                    across = np.array([math.cos(a), math.sin(a), 0.0])
            else:
                a = math.radians(val(st["direction"]))
                target = p0 + val(st["distance"]) * np.array([math.cos(a), math.sin(a), 0.0])
                point, across = None, None
            a0 = rn.pointing(part)
            if st["do"] == "move" and (point is not None or across is not None):
                # can the straight path get there from how the limbs are now? If not, but the end pose is possible from
                # another arrangement of the joints (elbow up instead of down), swing there joint by joint first
                q_end, pe_end, ae_end = rn.ik(part, target, point, across, q_part[st["part"]], iters=150)
                if pe_end > 0.01 or ae_end > 0.1:
                    best = rn.reconfigure(part, target, point, across)
                    if best is not None:
                        q0_ = rn.d.qpos[[rn.m.jnt_qposadr[j] for j in part.joints]].copy()
                        steps_n = max(10, int(np.max(np.abs(best - q0_)) / 0.03))
                        for k in range(steps_n):
                            rn.command(part, q0_ + (best - q0_) * (k + 1) / steps_n)
                            tick()
                        q_part[st["part"]] = best
                        note += "turned its joints around to get there (the straight way was blocked by its joint limits); "
                        p0, a0 = rn.tip(part), rn.pointing(part)
            dist = float(np.linalg.norm(target - p0))
            turn_ang = math.acos(float(np.clip(a0 @ point, -1, 1))) if point is not None else 0.0
            n = max(1, math.ceil(max(dist / SPEEDS[st["speed"]], turn_ang / 0.05)))
            worst, touched_at = 0.0, None
            hold = point is None and st["do"] == "push"
            hold_dir = a0 if hold else None
            for k in range(n):
                f = (k + 1) / n
                p = p0 + (target - p0) * f
                pt = None
                if point is not None:
                    pt = a0 + (point - a0) * f
                    pt = pt / (np.linalg.norm(pt) + 1e-9)
                elif hold_dir is not None:
                    pt = hold_dir
                q, pe, ae = rn.ik(part, p, pt, across, q_part[st["part"]])
                worst = max(worst, pe)
                q_part[st["part"]] = q
                rn.command(part, q)
                tick()
                if st.get("until") == "touch" and rn.touches_object():
                    touched_at = f
                    break
                if fell():
                    break
            for _ in range(25):  # let the servos catch up (0.5 s at most)
                if touched_at is not None or np.linalg.norm(rn.tip(part) - target) < 0.005:
                    break
                tick()
            err = float(np.linalg.norm(rn.tip(part) - target))
            pre, note = note, ""
            if touched_at is not None:
                note = f"touched the {scene.object_name} {touched_at:.0%} of the way there (stopped)"
            elif worst > 0.01:
                note = (f"the {st['part']} can't get there: it stays {worst * 100:.0f} cm short at worst (out of reach or "
                        f"the pointing asked for isn't possible there); ended {err * 100:.1f} cm from the point")
            else:
                note = f"reached it (within {err * 100:.1f} cm)" if err < 0.015 else f"ended {err * 100:.1f} cm from it (blocked)"
            note = pre + note
            if point is not None:
                miss = math.degrees(math.acos(float(np.clip(rn.pointing(part) @ point, -1, 1))))
                if miss > 10:
                    note += f"; it points {miss:.0f}° off the direction asked for"
        elif st["do"] == "grip":
            rn.ctrl[ix.grip_act] = ix.grip_close if st["how"] == "close" else ix.grip_open
            last = float(rn.d.actuator_length[ix.grip_act])
            for k in range(40):
                tick()
                now = float(rn.d.actuator_length[ix.grip_act])
                if k > 5 and abs(now - last) < 1e-4:
                    break
                last = now
            held = contact_held(m, rn.d, ix) > 0.5 or "object" in _touching(rn, grippers[0].bodies)
            if st["how"] == "close":
                note = ("the fingers are on the object" if held else
                        f"the fingers closed on nothing: the {scene.object_name} is "
                        f"{np.linalg.norm(rn.tip(grippers[0]) - np.array([before.x, before.y, before.z])) * 100:.0f} cm "
                        "from the point between them")
            else:
                note = "the gripper is open"
        elif st["do"] in ("joints", "home"):
            if st["do"] == "home":
                goal_q = {j: rn.home_q[m.jnt_qposadr[j]] for p in rn.parts.values() for j in p.joints}
            else:
                goal_q = {m.joint("robot_" + k).id: v for k, v in st["joints"].items()}
            servo = _servo_joints(m)
            start = {j: float(rn.d.qpos[m.jnt_qposadr[j]]) for j in goal_q}
            n = max(1, int(st["seconds"] / CTRL_DT))
            for k in range(n):
                f = (k + 1) / n
                for j, v in goal_q.items():
                    if j in servo:
                        a, gr = servo[j]
                        rn.ctrl[a] = np.clip(gr * (start[j] + (v - start[j]) * f), ix.act_lo[a], ix.act_hi[a])
                tick()
            for nm, p in rn.parts.items():
                q_part[nm] = rn.d.qpos[[m.jnt_qposadr[j] for j in p.joints]].copy()
            note = "done"
        else:
            for _ in range(int(st["seconds"] / CTRL_DT)):
                tick()
            note = "waited"
        after = rn.env.block_pose()
        moved = math.hypot(after.x - before.x, after.y - before.y)
        bottom = max(0.0, after.z - (abs(math.cos(after.tilt)) * b.half_z + abs(math.sin(after.tilt)) * b.half_y))
        obj = f"; the {scene.object_name} is at ({after.x:.2f}, {after.y:.2f}), bottom {bottom * 100:.0f} cm up"
        if moved > 0.01:
            obj += f" (moved {moved * 100:.0f} cm)"
        if math.degrees(after.tilt) > 15 and getattr(b, "shape", "box") != "sphere":
            obj += f", tipped {math.degrees(after.tilt):.0f}°"
        if contact_held(m, rn.d, ix) > 0.5:
            obj += ", held"
        hit = sorted(state["touched"] - {"robot"})
        if hit:
            obj += f"; the robot touched: {', '.join(x if x != 'object' else scene.object_name for x in hit)}"
        if fell():
            obj += "; THE ROBOT FELL OVER"
        report.append(f"{i}. {describe(st)}: {note}{obj}")
        if fell():
            break
    for _ in range(int(settle_s / CTRL_DT)):
        tick()
    pose = rn.env.block_pose()
    held = contact_held(m, rn.d, ix) > 0.5
    res = check(goal, scene, m, rn.d, pose, rn.env.block_speed(), not held, history=hist)
    if fell():
        res["success"] = False
        res["reasons"].insert(0, "the robot fell over")
    from .goals import placement
    pl = placement(goal, scene)
    miss = None
    if pl.get("has_place"):
        dx, dy = pose.x - pl["x"], pose.y - pl["y"]
        miss = math.hypot(dx, dy)
        if miss > 0.02:
            ns = f"{abs(dy) * 100:.0f} cm {'north' if dy > 0 else 'south'}" if abs(dy) > 0.01 else ""
            ew = f"{abs(dx) * 100:.0f} cm {'east' if dx > 0 else 'west'}" if abs(dx) > 0.01 else ""
            report.append(f"in the end the {scene.object_name} is at ({pose.x:.2f}, {pose.y:.2f}): {miss * 100:.0f} cm from "
                          f"where it had to end ({', '.join(x for x in (ns, ew) if x)} of it)")
    return {"success": res["success"], "results": res["results"], "reasons": res["reasons"], "report": report,
            "sim_s": round(state["t"], 1), "miss_m": miss}


# ------------------------------------------------------------------ the grip: firm enough to lift the object
def squeeze_force(rn: BodyRunner) -> float:
    """Total force (N) the gripper's fingers press on the object with, right now."""
    m, d = rn.m, rn.d
    hand = next((p for p in rn.parts.values() if p.can_grip), None)
    if hand is None:
        return 0.0
    blk = set(rn.ix.block_geoms.tolist())
    f6, total = np.zeros(6), 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in blk and int(m.geom_bodyid[g2]) in hand.bodies) or (g2 in blk and int(m.geom_bodyid[g1]) in hand.bodies):
            mujoco.mj_contactForce(m, d, i, f6)
            total += abs(f6[0])
    return total


def calibrate_grip(rn: BodyRunner, scene, scales=(1, 2, 4, 8, 16), log=print) -> float:
    """Close the hand on the object where it starts and measure the squeeze; stiffen the gripper's servo (its force
    limit unchanged) until it squeezes hard enough to lift the object with margin (3 x its weight over friction),
    then one step more, as the arm setup does. Returns the scale (also applied to rn's model)."""
    hand = next((p for p in rn.parts.values() if p.can_grip), None)
    if hand is None:
        return 1.0
    m, a = rn.m, rn.ix.grip_act
    g0 = (m.actuator_gainprm[a, 0], m.actuator_biasprm[a, 1], m.actuator_biasprm[a, 2])
    b = scene.block
    need = 3.0 * b.mass * 9.81 / max(2 * b.friction, 0.2)
    if getattr(b, "shape", "box") != "box":  # round: flat fingers touch it at points and it squeezes out easily
        need *= 3.0
    probe = [{"do": "grip", "how": "open"},
             {"do": "move", "part": hand.name, "x": "obj_x", "y": "obj_y", "z": "obj_top + 0.06", "point": "down", "turn": "object"},
             {"do": "move", "part": hand.name, "z": "obj_z", "speed": "slow"}, {"do": "grip", "how": "close"}]
    from .goals import GoalSpec
    found, forces = scales[-1], []
    for k in scales:
        m.actuator_gainprm[a, 0], m.actuator_biasprm[a, 1], m.actuator_biasprm[a, 2] = g0[0] * k, g0[1] * k, g0[2] * k
        rn.reset(0, scene.start, None)
        run_body_program(rn, check_body_program(probe, rn.parts, []), GoalSpec(object=scene.object_name, conditions=[]), scene, settle_s=0.3)
        f = squeeze_force(rn)
        forces.append(round(f, 1))
        if f >= need:
            found = k
            break
        if f < 0.05:  # the fingers aren't on it at all (the hand couldn't get around it here): stiffer won't help
            found = 1
            break
    nxt = [k for k in scales if k > found]
    found = nxt[0] if nxt and forces[-1] >= need else found  # a step of margin: real grasps land off-centre
    m.actuator_gainprm[a, 0], m.actuator_biasprm[a, 1], m.actuator_biasprm[a, 2] = g0[0] * found, g0[1] * found, g0[2] * found
    if forces[-1] < 0.05:
        log("  grip test: the hand couldn't close around the object from straight above here (gripper left as it is)")
    else:
        log(f"  grip test: squeeze {forces[-1]} N (needs about {need:.0f} N to lift it): gripper servo x{found:g}")
    return float(found)


# ------------------------------------------------------------------ write, run, fix
def part_radius(rn: BodyRunner, part: Part) -> float:
    """How far the part's shapes reach sideways from the line through its tip along its pointing: what has to clear the
    object when it comes down beside it."""
    m, d = rn.m, rn.d
    tip, ax = rn.tip(part), rn.pointing(part)
    near_tip = {b for b in part.bodies if np.linalg.norm(d.xpos[b] - tip) < 0.25}
    r = 0.0
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) not in near_tip or m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        for k in np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c:
            w = p + R @ k - tip
            if w @ ax > -0.06:  # only the end of the part (not the whole forearm behind it)
                r = max(r, float(np.linalg.norm(w - (w @ ax) * ax)))
    return r


def body_facts(rn: BodyRunner, scene, goal: GoalSpec) -> dict:
    from .goals import placement, release_spot
    p = placement(goal, scene)
    spot = release_spot(goal, scene)
    m, b = rn.m, scene.block
    parts = {}
    for n, part in rn.parts.items():
        t = rn.tip(part)
        root = part.joints[0] if part.joints else None
        base = rn.d.xanchor[root] if root is not None else t
        reach = sum(np.linalg.norm(rn.d.xpos[int(m.jnt_bodyid[j])] - rn.d.xpos[int(m.body_parentid[m.jnt_bodyid[j]])])
                    for j in part.joints[1:]) + np.linalg.norm(part.offset)
        parts[n] = {"tip_now": [round(float(v), 3) for v in t], "pointing_now": [round(float(v), 2) for v in rn.pointing(part)],
                    "radius_m": round(part_radius(rn, part), 3),
                    "can_grip": part.can_grip, "moved_by_joints": len(part.joints),
                    "reach_from": [round(float(v), 3) for v in base], "reach_m_about": round(float(reach), 2)}
    return {"robot_stands_at": [round(float(v), 3) for v in rn.task["mount"][:2]],
            "robot_floating_base": rn.ix.base >= 0, "parts": parts,
            "joints": {m.joint(j).name.removeprefix("robot_"): [round(float(v), 2) for v in m.jnt_range[j]]
                       for p in rn.parts.values() for j in p.joints},
            "object_half_size_m": [b.half_x, b.half_y, b.half_z], "object_start": [round(float(v), 3) for v in scene.start],
            "where_it_must_end": {k: round(float(p[k]), 3) for k in ("x", "y", "support_z")} if p.get("has_place") else None,
            "where_to_let_go": [round(v, 3) for v in spot] if spot else None,
            "tips": ["a part is radius_m wide around its tip: to stand it beside the object without landing on it, keep its "
                     "tip at least (object half size + radius_m + 0.02) from the object's centre, come down there, then "
                     "move in sideways until it touches",
                     "to push it, put the part against the side of the object facing away from where it must go, low on "
                     "its face (z about obj_bottom + 0.02), then push", "to tip it over, push high on its face (near obj_top)",
                     "to grip: open, move the hand above it pointing down with turn 'object', come down slowly to obj_z, close"]}


def _cache_key(task: dict, goal: GoalSpec, scene) -> Path:
    key = json.dumps([task["robot"], [c.model_dump() for c in goal.conditions], str(getattr(scene, "path", "")),
                      list(scene.start), [round(float(v), 3) for v in task["mount"]]], sort_keys=True, default=str)
    return CACHE / f"body_{task['robot']}__{hashlib.sha1(key.encode()).hexdigest()[:12]}.json"


def solve_body(pj, goal: GoalSpec, task_path: Path, llm, log=print, emit=lambda **e: None, n: int = 5,
               attempts: int = 5) -> dict:
    """Nemotron writes a program for this body from the general building blocks; Morph runs it on the learner's own
    model, says what happened, and repeats until the checklist is met. A program that works is run from n slightly
    different starts and the successful runs are kept as demonstrations (recorded on the task model)."""
    from .goals import explain
    from .introspect import profile_robot
    from .project import _params, _start_poses
    from .skill_steps import robot_brief
    from .task_model import load_task
    scene = pj.scene
    task = load_task(task_path)
    name = task["robot"]
    emit(type="stage", robot=name, stage="setup", text="finding the robot's parts for Nemotron's program")
    rn = BodyRunner(task)
    if not rn.parts:
        return {"success": 0, "n": 0, "setup_error": "no part of this robot can be moved to a point (no position servos)"}
    if any(p.can_grip for p in rn.parts.values()) and not task.get("gripper", {}).get("calibrated"):
        k = calibrate_grip(rn, scene, log=log)  # the learner trains with the same gripper
        task["gripper"]["gain_scale"] = float(task["gripper"].get("gain_scale", 1.0)) * k
        task["gripper"]["calibrated"] = True
        task_path.write_text(json.dumps(task, indent=1))
        rn = BodyRunner(task)
    prof = profile_robot(task["robot_xml"])
    joints = [m_j for p in rn.parts.values() for m_j in (rn.m.joint(j).name.removeprefix("robot_") for j in p.joints)]
    top = max([f.top_z for f in scene.fixtures.values()] + [2 * scene.block.half_z])
    from .goals import placement as _placement
    pl = _placement(goal, scene)
    names = scene_names(scene, top + 0.12, 0.01, (pl["x"], pl["y"]) if pl.get("has_place") else None)
    facts = body_facts(rn, scene, goal)
    facts["names_for_formulas"] = {k: round(v, 3) for k, v in names.items()}
    checklist = [f"{c.describe()}: {explain(c, scene)}" for c in goal.conditions]
    tries, steps, ok_run = [], None, None
    cache = _cache_key(task, goal, scene)
    if cache.exists():
        try:
            cached = check_body_program(json.loads(cache.read_text())["program"], rn.parts, joints, set(names),
                                        task.get("fixtures"))
            rn.reset(0, scene.start, _params(0))
            run = run_body_program(rn, cached, goal, scene, names=names)
            if run["success"]:
                log("  using the program that worked last time (no new Nemotron call)")
                steps, ok_run, attempts = cached, run, 0
        except (ProgramError, KeyError, ValueError):
            pass
    a, fixes = 0, 0
    while a < attempts:
        emit(type="stage", robot=name, stage="trials", text=f"Nemotron is writing program {a + 1}")
        try:
            prop, rec = llm.write_program(robot_brief(prof), checklist, scene.geometry(), facts, BLOCKS_DOC, tries,
                                          body="any")
        except Exception as e:  # noqa: BLE001
            log(f"  Nemotron's reply couldn't be read ({e}); asking again")
            fixes += 1
            if fixes > 3:
                break
            continue
        try:
            steps = check_body_program(prop.steps, rn.parts, joints, set(names), task.get("fixtures"))
        except ProgramError as e:
            tries.append({"program": prop.steps, "problem": f"Morph couldn't run it: {e}"})
            log(f"  program {a + 1}: can't run it: {e}")
            fixes += 1
            if fixes > 3:
                a += 1
            continue
        log(f"  program {a + 1} ({rec.latency_s}s): " + "; ".join(describe(s) for s in steps))
        emit(type="program", robot=name, attempt=a + 1, steps=[describe(s) for s in steps], rationale=prop.rationale)
        rn.reset(0, scene.start, _params(0))
        run = run_body_program(rn, steps, goal, scene, names=names,
                               on_step=lambda i, d, k=a: emit(type="caption", robot=name, text=f"Program {k + 1}, step {i}: {d}"))
        from .programs import feedback
        tries.append({"program": steps, "rationale": prop.rationale, "result": feedback(run)})
        emit(type="program_result", robot=name, attempt=a + 1, success=run["success"], report=run["report"],
             reasons=run["reasons"])
        log(f"  program {a + 1}: {'reached the goal' if run['success'] else 'failed: ' + '; '.join(run['reasons'][:2])}")
        if run["success"]:
            ok, r2 = 0, None
            for j, st0 in enumerate(_start_poses(scene, 3)[1:], 1):
                rn.reset(j, st0, _params(j))
                r = run_body_program(rn, steps, goal, scene, names=names)
                ok += r["success"]
                r2 = r if not r["success"] else r2
            if ok == 0:
                run = dict(run, success=False)
                tries[-1]["result"]["other_problems"].append(
                    f"it worked from the scene's start, but failed when the {scene.object_name} started about 2 cm away "
                    f"or turned 10°: {'; '.join(r2['reasons'][:2])}. Make it follow the object (obj_x, obj_y, ...)")
        if run["success"]:
            ok_run = run
            break
        a += 1
    out = {"program_attempts": tries}
    if ok_run is None:
        return out | {"success": 0, "n": 0, "why_not": f"Nemotron's programs didn't reach the goal in {a} tries"}
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"robot": name, "goal": goal.describe(), "program": steps}, indent=1))
    emit(type="stage", robot=name, stage="trials", text="trying the program from slightly different starts")
    trials, demos = [], []
    for i, start in enumerate(_start_poses(scene, n)):
        emit(type="trial_start", robot=name, i=i, n=n)
        rn.reset(i, start, _params(i))
        rec_: list = []
        r = run_body_program(rn, steps, goal, scene, names=names, record=rec_)
        if r["success"]:
            demos.append(rec_)
        trials.append({"success": r["success"], "conditions": r["results"], "reasons": r["reasons"]})
        emit(type="trial", robot=name, i=i, n=n, success=r["success"], conditions=r["results"], reasons=r["reasons"])
        log(f"  trial {i + 1}: {'success' if r['success'] else 'fail: ' + '; '.join(r['reasons'][:2])}")
    ok = sum(t["success"] for t in trials)
    log(f"{name}: {ok}/{n} with Nemotron's program")
    return out | {"success": ok, "n": n, "trials": trials, "program": steps, "demos": demos, "body_program": True}
