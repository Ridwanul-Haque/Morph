"""Goals as measurable conditions on the object.

A goal is a list of conditions, all of which must hold when the object has come to rest and the
robot has let go:

  at_position(x, y, tol)   object centre within tol of (x, y)
  heading(yaw, tol)        arrow / object +x axis within tol of yaw
  upright(max_tilt)        top face still up
  on_top_of(surface)       resting on that fixture's top, inside its outline
  inside(container)        resting on the container floor, inside its walls
  touching(fixture)        in contact with (or within 5 mm of) that fixture
  near / beside(fixture)   close to it / on one side of it

and, for everything else a person might ask (measured in goal_terms, learned rather than scripted):
  orientation, height, away_from, moved, "not <relation>", and expression (any formula over measured
  quantities, e.g. "z > 0.3 and tilt_deg < 20")

Conditions come from a demonstration (measured), a description (Nemotron or the rule parser),
or both. They are checked in the simulation the same way for every robot, and `placement` turns
them into the concrete target the skills work to: where the object goes, how high it rests,
what the robot must lift it over.
"""

from __future__ import annotations

import math
from typing import Literal

import mujoco
import numpy as np
from pydantic import BaseModel, Field

from .geometry import wrap_angle, yaw_error
from .task import Field_, Goal, TaskSpec
from .user_scene import UserScene

Kind = Literal["at_position", "heading", "upright", "on_top_of", "inside", "touching", "near", "beside",
               "orientation", "height", "away_from", "moved", "expression"]
RELATIONS = ("on_top_of", "inside", "touching", "near", "beside")
SIDES = {"east": (1, 0), "west": (-1, 0), "north": (0, 1), "south": (0, -1)}


class Condition(BaseModel):
    kind: Kind
    target: str | None = None  # fixture name for on_top_of / inside / touching
    x: float | None = None
    y: float | None = None
    yaw_deg: float | None = None
    tol: float | None = None  # m for at_position / near, deg for heading / upright
    side: str | None = None  # beside / moved: north | south | east | west
    negate: bool = False  # "not" a relation: not inside / not on top of / not touching / not near
    up_face: str | None = None  # orientation: top | bottom | side | +x | -x | +y | -y
    min_z: float | None = None  # height: the object's bottom at least this high (m)
    hold: bool = False  # height: the robot may keep holding it (no need to let go)
    expr: str | None = None  # expression: a formula over measured quantities (see goal_terms)
    text: str | None = None  # expression: what it means in words
    # when it must hold: at the end (default), at the moment the robot lets go (throwing), the whole time,
    # or at some point during the task
    # ... or for the last `seconds` of the try, without a break (keep it spinning, keep it balanced)
    when: Literal["end", "release", "always", "ever", "last"] = "end"
    seconds: float | None = None  # when = "last": for how long (default 2 s)
    source: Literal["demo", "description", "default"] = "default"

    def describe(self) -> str:
        base = self._describe()
        return {"end": base, "release": f"when it lets go: {base}", "always": f"the whole time: {base}",
                "ever": f"at some point: {base}", "last": f"for the last {self.seconds or 2.0:.0f} s: {base}"}[self.when]

    def _describe(self) -> str:
        k = self.kind
        if k == "at_position":
            return f"centre within {self.tol * 100:.0f} cm of ({self.x:.2f}, {self.y:.2f})"
        if k == "heading":
            names = {0: "east", 45: "north-east", 90: "north", 135: "north-west", 180: "west", -180: "west",
                     -135: "south-west", -90: "south", -45: "south-east"}
            name = names.get(int(round(self.yaw_deg)) if abs(self.yaw_deg - round(self.yaw_deg)) < 0.5 else None)
            return f"facing {name + f' ({self.yaw_deg:.0f}°)' if name else f'{self.yaw_deg:.0f}°'} ±{self.tol:.0f}°"
        if k == "upright":
            return f"stays upright (not tipped over, tilt under {self.tol:.0f}°)"
        if k == "near":
            return f"near the {self.target} (within {(self.tol or 0.1) * 100:.0f} cm)"
        if k == "beside" and not self.negate:
            return f"beside the {self.target}, on its {self.side} side"
        if k == "orientation":
            face = {"top": "the right way up", "bottom": "upside down", "side": "lying on its side"}.get(
                self.up_face or "top", f"with its {self.up_face} face up")
            return f"{face} (within {self.tol or 20:.0f}°)"
        if k == "height":
            return (f"lifted, its bottom at least {(self.min_z or 0.1) * 100:.0f} cm above the floor"
                    + (" (the robot may keep holding it)" if self.hold else ""))
        if k == "away_from":
            return f"at least {(self.tol or 0.3) * 100:.0f} cm away from the {self.target}"
        if k == "moved":
            return f"moved at least {(self.tol or 0.2) * 100:.0f} cm" + (f" {self.side}" if self.side else " from where it started")
        if k == "expression":
            return f"{self.text} ({self.expr})" if self.text else self.expr
        rel = {"on_top_of": "on top of", "inside": "inside", "touching": "touching", "near": "near",
               "beside": "beside"}[k] + f" the {self.target}"
        return ("not " + rel) if self.negate else rel


class GoalSpec(BaseModel):
    object: str
    conditions: list[Condition] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)  # possible but unlikely: Morph still tries

    def describe(self) -> list[str]:
        return [f"{c.describe()}  [{c.source}]" for c in self.conditions]


def warnings(goal: GoalSpec, scene: UserScene) -> list[str]:
    """Things that can be tried but will probably fail (said plainly, with the measurements)."""
    out = []
    b = scene.block
    ow = 2 * min(b.half_x, b.half_y) * 100
    for c in goal.conditions:
        f = scene.fixtures.get(c.target or "")
        if f is None:
            continue
        if c.kind == "on_top_of":
            tw = min(f.top_size) * 100
            if tw < 0.8 * ow:
                what = "rim" if f.kind == "container" else "top"
                msg = (f"the {c.target}'s {what} is only {tw:.0f} cm wide and the {scene.object_name} is {ow:.0f} cm "
                       f"across, so balancing it there will very likely fail; Morph will still try")
                if f.kind == "container":
                    msg += f" (if you meant inside the {c.target}, say \"in\")"
                out.append(msg)
    return out


def validate(goal: GoalSpec, scene: UserScene) -> list[str]:
    """Why a goal can't work in this scene, in physical terms (sizes, openings), not categories."""
    errs = []
    b = scene.block
    obj = scene.object_name
    ow = 2 * min(b.half_x, b.half_y) * 100  # object's narrow side, cm
    for c in goal.conditions:
        if c.kind in RELATIONS:
            f = scene.fixtures.get(c.target or "")
            if f is None:
                errs.append(f"there's nothing called {c.target!r} in the scene (it has: {', '.join(sorted(scene.fixtures))})")
                continue
            if c.kind == "inside" and f.kind != "container":
                errs.append(f"the {c.target} is solid ({f.geometry()['footprint_cm'][0]:.0f} x {f.geometry()['footprint_cm'][1]:.0f} cm, "
                            f"{f.geometry()['height_cm']:.0f} cm tall), there's no inside to put the {obj} in")

            if c.kind == "beside" and c.side not in SIDES:
                errs.append(f"beside the {c.target}: which side? (north, south, east or west)")
        if c.kind == "at_position" and (c.x is None or c.y is None):
            errs.append("at_position needs x and y")
        if c.kind == "heading" and c.yaw_deg is None:
            errs.append("heading needs yaw_deg")
        if c.kind == "away_from" and c.target not in scene.fixtures:
            errs.append(f"there's nothing called {c.target!r} in the scene (it has: {', '.join(sorted(scene.fixtures))})")
        if c.kind == "orientation" and (c.up_face or "top") not in ("top", "bottom", "side", "+x", "-x", "+y", "-y"):
            errs.append(f"orientation: {c.up_face!r} isn't a face (top, bottom, side, +x, -x, +y, -y)")
        if c.kind == "moved" and c.side and c.side not in SIDES:
            errs.append(f"moved: {c.side!r} isn't a direction (north, south, east, west)")
        if c.kind == "expression":
            from .goal_terms import ExprError, parse
            try:
                parse(c.expr or "", list(scene.fixtures))
            except ExprError as e:
                errs.append(f"the goal formula {c.expr!r}: {e}")
    ends_at = {c.target for c in goal.conditions if c.kind in ("on_top_of", "inside", "touching", "near", "beside")
               and not c.negate and c.when == "end"}
    for c in goal.conditions:
        if c.when == "end" and ((c.kind == "away_from" and c.target in ends_at)
                                or (c.negate and c.kind in ("inside", "on_top_of", "touching") and c.target in ends_at)):
            errs.append(f"the {obj} can't end up both {c.describe()} and at the {c.target}")
    places = [c for c in goal.conditions if c.kind in ("on_top_of", "inside") and not c.negate and c.when == "end"]
    if len(places) > 1:
        errs.append("the object can only end up in one place: " + ", ".join(c.describe() for c in places)
                    + " (for one place on the way, give it when = ever: at some point)")
    return errs


# ------------------------------------------------------------------ from conditions to a target
def placement(goal: GoalSpec, scene: UserScene) -> dict:
    """Target x, y, heading and tolerances, the height the object must rest at, and what it has to
    clear on the way (for the skills and the capability check)."""
    b = scene.block
    reach = math.hypot(b.half_x, b.half_y)
    x, y = scene.start[0], scene.start[1]
    pos_tol, yaw, yaw_tol = 0.05, scene.start[2], math.pi
    support, clear, lift, why = 0.0, 0.0, False, []
    has_place = False
    for c in goal.conditions:
        if c.kind in ("inside", "on_top_of") and not c.negate and c.when == "end":
            has_place = True
            f = scene.fixtures[c.target]
            lo, hi = (f.inner_lo, f.inner_hi) if c.kind == "inside" else (f.top_lo, f.top_hi)
            x, y = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)
            # the whole footprint has to fit: tolerance is what's left after the object's half-diagonal
            pos_tol = max(0.005, float(min(hi - lo)) / 2 - reach)
            support = f.rest_z if c.kind == "inside" else f.top_z
            clear = max(clear, f.top_z)
            if f.top_z > 0.01:
                lift = True
                why.append(f"{c.target}'s {'walls' if c.kind == 'inside' else 'top'} at {f.top_z * 100:.0f} cm")
    for c in goal.conditions:
        if c.kind == "touching" and not c.negate and c.when == "end":
            has_place = True
            f = scene.fixtures[c.target]
            if any(k.kind in ("inside", "on_top_of") for k in goal.conditions):
                continue  # resting place already fixed; touching is then just checked
            # the face of the fixture nearest the object, and the spot just in front of it
            gap = 0.003
            px, py = x, y
            # the spot in front of each face, clamped to the face so the whole footprint meets it
            ym = (float(np.clip(py, f.lo[1] + b.half_y, f.hi[1] - b.half_y)) if f.hi[1] - f.lo[1] > 2 * b.half_y
                  else (f.lo[1] + f.hi[1]) / 2)
            xm = (float(np.clip(px, f.lo[0] + b.half_x, f.hi[0] - b.half_x)) if f.hi[0] - f.lo[0] > 2 * b.half_x
                  else (f.lo[0] + f.hi[0]) / 2)
            cands = [(f.lo[0] - b.half_x - gap, ym), (f.hi[0] + b.half_x + gap, ym),
                     (xm, f.lo[1] - b.half_y - gap), (xm, f.hi[1] + b.half_y + gap)]
            x, y = min(cands, key=lambda c: math.hypot(c[0] - px, c[1] - py))
            pos_tol = 0.02  # the contact itself is what gets checked (within 5 mm)
    for c in goal.conditions:
        if c.kind in ("near", "beside") and not c.negate and c.when == "end" and not any(
                k.kind in ("inside", "on_top_of", "at_position") and not k.negate and k.when == "end" for k in goal.conditions):
            has_place = True
            f = scene.fixtures[c.target]
            gap = 0.04
            if c.kind == "beside":
                dx, dy = SIDES[c.side]
                cx, cy = (f.lo[0] + f.hi[0]) / 2, (f.lo[1] + f.hi[1]) / 2
                x = float(f.hi[0] + b.half_x + gap if dx > 0 else f.lo[0] - b.half_x - gap if dx < 0 else cx)
                y = float(f.hi[1] + b.half_y + gap if dy > 0 else f.lo[1] - b.half_y - gap if dy < 0 else cy)
                pos_tol = 0.04
            else:  # near: the closest spot just outside the fixture (edge-to-edge gap), from where the object starts
                px, py = x, y
                qx, qy = float(np.clip(px, f.lo[0], f.hi[0])), float(np.clip(py, f.lo[1], f.hi[1]))
                d = np.array([px - qx, py - qy])
                n = d / (np.linalg.norm(d) + 1e-9) if np.linalg.norm(d) > 1e-6 else np.array([-1.0, 0.0])
                tol = c.tol or 0.1
                gap = min(0.03, tol / 2)
                half = max(b.half_x, b.half_y)
                x, y = float(qx + n[0] * (half + gap)), float(qy + n[1] * (half + gap))
                pos_tol = max(0.01, tol - gap)
    for c in goal.conditions:
        if c.when != "end":
            continue  # about a moment on the way: not where it ends up
        if c.kind == "at_position":
            x, y, pos_tol = float(c.x), float(c.y), float(c.tol or 0.05)
            has_place = True
        elif c.kind == "heading":
            yaw, yaw_tol = math.radians(c.yaw_deg), math.radians(c.tol or 15)
        elif c.kind == "moved" and c.side in SIDES and not has_place:  # a direction gives a spot to aim for
            sx, sy = SIDES[c.side]
            d = (c.tol or 0.2) + 0.03
            x, y, pos_tol, has_place = x + sx * d, y + sy * d, 0.05, True
        elif c.kind == "height":
            lift = True
            why.append(f"lifted {(c.min_z or 0.1) * 100:.0f} cm")
    return {"x": x, "y": y, "yaw": wrap_angle(yaw), "pos_tol": pos_tol, "yaw_tol": yaw_tol, "support_z": support,
            "clear_z": clear, "needs_lift": lift, "lift_reason": "; ".join(why), "has_place": has_place}


def needs_learning(goal: GoalSpec) -> str | None:
    """Why no ready-made skill can do this goal (they carry or push the object to a spot and let go)."""
    from .goal_terms import is_generic
    odd = [c for c in goal.conditions if is_generic(c.model_dump())]
    if not odd:
        return None
    return ("the ready-made skills carry or push the object to a spot and let go; this goal asks for "
            + "; ".join(c.describe() for c in odd) + ", so Nemotron writes a program for it instead (and if that "
            "doesn't work out, it can be learned)")


def lets_go(goal: GoalSpec) -> bool:
    """Must the robot have let go at the end? Not when the goal says it may hold on."""
    return not any((c.kind == "height" and c.hold) or (c.kind == "expression" and "held" in (c.expr or ""))
                   for c in goal.conditions)


def to_taskspec(goal: GoalSpec, scene: UserScene, instruction: str = "") -> TaskSpec:
    p = placement(goal, scene)
    prov = "user"
    spec = TaskSpec(goal=Goal(
        x=Field_(value=p["x"], provenance=prov), y=Field_(value=p["y"], provenance=prov),
        yaw=Field_(value=p["yaw"], provenance=prov),
        pos_tol_m=Field_(value=p["pos_tol"], provenance=prov), yaw_tol_rad=Field_(value=p["yaw_tol"], provenance=prov),
        keep_upright=Field_(value=True, provenance="default"),
        support_z=Field_(value=p["support_z"], provenance=prov, note="from the goal conditions")),
        instruction=instruction)
    return spec


# ------------------------------------------------------------------ checking
def check(goal: GoalSpec, scene: UserScene, model: mujoco.MjModel, data: mujoco.MjData, pose, speed: float,
          released: bool, history=None) -> dict:
    """Every condition measured on the final state (and, for conditions about a moment during the task,
    from history: goal_terms.History). Returns {success, results: [...], reasons}."""
    b = scene.block
    results = []
    bottom = pose.z - b.half_z

    def inside_xy(lo, hi, margin=0.0):
        return lo[0] - margin <= pose.x <= hi[0] + margin and lo[1] - margin <= pose.y <= hi[1] + margin

    gv, fx = None, None
    for c in goal.conditions:
        ok, got = False, ""
        cd = c.model_dump()
        if c.when != "end":
            i = goal.conditions.index(c)
            val = history.result().get(i, math.inf) if history is not None else math.inf
            snap = history.snap.get(i) if history is not None else None
            if c.when == "release" and (history is None or not history.released):
                got = "it never let go" if history is not None else "not followed during this run"
            else:
                got = ("at that moment: " + _measured_text(c, snap)) if snap else "never happened"
            results.append({"condition": c.describe(), "ok": bool(val <= 1e-6), "measured": got})
            continue
        if _generic(cd):
            if gv is None:
                gv, fx = _measure(goal, scene, model, data, pose, released)
            try:
                viol = float(condition_violation(np, cd, gv, fx, (b.half_x, b.half_y, b.half_z)))
                ok, got = viol <= 1e-6, _measured_text(c, gv)
            except KeyError as e:  # a robot quantity where there's no robot to measure
                ok, got = False, f"can't measure {e.args[0]} here"
            results.append({"condition": c.describe(), "ok": bool(ok), "measured": got})
            continue
        if c.kind == "at_position":
            e = math.hypot(pose.x - c.x, pose.y - c.y)
            ok, got = e <= c.tol, f"{e * 100:.1f} cm away"
        elif c.kind == "heading":
            e = math.degrees(yaw_error(pose.yaw, math.radians(c.yaw_deg)))
            ok, got = e <= c.tol, f"{e:.0f} deg off"
        elif c.kind == "upright":
            t = math.degrees(pose.tilt)
            ok, got = t <= c.tol, f"tilt {t:.0f} deg"
        elif c.kind in ("inside", "on_top_of"):
            f = scene.fixtures[c.target]
            lo, hi = (f.inner_lo, f.inner_hi) if c.kind == "inside" else (f.top_lo, f.top_hi)
            within = inside_xy(lo, hi)
            surface = f.rest_z if c.kind == "inside" else f.top_z
            resting = abs(bottom - surface) < 0.012
            ok = within and resting
            got = f"centre {'inside' if within else 'outside'} the outline, bottom at {bottom * 100:.1f} cm " \
                  f"(surface {surface * 100:.1f} cm)"
        elif c.kind in ("near", "beside"):
            f = scene.fixtures[c.target]
            dx = max(f.lo[0] - pose.x, 0, pose.x - f.hi[0])
            dy = max(f.lo[1] - pose.y, 0, pose.y - f.hi[1])
            dist = math.hypot(dx, dy) - max(b.half_x, b.half_y)  # object's edge to the fixture's outline
            if c.kind == "near":
                ok, got = dist <= (c.tol or 0.1) and not inside_xy(f.lo[:2], f.hi[:2]), f"{max(dist, 0) * 100:.0f} cm from it"
            else:
                sx, sy = SIDES[c.side]
                on_side = (pose.x > f.hi[0] if sx > 0 else pose.x < f.lo[0] if sx < 0 else True) and \
                          (pose.y > f.hi[1] if sy > 0 else pose.y < f.lo[1] if sy < 0 else True)
                lateral = (f.lo[1] - 0.05 <= pose.y <= f.hi[1] + 0.05) if sx else (f.lo[0] - 0.05 <= pose.x <= f.hi[0] + 0.05)
                ok = on_side and lateral and dist <= 0.2
                got = f"{'on' if on_side else 'not on'} the {c.side} side, {max(dist, 0) * 100:.0f} cm away"
        elif c.kind == "touching":
            d = _distance_to_fixture(model, data, c.target)
            ok, got = d <= 0.005, ("in contact" if d <= 0.0005 else f"{d * 1000:.0f} mm apart")
        results.append({"condition": c.describe(), "ok": bool(ok), "measured": got})
    reasons = [f"{r['condition']}: {r['measured']}" for r in results if not r["ok"]]
    if speed > 0.02 and (released or lets_go(goal)):
        reasons.append(f"object still moving ({speed:.3f})")
    if not released and lets_go(goal):
        reasons.append("robot still holding the object")
    return {"success": not reasons, "results": results, "reasons": reasons}


def _generic(cd: dict) -> bool:
    from .goal_terms import is_generic
    return is_generic(cd)


def condition_violation(xp, cd, v, fx, half):
    from .goal_terms import condition_violation as cv
    return cv(xp, cd, v, fx, half)


def _measure(goal, scene, model, data, pose, released) -> tuple[dict, dict]:
    """The quantities goal_terms reads, from the final simulation state."""
    from .goal_terms import fixtures_geometry, quantities
    b = scene.block
    bid = model.body("block").id
    j = model.joint("block_free")
    fx = fixtures_geometry(scene)
    robot = [i for i in range(1, model.nbody) if model.body(i).name.startswith("robot_")]
    pos = data.xpos[bid].copy()
    kw = {}
    if robot:
        kw = {"robot_xy": data.xpos[robot[0]][:2].copy(),
              "hand_dist": float(min(np.linalg.norm(data.xpos[i] - pos) for i in robot))}
    body = robot_body(model, data)
    p = placement(goal, scene)
    v = quantities(np, pos=pos, R=data.xmat[bid].reshape(3, 3).copy(), vel=data.qvel[j.dofadr[0]:j.dofadr[0] + 3].copy(),
                   ang=data.qvel[j.dofadr[0] + 3:j.dofadr[0] + 6].copy(), start=scene.start[:2], half=(b.half_x, b.half_y, b.half_z), fixtures=fx,
                   held=0.0 if released else 1.0,
                   goal=(p["x"], p["y"], p["support_z"] + b.half_z) if p.get("has_place") else None, **kw)
    v |= body
    return v, fx


def robot_body(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    """The robot's own body in any of Morph's models (with or without the robot_ prefix): how far each of its joints is
    through its range (0..1) and where each of its end parts is, for goals about the robot itself."""
    bid = model.body("block").id
    joints, parts = {}, {}
    for j in range(model.njnt):
        if int(model.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        lo, hi = model.jnt_range[j] if model.jnt_limited[j] and model.jnt_range[j][1] > model.jnt_range[j][0] else (-np.pi, np.pi)
        joints[model.joint(j).name.removeprefix("robot_")] = float((data.qpos[model.jnt_qposadr[j]] - lo) / (hi - lo))
    has_child = {int(model.body_parentid[b]) for b in range(1, model.nbody)}
    moving = set()  # bodies with a joint at or above them (the robot, not the fixtures)
    for b in range(1, model.nbody):
        a = b
        while a > 0:
            if model.body_jntnum[a] and a != bid:
                moving.add(b)
                break
            a = int(model.body_parentid[a])
    for b in moving - has_child - {bid}:
        if any(model.geom_bodyid[g] == b for g in range(model.ngeom)) or model.body_jntnum[b]:
            parts[model.body(b).name.removeprefix("robot_")] = data.xpos[b].copy()
    return {"joints": joints, "parts": parts}


def _measured_text(c: Condition, v: dict) -> str:
    k = c.kind
    if k in ("orientation", "upright"):
        return f"tilted {float(v['tilt_deg']):.0f}°"
    if k == "height":
        return f"bottom at {float(v['bottom']) * 100:.1f} cm"
    if k == "away_from":
        return f"{float(v['dist'][c.target]) * 100:.0f} cm away"
    if k == "moved":
        return f"moved {float(v['travelled']) * 100:.0f} cm"
    if k == "expression":
        from .goal_terms import uses
        names = sorted(uses(c.expr) & (set(v) - {"dist", "R"}))
        dists = [n for n in ("dist_to",) if n in uses(c.expr)]
        parts = [f"{n} = {float(v[n]):.3g}" for n in names]
        if dists:
            parts += [f"dist_to('{t}') = {float(d):.3g}" for t, d in v["dist"].items() if t in c.expr]
        parts += _body_text(c.expr, v)
        return ", ".join(parts)
    return f"centre at ({float(v['x']):.2f}, {float(v['y']):.2f}), {float(v['dist'].get(c.target, 0)) * 100:.0f} cm from it"


def _body_text(expr: str, v: dict) -> list[str]:
    """What the robot-body functions in a formula measured, or that this robot has nothing of that name."""
    import ast as _ast
    from .goal_terms import BODY_FUNCS, body_names, body_value, normalize
    out = []
    for n in _ast.walk(_ast.parse(normalize(expr), mode="eval")):
        if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name) and n.func.id in BODY_FUNCS and n.args:
            pat = getattr(n.args[0], "value", "")
            kind = "joints" if n.func.id.startswith("joint") else "parts"
            hits = body_names(str(pat), v.get(kind) or {})
            if not hits:
                have = ", ".join(sorted(v.get(kind) or {})[:12]) or "none"
                out.append(f"this robot has no {kind[:-1]} matching '{pat}' (its {kind}: {have})")
            else:
                out.append(f"{n.func.id}('{pat}') = {float(body_value(np, n.func.id, str(pat), v)):.2f} "
                           f"({', '.join(hits[:6])}{'…' if len(hits) > 6 else ''})")
    return out


def _distance_to_fixture(m: mujoco.MjModel, d: mujoco.MjData, fixture: str) -> float:
    bg = m.geom("block_geom").id
    fb = m.body(fixture).id
    best = 1.0
    fromto = np.zeros(6)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == fb:
            best = min(best, mujoco.mj_geomDistance(m, d, bg, g, 0.05, fromto))
    return float(best)


# ------------------------------------------------------------------ measuring a goal from a final pose
def measure_relations(scene: UserScene, x: float, y: float, z: float, yaw: float, tilt: float) -> list[Condition]:
    """What an object resting at this pose is doing relative to the scene (used for demonstrations)."""
    b = scene.block
    bottom = z - b.half_z
    out = []
    for name, f in scene.fixtures.items():
        if f.kind == "container" and f.inner_lo[0] <= x <= f.inner_hi[0] and f.inner_lo[1] <= y <= f.inner_hi[1] \
                and abs(bottom - f.rest_z) < 0.015:
            out.append(Condition(kind="inside", target=name, source="demo"))
        elif f.kind == "surface" and f.lo[0] <= x <= f.hi[0] and f.lo[1] <= y <= f.hi[1] and abs(bottom - f.rest_z) < 0.015:
            out.append(Condition(kind="on_top_of", target=name, source="demo"))
    if not out:  # touching: footprint (as a disc) against the fixture's outline
        r = math.hypot(b.half_x, b.half_y)
        for name, f in scene.fixtures.items():
            dx = max(f.lo[0] - x, 0, x - f.hi[0])
            dy = max(f.lo[1] - y, 0, y - f.hi[1])
            if math.hypot(dx, dy) <= r + 0.008 and bottom < f.top_z:
                out.append(Condition(kind="touching", target=name, source="demo"))
    return out


# ------------------------------------------------------------------ the checklist in plain words
def explain(c: Condition, scene: UserScene) -> str:
    """What exactly counts for one checklist item, with this scene's measurements, and how it's checked."""
    text = _explain(c, scene)
    when = {"release": " Checked at the moment the robot lets go of it (the last time it releases it), not at the end.",
            "always": " Checked at every moment from start to end: one slip fails it.",
            "ever": " It only has to happen once, at any moment during the try.",
            "last": f" It has to hold without a break for the last {c.seconds or 2.0:.0f} s of the try (keep it up)."}.get(c.when, "")
    return (text + when).strip()


def _explain(c: Condition, scene: UserScene) -> str:
    obj = scene.object_name
    b = scene.block
    size = f"{2 * b.half_x * 100:.0f} × {2 * b.half_y * 100:.0f} × {2 * b.half_z * 100:.0f} cm"
    f = scene.fixtures.get(c.target or "")
    cm = lambda v: f"{v * 100:.0f} cm"
    k = c.kind
    if k in ("inside", "on_top_of", "touching", "near", "beside") and f is None:
        return ""
    if c.negate and f is not None:
        what = {"inside": f"its centre ends up outside the {c.target}'s walls",
                "on_top_of": f"it no longer rests on the {c.target}'s top: its centre is off the top face",
                "touching": f"there's a gap of at least 2 cm between it and the {c.target}",
                "near": f"it ends up more than {cm(c.tol or 0.1)} from the {c.target}",
                "beside": f"it ends up more than {cm(c.tol or 0.1)} from the {c.target}"}[k]
        return f"The {obj} starts wherever the scene puts it; this passes when {what}."
    if k == "inside":
        lo, hi = f.inner_lo, f.inner_hi
        return (f"The {obj} ({size}) rests on the {c.target}'s floor, {cm(f.rest_z)} up, with its centre inside the "
                f"inner walls ({(hi[0] - lo[0]) * 100:.0f} × {(hi[1] - lo[1]) * 100:.0f} cm). It may stick out above the "
                f"{cm(f.top_z)} walls, like a tall bottle in a short box.")
    if k == "on_top_of":
        tw, td = f.top_size
        return (f"The {obj} rests on the {c.target}'s top face ({tw * 100:.0f} × {td * 100:.0f} cm, {cm(f.top_z)} up): its "
                f"bottom sits on that face (within 1.2 cm) and its centre is over it, so it isn't hanging off the edge.")
    if k == "touching":
        return (f"Some part of the {obj} is within 5 mm of the {c.target}: pushed up against it, not just near it. "
                f"Measured as the smallest gap between their shapes.")
    if k == "near":
        return (f"The {obj}'s nearest edge is within {cm(c.tol or 0.1)} of the {c.target}'s outline (edge to edge, not "
                f"centre to centre), without being on it or in it.")
    if k == "beside":
        return (f"The {obj} ends up on the {c.side} side of the {c.target}, within 20 cm of it and alongside it "
                f"(not off past one of its corners).")
    if k == "at_position":
        return f"The {obj}'s centre ends up within {cm(c.tol or 0.05)} of the point ({c.x:.2f}, {c.y:.2f}) m on the floor."
    if k == "heading":
        return (f"The arrow on the {obj} (its front) ends up pointing at {c.yaw_deg:.0f}° give or take {c.tol or 15:.0f}° "
                f"(0° is east, 90° north, 180° west). Measured from its final rotation.")
    if k == "upright":
        return (f"Its top face still faces up: tilted less than {c.tol:.0f}° from standing straight. Lying on its side "
                f"would be 90°, upside down 180°.")
    if k == "orientation":
        face = c.up_face or "top"
        what = {"top": "standing the right way up", "bottom": "upside down: its top face points at the floor",
                "side": "lying on one of its sides: its top face points sideways"}.get(face, f"with its {face} face up")
        return f"The {obj} ends up {what}, within {c.tol or 20:.0f}°, and has settled there (not still rolling)."
    if k == "height":
        return (f"The {obj}'s lowest point is at least {cm(c.min_z or 0.1)} above the floor at the end"
                + (", and the robot may still be holding it up (it doesn't have to let go)." if c.hold else
                   ": resting on something that high, since the robot has to let go."))
    if k == "away_from":
        return f"The gap between the {obj} and the {c.target} (edge to edge) is at least {cm(c.tol or 0.3)} at the end."
    if k == "moved":
        return (f"The {obj}'s centre ends at least {cm(c.tol or 0.2)} from where it started"
                + (f", measured along the {c.side} direction." if c.side else ", in any direction."))
    if k == "expression":
        return (f"{c.text + '. ' if c.text else ''}Checked as the formula {c.expr}, over what Morph measures "
                f"(positions in metres, angles in degrees, held = the robot is holding it).")
    return ""


def checked_when(goal: GoalSpec) -> str:
    return ("Checked at the end of each try, once the object has stopped moving"
            + (" (the robot may still be holding it)." if not lets_go(goal) else " and the robot has let go."))


def release_spot(goal: GoalSpec, scene: UserScene) -> tuple[float, float] | None:
    """For goals about where the robot lets go (throwing, dropping): the spot nearest where the object has to
    end up (or its start) at which every release condition holds, on a 5 cm grid over the scene."""
    rel = [c.model_dump() | {"when": "end"} for c in goal.conditions if c.when == "release"]
    if not rel:
        return None
    import json as _json
    key = (scene.path, _json.dumps(rel, sort_keys=True), tuple(scene.start), _json.dumps(goal.model_dump(), sort_keys=True))
    if key in _SPOTS:
        return _SPOTS[key]
    p = placement(goal, scene)
    near = (p["x"], p["y"]) if p.get("has_place") else tuple(scene.start[:2])  # aim: as close to where it must end up
    _SPOTS[key] = _release_spot(rel, scene, near)
    return _SPOTS[key]


_SPOTS: dict = {}


def _release_spot(rel: list[dict], scene: UserScene, near) -> tuple[float, float] | None:
    from .goal_terms import fixtures_geometry, quantities
    b = scene.block
    fx = fixtures_geometry(scene)
    xs = [scene.start[0]] + [v for f in fx.values() for v in (f["lo"][0], f["hi"][0])]
    ys = [scene.start[1]] + [v for f in fx.values() for v in (f["lo"][1], f["hi"][1])]
    best, best_d = None, math.inf
    for gx in np.arange(min(xs) - 2.0, max(xs) + 2.0, 0.05):
        for gy in np.arange(min(ys) - 2.0, max(ys) + 2.0, 0.05):
            for gz in (b.half_z + 0.25, b.half_z + 0.6):  # held up in the air when it's let go
                v = quantities(np, pos=np.array([gx, gy, gz]), R=np.eye(3), vel=np.zeros(3), start=scene.start[:2],
                               half=(b.half_x, b.half_y, b.half_z), fixtures=fx, held=1.0)
                try:
                    ok = all(float(condition_violation(np, c, v, fx, (b.half_x, b.half_y, b.half_z))) <= 1e-6 for c in rel)
                except KeyError:  # a condition about the robot itself: can't place it from here
                    return None
                dd = math.hypot(gx - near[0], gy - near[1])
                if ok and dd < best_d:
                    best, best_d = (float(gx), float(gy)), dd
    return best


def reach_problem(prof, goal: GoalSpec, scene: UserScene) -> str | None:
    """A robot bolted to one spot can't pick the object up and also let go of it somewhere farther than its
    arm spans. Said with the numbers, and what kind of robot could."""
    if prof.base != "fixed":
        return None
    spot = release_spot(goal, scene)
    if spot is None:
        return None
    reach = max((v["max_horizontal"] for v in prof.reach_m.values()), default=0.0)
    d = math.hypot(spot[0] - scene.start[0], spot[1] - scene.start[1])
    if d <= 1.8 * reach:
        return None
    what = "; ".join(c.text or c.describe() for c in goal.conditions if c.when == "release")
    return (f"to {what}, the {scene.object_name} has to be let go at about ({spot[0]:.2f}, {spot[1]:.2f}), {d:.2f} m from "
            f"where it starts, but {prof.name} is bolted down and reaches {reach:.2f} m: from any one spot it can't "
            f"both pick it up and get there. A robot that can walk or drive (or a longer arm) could")


def about_body(c: "Condition") -> bool:
    """Is this condition about the robot's own body (joint_frac, part_z, ...) rather than the object?"""
    from .goal_terms import BODY_FUNCS, uses
    return c.kind == "expression" and bool(c.expr) and bool(uses(c.expr) & BODY_FUNCS)


def unsatisfiable(goal: GoalSpec, scene: UserScene) -> list[str]:
    """Can the end of this checklist ever be met? Tries resting states of the object over the whole scene
    (on the floor, on fixtures' tops and floors; upright, on a side, upside down; turned four ways; and held,
    if the goal lets the robot hold on). Returns the items no state satisfies (alone or together), in words."""
    from .goal_bench import Moment, _end_ok, _v
    from .goal_terms import fixtures_geometry
    ends = [c for c in goal.conditions if c.when == "end" and not about_body(c)]  # the robot's own body: can't
    if not ends:                                                                     # be judged from the scene
        return []
    fx = fixtures_geometry(scene)
    b = scene.block
    half = (b.half_x, b.half_y, b.half_z)
    xs = [scene.start[0]] + [v for f in fx.values() for v in (f["lo"][0], f["hi"][0])]
    ys = [scene.start[1]] + [v for f in fx.values() for v in (f["lo"][1], f["hi"][1])]
    hold = not lets_go(goal)
    states = []
    # a 2 cm grid over everything that matters (around the fixtures and the start), a coarse one further out
    grid = sorted({round(float(v), 3) for v in np.arange(min(xs) - 0.6, max(xs) + 0.6, 0.06)}
                  | {round(float(v), 3) for f in fx.values() for v in np.arange(f["lo"][0] - 0.12, f["hi"][0] + 0.12, 0.02)})
    gridy = sorted({round(float(v), 3) for v in np.arange(min(ys) - 0.6, max(ys) + 0.6, 0.06)}
                   | {round(float(v), 3) for f in fx.values() for v in np.arange(f["lo"][1] - 0.12, f["hi"][1] + 0.12, 0.02)})
    for gx in grid:
        for gy in gridy:
            floors = [0.0] + [f["rest_z"] if f["kind"] == "container" else f["top_z"] for f in fx.values()
                              if f["lo"][0] <= gx <= f["hi"][0] and f["lo"][1] <= gy <= f["hi"][1]]
            for fl in floors:
                states.append((gx, gy, fl, False))
            if hold:
                states.append((gx, gy, 0.25, True))
    poses = [(t, y) for t in (0, 90, 180) for y in (0, 90, 180, 270)]
    ok_any = [False] * len(ends)
    for gx, gy, fl, held in states:
        for tilt, yaw in poses:
            v = _v(scene, fx, Moment(gx, gy, fl, tilt=tilt, yaw=yaw, held=held, hand_dist=0.03 if held else 0.4))
            res = []
            for c in ends:
                try:
                    res.append(_end_ok(c, v, scene, fx, half))
                except KeyError:
                    res.append(True)  # about the robot: can't judge it here
            for i, r in enumerate(res):
                ok_any[i] |= r
            if all(res):
                return []
    alone = [ends[i].describe() for i, ok in enumerate(ok_any) if not ok]
    if alone:
        return [f"'{d}' can never be true for the {scene.object_name} resting anywhere in this scene (check the "
                f"quantity: z is its centre, {b.half_z * 100:.0f} cm up when it rests on the floor; bottom is its "
                f"lowest point)" for d in alone]
    return ["each item can be met on its own, but no single resting state meets them all together: "
            + "; ".join(c.describe() for c in ends)]
