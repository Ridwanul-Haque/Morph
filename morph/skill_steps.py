"""How to go about a task, step by step, for this particular body.

A person standing up from a chair doesn't think "stand up" and hope: they put their feet under them, lean
forward over them, push, and steady themselves. Robots learning from scratch don't know any of that, so
they wander for millions of tries before stumbling on it. Here Nemotron (or, without it, Morph's own
rules) writes those unconscious steps for this robot and this goal: what to do with which body part, in
what order, and how to tell a step is done, as a formula over measured quantities (goal_terms). The
learner then gets a reward for each step it reaches, in order, instead of only for the end result, and
the practice runs show how far along it is.

The last step is always the goal itself (every checklist item). Steps can also say what to keep true while
doing them ("keep it upright while carrying it", "keep the body level").
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, model_validator

from .goal_terms import HOLD_TICKS, ExprError, parse  # noqa: F401  (HOLD_TICKS: also used by the trainer)

GOAL = "goal"  # done_when of the last step: every goal condition holds


class Step(BaseModel):
    name: str = Field(max_length=80)  # "Open the hand and move it over the parcel"
    how: str = Field(default="", max_length=400)  # what this body does, in plain words
    why: str = Field(default="", max_length=240)  # the unconscious reason a person would do it
    body_parts: list[str] = Field(default_factory=list, max_length=6)
    done_when: str = Field(max_length=300)  # a formula (goal_terms), or "goal" for the last step
    keep: str = Field(default="", max_length=300)  # a formula that should stay true during this step

    @model_validator(mode="before")
    @classmethod
    def _lenient(cls, d):
        """The same step written loosely (a model's reply): keep false / null for "nothing to keep", a word where a
        list is expected, text a little over the length limits. None of that should throw a whole plan away."""
        if not isinstance(d, dict):
            return d
        d = dict(d)
        for k, n in (("name", 80), ("how", 400), ("why", 240), ("done_when", 300), ("keep", 300)):
            v = d.get(k)
            if v is None or isinstance(v, bool):
                d[k] = "" if k != "name" else "step"
            elif not isinstance(v, str):
                d[k] = str(v)
            d[k] = d[k][:n]
        bp = d.get("body_parts")
        if isinstance(bp, str):
            d["body_parts"] = [bp]
        elif bp is None:
            d["body_parts"] = []
        if isinstance(d.get("body_parts"), list):
            d["body_parts"] = [str(x)[:60] for x in d["body_parts"]][:6]
        return d


def robot_brief(prof) -> dict:
    """What the body has to work with, for writing steps a person would take in its place."""
    grip = prof.gripper or {}
    parts = []
    if prof.n_arms:
        parts.append(f"{prof.n_arms} arm(s)")
    if prof.has_gripper:
        parts.append(f"a {grip.get('type', 'gripper').replace('_', ' ')} opening {grip.get('max_opening_m', 0) * 100:.0f} cm")
    elif prof.n_arms:
        parts.append("no gripper: the arm ends in a tool flange it can push with")
    if prof.n_legs:
        parts.append(f"{prof.n_legs} legs (feet can push)")
    if prof.base in ("planar_mobile", "wheeled"):
        parts.append("a driving base")
    reach = max((v["max_horizontal"] for v in prof.reach_m.values()), default=0.0)
    return {"name": prof.name, "kind": prof.kind.replace("_", " "), "base": prof.base, "body_parts": parts,
            "degrees_of_freedom": prof.dof, "height_m": round(prof.size_m.get("height", 0), 2),
            "arm_reach_m": round(reach, 2), "mass_kg": round(prof.mass_kg, 1),
            "stands_on_the_floor": prof.base == "floating" and prof.n_legs > 0}


MAX_TERMS = 4


def check_terms(raw, fixtures, object_name: str = "") -> tuple[list[dict], list[str]]:
    """Reward terms Nemotron proposes (Eureka-style shaping on top of the steps): each {name, formula, weight}. formula is
    a number formula over the measured quantities (or a condition: then minus how far from true it is); weight in
    -5..5 (> 0 rewards it, < 0 penalises). Returns the terms in Morph's form and the problems with any it can't use."""
    from .goal_terms import normalize
    out, errors = [], []
    for t in (raw or [])[:MAX_TERMS]:
        if not isinstance(t, dict) or not t.get("formula"):
            errors.append(f"a reward term needs a formula: {t!r}")
            continue
        try:
            f = normalize(str(t["formula"]), object_name)
            parse(f, fixtures, object_name)
            w = float(t.get("weight", 1.0))
        except (ExprError, ValueError, TypeError) as e:
            errors.append(f"reward term {t.get('name', t['formula'])!r}: {e}")
            continue
        out.append({"name": str(t.get("name") or f)[:40], "formula": f, "weight": max(-5.0, min(5.0, w)),
                    "why": str(t.get("why", ""))[:200]})
    return out, errors


def check_steps(steps: list[Step], fixtures, object_name: str = "") -> tuple[list[Step], list[str]]:
    """The steps with their formulas in Morph's form, and the problems with any it can't measure."""
    from .goal_terms import normalize
    ok, errors = [], []
    for s in steps:
        if s.done_when.strip().lower() == GOAL:
            continue  # added at the end
        try:
            parse(s.done_when, fixtures, object_name)
            if s.keep:
                parse(s.keep, fixtures, object_name)
        except ExprError as e:
            errors.append(f"step '{s.name}': {e}")
            continue
        ok.append(s.model_copy(update={"done_when": normalize(s.done_when, object_name).strip(),
                                       "keep": normalize(s.keep, object_name).strip() if s.keep else ""}))
    return ok[:6], errors


def scales(scene, p: dict) -> dict:
    """Realistic numbers for thresholds (what hand_dist reads when touching or gripping, and so on)."""
    b = scene.block
    r = math.hypot(b.half_x, b.half_y)
    return {"object_half_size_m": [round(b.half_x, 3), round(b.half_y, 3), round(b.half_z, 3)],
            "hand_dist_when_touching_its_side_m": round(min(b.half_x, b.half_y) + 0.01, 3),
            "hand_dist_when_gripping_m": f"{min(b.half_x, b.half_y):.3f} to {r + 0.03:.3f} (fingertips sit either side, not at the centre)",
            "object_z_when_resting_on_floor_m": round(b.half_z, 3), "bottom_when_on_floor_m": 0.0,
            "target": {k: round(float(v), 3) for k, v in p.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
            "grip_scale": "grip is 0..1: about 0.25 with the open hand around it, 0.6 closed on it, 1 held and lifted; closure 0..1 = how far the gripper is commanded shut",
            "note": "thresholds must be reachable: allow a few cm of slack; 'held' needs the object off the ground"}


def _body_facts(prof) -> dict:
    """The robot's joints (with ranges) and end parts, for planning steps of a goal about its own body."""
    from .project import robot_bodies
    b = robot_bodies([prof.source])
    return b[0] if b else {}


def final_step(goal, robot_name: str) -> Step:
    items = "; ".join(c.text or c.describe() if c.kind == "expression" else c.describe() for c in goal.conditions)
    return Step(name="Finish: everything on the goal checklist",
                how=(f"{robot_name} settles it so that: {items}")[:400], why="the goal itself", done_when=GOAL)


# ------------------------------------------------------------------ Morph's own steps (no Nemotron)
def _minus(v: float) -> str:
    """ "- 0.180" or "+ 0.180": subtracting a coordinate, written the way a person would."""
    return f"- {v:.3f}" if v >= 0 else f"+ {-v:.3f}"


def rule_steps(prof, goal, scene, p: dict) -> list[Step]:
    """Common-sense steps by body type, with numbers from the scene. Nemotron's are usually richer."""
    b = scene.block
    obj = scene.object_name
    r = math.hypot(b.half_x, b.half_y)
    sx, sy = scene.start[0], scene.start[1]
    tx, ty = p["x"], p["y"]
    kinds = {c.kind for c in goal.conditions}
    upright = "upright" in kinds
    keep_up = "tilt_deg < 25" if upright else ""
    grasp = prof.has_gripper and prof.n_arms > 0
    legged = prof.n_legs >= 2 and prof.base == "floating"
    lift = p.get("needs_lift") or "height" in kinds
    out: list[Step] = []
    from .goals import release_spot
    spot = release_spot(goal, scene)
    if grasp and spot is not None:  # throwing / dropping from somewhere: grip, carry there, let go on the way
        rx, ry = spot
        aim = "towards the target" if p.get("has_place") else "where it should land"
        return [Step(name=f"Grip the {obj} firmly", how=f"come down over it with the hand open, close on two opposite sides",
                     why="a throw needs a hold that won't slip mid-swing", body_parts=["arm", "gripper"],
                     done_when="held"),
                Step(name="Lift it clear", how="raise it off the floor before moving anywhere",
                     why="dragging it would knock things", body_parts=["arm"], done_when="held and bottom > 0.05",
                     keep="held"),
                Step(name="Carry it to where it's let go", how=f"bring it to about ({rx:.2f}, {ry:.2f}), held up high",
                     why="the throw has to start from there", body_parts=["arm"] + (["legs"] if legged else []),
                     done_when=f"held and hypot(x {_minus(rx)}, y {_minus(ry)}) < 0.1", keep="held"),
                Step(name=f"Swing {aim} and open the hand mid-swing",
                     how="move the hand quickly along the throw and let go while it's still moving, so it flies on",
                     why="it only flies if it's moving when you let go", body_parts=["arm", "gripper"],
                     done_when="not held and speed > 0.3")]
    if "orientation" in kinds and not lift:  # knock it over / lay it on its side
        out = [Step(name=f"Bring a {'foot' if legged else 'hand'} to the top half of the {obj}",
                    how="touch it high up, above its middle, where a push makes it tip instead of slide",
                    why="pushing low slides it; pushing high tips it", body_parts=["foot" if legged else "hand"],
                    done_when=f"hand_dist < {r + 0.05:.3f} and hand_z > z"),
               Step(name="Push the top over", how="keep pushing sideways at the top until it falls",
                    why="once past its tipping point it falls on its own", body_parts=["foot" if legged else "hand"],
                    done_when="tilt_deg > 35")]
    elif grasp and lift:
        clear = max(p.get("clear_z", 0.0), 0.0)
        out = [Step(name=f"Open the hand and move it above the {obj}",
                    how=f"bring the open gripper over the {obj}, a few cm above its top, fingers pointing down",
                    why="coming down from above avoids knocking it", body_parts=["arm", "gripper"],
                    done_when=f"hypot(hand_x - x, hand_y - y) < 0.04 and hand_z > top + 0.02"),
               Step(name=f"Lower the hand around the {obj}", how="descend slowly until the fingers are either side of it",
                    why="fingers must surround it before closing", body_parts=["arm", "gripper"],
                    done_when=f"hand_dist < {r + 0.02:.3f}", keep=keep_up),
               Step(name="Close the grip and lift it clear", how="squeeze just hard enough, then raise it off the floor",
                    why="a firm grip first, or it slips when lifted", body_parts=["gripper", "arm"],
                    done_when="held and bottom > 0.03", keep=keep_up)]
        if "height" not in kinds:
            out.append(Step(name=f"Carry it over the target, high enough to clear {clear * 100:.0f} cm",
                            how="move across with it held level, above whatever it passes over",
                            why="lifting above the walls before moving avoids hitting them", body_parts=["arm"],
                            done_when=f"hypot(x {_minus(tx)}, y {_minus(ty)}) < 0.04 and bottom > {clear + 0.01:.3f}",
                            keep="held" + (" and tilt_deg < 25" if upright else "")))
    elif p.get("has_place") and (abs(tx - sx) + abs(ty - sy)) > 0.03:  # push it somewhere
        d = math.hypot(tx - sx, ty - sy)
        ux, uy = (tx - sx) / d, (ty - sy) / d
        k = r + 0.04
        part = "foot" if legged else "hand"
        if legged:
            out.append(Step(name=f"Walk to stand behind the {obj}", how=f"walk round so the {obj} is between you and the goal",
                            why="you push things away from you, not sideways", body_parts=["legs"],
                            done_when=f"hypot(robot_x - (x {_minus(ux * 0.4)}), robot_y - (y {_minus(uy * 0.4)})) < 0.15",
                            keep="base_tilt_deg < 25"))
        out += [Step(name=f"Put the {part} behind the {obj}, low down",
                     how=f"place it on the side away from the goal, near the bottom so it slides instead of tipping",
                     why="pushing low and from behind moves it straight", body_parts=[part],
                     done_when=f"hypot(hand_x - (x {_minus(ux * k)}), hand_y - (y {_minus(uy * k)})) < 0.05 "
                               f"and hand_z < z + 0.02", keep="base_tilt_deg < 25" if legged else ""),
                Step(name="Push it steadily most of the way", how="keep contact and push towards the goal, not too fast",
                     why="a steady push keeps it from spinning away", body_parts=[part],
                     done_when=f"hypot(x {_minus(tx)}, y {_minus(ty)}) < {max(0.03, d / 2):.3f}",
                     keep="speed < 0.5" + (" and tilt_deg < 25" if upright else ""))]
    else:
        out = [Step(name=f"Reach the {obj}", how=f"bring the nearest {'foot' if legged else 'hand'} to it",
                    why="you have to touch it to move it", body_parts=["foot" if legged else "hand"],
                    done_when=f"hand_dist < {r + 0.05:.3f}")]
    return out


OBJECT_VARS = {"x", "y", "z", "bottom", "top", "tilt_deg", "yaw_deg", "speed", "vx", "vy", "vz", "travelled", "held", "grip",
               "touch_dist", "lifted", "hand_dist", "spin_dps", "spin_z_dps", "dist_to"}


def body_only(goal) -> bool:
    """Is the whole goal about the robot's own body (a fist, a raised arm), with nothing to do with the object?"""
    from .goals import about_body
    return bool(goal.conditions) and all(about_body(c) for c in goal.conditions)


def object_steps(steps: list[Step]) -> list[str]:
    """For a goal about the robot's own body: the steps that bring the object in anyway, in words."""
    from .goal_terms import uses
    out = []
    for st in steps:
        used = sorted((uses(st.done_when) | (uses(st.keep) if st.keep else set())) & OBJECT_VARS)
        if used:
            out.append(f"step '{st.name}' is about the object ({', '.join(used)}), but the goal is only about the "
                       "robot's own body: steps must use joint_frac / part_x / part_y / part_z only")
    return out


def plan_steps(prof, goal, scene, p: dict, llm=None, log=print) -> tuple[list[Step], dict]:
    """Nemotron's steps if it's available and they check out, else Morph's own. The goal is always last."""
    fixtures = list(scene.fixtures)
    info = {"from": "rules", "notes": []}
    steps = None
    obj = scene.object_name
    body = body_only(goal)
    if llm is not None and getattr(llm, "available", False):
        try:
            errors, fix = [], None
            for attempt in range(2):  # one chance to fix formulas Morph can't read
                prop, rec = llm.skill_steps(robot_brief(prof), [c.describe() for c in goal.conditions], scene.geometry(),
                                            scales(scene, p), fix, body=_body_facts(prof) if body else None)
                raw = [Step(**s) for s in prop.steps if isinstance(s, dict) and s.get("done_when")]
                cand, errors = check_steps(raw, fixtures, obj)
                if body:
                    errors += object_steps(cand)
                if cand and not errors:
                    steps, info["from"] = cand, f"nemotron ({rec.latency_s}s)"
                    info["rationale"] = prop.rationale
                    break
                fix = {"your_previous_steps": [s.model_dump() for s in raw], "problems": errors}
            if steps is None:
                info["notes"].append("Nemotron's steps had formulas Morph can't measure (" + "; ".join(errors[:2])
                                     + "); using Morph's own")
        except Exception as e:  # noqa: BLE001  (no key, network, bad reply: Morph's own steps)
            info["notes"].append(f"Nemotron couldn't plan the steps ({e}); using Morph's own")
    if steps is None:
        # Morph's own: for a goal about the robot's body, no object steps (the goal itself is the one step)
        steps, errors = check_steps([] if body else rule_steps(prof, goal, scene, p), fixtures, obj)
        info["notes"] += errors
    for n in info["notes"]:
        log(f"  ({n})")
    return steps + [final_step(goal, prof.name)], info




class StepTracker:
    """Follows a run through the steps (same rule as the learner: one at a time, in order, each held for
    HOLD_TICKS; a step that's too strict can be skipped when the next one holds that long)."""

    def __init__(self, steps: list[dict], fixtures: dict, on_step=None):
        import numpy as np
        self.np = np
        self.steps = [(s["name"], parse(s["done_when"], fixtures)) for s in steps]
        self.reached, self.on_step = 0, on_step
        self.streak = [0, 0]  # ticks the current step / the one after it have held in a row

    def update(self, v: dict) -> None:
        from .goal_terms import violation
        k, n = self.reached, len(self.steps)
        done = lambda i: i < n and float(violation(self.np, self.steps[i][1], v)) <= 1e-3
        d0, d1 = done(k), (not done(k)) and done(k + 1)
        self.streak = [self.streak[0] + 1 if d0 else 0, self.streak[1] + 1 if d1 else 0]
        if self.streak[0] >= HOLD_TICKS:
            self.reached = k + 1
        elif self.streak[1] >= HOLD_TICKS:
            self.reached = k + 2
        if self.reached != k:
            self.streak = [0, 0]
            if self.on_step:
                self.on_step(self.reached, self.steps[self.reached - 1][0])

    def summary(self) -> str:
        if not self.steps:
            return ""
        nxt = f"; stuck at: {self.steps[self.reached][0]}" if self.reached < len(self.steps) else ""
        return f"reached {self.reached} of {len(self.steps)} steps{nxt}"


def steps_with_words(steps: list[dict], object_name: str = "the object") -> list[dict]:
    """The steps as the app shows them: each with its done_when / keep in plain words (so a person can see when a step
    was understood wrongly, and say so)."""
    from .goal_terms import describe_formula
    out = []
    for s in steps:
        s = dict(s)
        if s.get("done_when"):
            s["done_words"] = describe_formula(s["done_when"], object_name)
        if s.get("keep"):
            s["keep_words"] = describe_formula(s["keep"], object_name)
        out.append(s)
    return out
