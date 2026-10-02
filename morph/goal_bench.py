"""Does Morph understand what was asked? A benchmark of varied requests in the parcel scene.

Each request comes with outcomes a person would judge by hand: some that do what was asked (must pass)
and some that look close but don't (must fail), each a short sequence of moments (where the parcel is,
how it's turned, whether the robot is holding it, how fast it moves). A goal reading is right only if it
passes every "must pass" outcome and fails every "must fail" one. That tests what the goal means, not
how it's worded, for Nemotron and for Morph's built-in reader alike.

    morph goal-bench [--no-llm] [--only NAME]
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .goal_terms import History, condition_violation, fixtures_geometry, quantities
from .goals import SIDES, GoalSpec, lets_go

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ outcomes
def rot(tilt_deg: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    """Turned yaw_deg about up, then tipped tilt_deg about its own x axis."""
    t, y = math.radians(tilt_deg), math.radians(yaw_deg)
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
    return Rz @ Rx


@dataclass
class Moment:
    x: float
    y: float
    bottom: float = 0.0
    tilt: float = 0.0
    yaw: float = 0.0
    held: bool = False
    vel: tuple = (0.0, 0.0, 0.0)
    hand_dist: float = 0.4


@dataclass
class Outcome:
    what: str
    moments: list  # the last one is how it ends
    should_pass: bool


@dataclass
class Case:
    name: str
    request: str
    outcomes: list = field(default_factory=list)
    impossible: bool = False  # the right answer is "can't": it mentions something that isn't there


def _v(scene, fx, m: Moment) -> dict:
    b = scene.block
    half = (b.half_x, b.half_y, b.half_z)
    R = rot(m.tilt, m.yaw)
    z = m.bottom + abs(R[2, 0]) * half[0] + abs(R[2, 1]) * half[1] + abs(R[2, 2]) * half[2]
    return quantities(np, pos=np.array([m.x, m.y, z]), R=R, vel=np.array(m.vel, float), start=scene.start[:2], half=half,
                      fixtures=fx, held=1.0 if m.held else 0.0, robot_xy=np.array([0.0, -0.18]), hand_dist=m.hand_dist,
                      hand_pos=np.array([m.x, m.y, z + 0.05]), base_z=0.0, base_up=1.0)


def _end_ok(c, v, scene, fx, half) -> bool:
    """One end-of-task condition on a moment (the same tests check() makes in the simulation)."""
    from .goal_terms import is_generic
    cd = c.model_dump()
    if is_generic(cd):
        return float(condition_violation(np, cd, v, fx, half)) <= 1e-6
    k = c.kind
    x, y, bottom = float(v["x"]), float(v["y"]), float(v["bottom"])
    f = scene.fixtures.get(c.target or "")
    if k in ("inside", "on_top_of"):
        lo, hi = (f.inner_lo, f.inner_hi) if k == "inside" else (f.top_lo, f.top_hi)
        surface = f.rest_z if k == "inside" else f.top_z
        return lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1] and abs(bottom - surface) < 0.012
    if k == "touching":
        return float(v["dist"][c.target]) <= 0.005
    if k in ("near", "beside"):
        dist = float(v["dist"][c.target])
        if k == "near":
            return dist <= (c.tol or 0.1) and not (f.lo[0] <= x <= f.hi[0] and f.lo[1] <= y <= f.hi[1])
        sx, sy = SIDES[c.side]
        on = (x > f.hi[0] if sx > 0 else x < f.lo[0] if sx < 0 else True) and (y > f.hi[1] if sy > 0 else y < f.lo[1] if sy < 0 else True)
        lateral = (f.lo[1] - 0.05 <= y <= f.hi[1] + 0.05) if sx else (f.lo[0] - 0.05 <= x <= f.hi[0] + 0.05)
        return on and lateral and dist <= 0.2
    if k == "at_position":
        return math.hypot(x - c.x, y - c.y) <= (c.tol or 0.05)
    if k == "heading":
        e = abs((float(v["yaw_deg"]) - c.yaw_deg + 180) % 360 - 180)
        return e <= (c.tol or 15)
    if k == "upright":
        return float(v["tilt_deg"]) <= (c.tol or 10)
    raise ValueError(k)


def judge(goal: GoalSpec, scene, outcome: Outcome) -> bool:
    """Would Morph count this outcome as done?"""
    fx = fixtures_geometry(scene)
    b = scene.block
    half = (b.half_x, b.half_y, b.half_z)
    hist = History([c.model_dump() for c in goal.conditions], fx, half)
    vs = [_v(scene, fx, m) for m in outcome.moments]
    for v in vs:
        hist.update(v)
    res = hist.result()
    last = outcome.moments[-1]
    for i, c in enumerate(goal.conditions):
        if c.when != "end":
            if res.get(i, math.inf) > 1e-6:
                return False
        elif not _end_ok(c, vs[-1], scene, fx, half):
            return False
    if last.held and lets_go(goal):
        return False
    if not last.held and math.hypot(*last.vel) > 0.02:
        return False
    return True


# ------------------------------------------------------------------ the cases (parcel scene)
def cases() -> list[Case]:
    S = (0.45, -0.18)  # where the parcel starts
    BIN, SHELF = (0.42, 0.18), (0.62, -0.02)  # centres; bin floor 1 cm, shelf top 5 cm
    WALL_X = 0.795  # the wall's west face, spanning y -0.30 .. 0.10
    rest = lambda x, y, b=0.0, **k: Moment(x, y, b, **k)
    in_bin, on_shelf, on_floor = rest(*BIN, 0.01), rest(*SHELF, 0.05), rest(*S)
    carry = lambda x, y, b=0.15, **k: Moment(x, y, b, held=True, **k)
    O = Outcome
    return [
        Case("put in bin", "put the parcel in the bin",
             [O("resting in the bin", [carry(*BIN), in_bin], True), O("on the floor next to the bin", [rest(0.42, 0.0)], False)]),
        Case("on shelf", "place the parcel on the shelf",
             [O("on the shelf", [carry(*SHELF), on_shelf], True), O("on the floor by the shelf", [rest(0.62, -0.13)], False)]),
        Case("against wall", "push the parcel against the wall",
             [O("pressed against the wall", [rest(WALL_X - 0.027, -0.18)], True), O("10 cm short of it", [rest(0.67, -0.18)], False)]),
        Case("30 cm north", "move the parcel 30 cm north",
             [O("31 cm north of the start", [rest(0.45, 0.13)], True), O("31 cm east instead", [rest(0.76, -0.18)], False)]),
        Case("knock over", "knock the parcel over",
             [O("lying on its side", [rest(*S, tilt=90)], True), O("still standing", [on_floor], False)]),
        Case("upside down", "flip the parcel upside down",
             [O("upside down", [rest(*S, tilt=180)], True), O("on its side", [rest(*S, tilt=90)], False)]),
        Case("lift and hold", "lift the parcel 20 cm off the floor and hold it there",
             [O("held 22 cm up", [carry(0.45, -0.18, 0.22)], True), O("dropped back on the floor", [carry(0.45, -0.18, 0.22), on_floor], False)]),
        Case("keep away", "keep the parcel at least 40 cm away from the wall",
             [O("47 cm from the wall", [rest(0.30, -0.18)], True), O("17 cm from the wall", [rest(0.60, -0.18)], False)]),
        Case("throw 1 m north", "throw the parcel into the bin from 1 m north of the bin",
             [O("carried 1.1 m north, thrown, lands in the bin", [carry(0.45, 0.6), carry(0.44, 1.36), rest(0.44, 1.30, 0.2, vel=(0, -3.0, 0)), in_bin], True),
              O("carried over the bin and put in", [carry(*BIN, 0.1), in_bin], False),
              O("thrown from 1 m north but misses", [carry(0.44, 1.36), rest(0.44, 1.30, 0.2, vel=(0, -3.0, 0)), rest(0.42, 0.0)], False),
              O("thrown from only 40 cm north", [carry(0.42, 0.66), rest(0.42, 0.62, 0.2, vel=(0, -2.0, 0)), in_bin], False)]),
        Case("throw", "throw the parcel into the bin",
             [O("thrown from 40 cm away, lands in the bin", [carry(0.42, -0.25, 0.2), rest(0.42, -0.20, 0.2, vel=(0, 2.0, 0.5)), in_bin], True),
              O("carried over the bin and set down", [carry(*BIN, 0.05), in_bin], False),
              O("dropped from just above the bin", [carry(*BIN, 0.2), rest(*BIN, 0.19, vel=(0, 0, -0.3)), in_bin], False)]),
        Case("drop from 30 cm", "drop the parcel into the bin from at least 30 cm above it",
             [O("let go 35 cm above the bin", [carry(*BIN, 0.40), rest(*BIN, 0.39, vel=(0, 0, -0.5)), in_bin], True),
              O("set down gently in the bin", [carry(*BIN, 0.03), in_bin], False)]),
        Case("never tip", "put the parcel in the bin without ever tipping it more than 20 degrees",
             [O("level all the way", [carry(0.45, 0.0), carry(*BIN), in_bin], True),
              O("tipped 45° on the way", [carry(0.45, 0.0, tilt=45), carry(*BIN), in_bin], False)]),
        Case("lift high on the way", "put the parcel in the bin, but at some point lift it higher than 40 cm",
             [O("raised to 45 cm, then into the bin", [carry(0.45, 0.0, 0.45), carry(*BIN), in_bin], True),
              O("never above 20 cm", [carry(0.45, 0.0, 0.2), in_bin], False)]),
        Case("slide no lift", "slide the parcel along the floor to the wall without lifting it",
             [O("slid along the floor to the wall", [rest(0.6, -0.18, vel=(0.2, 0, 0)), rest(WALL_X - 0.027, -0.18)], True),
              O("lifted 10 cm on the way", [carry(0.6, -0.18, 0.1), rest(WALL_X - 0.027, -0.18)], False)]),
        Case("south of shelf", "put the parcel on the floor next to the shelf, on its south side",
             [O("just south of the shelf", [rest(0.62, -0.14)], True), O("north of the shelf", [rest(0.62, 0.10)], False)]),
        Case("arrow west", "turn the parcel so its arrow points west",
             [O("arrow west", [rest(*S, yaw=180)], True), O("arrow north", [rest(*S, yaw=90)], False)]),
        Case("bin facing north", "put the parcel in the bin with its arrow facing north",
             [O("in the bin, arrow north", [carry(*BIN), rest(*BIN, 0.01, yaw=90)], True), O("in the bin, arrow east", [in_bin], False)]),
        Case("bring to hand", "bring the parcel to the robot's hand and hold it",
             [O("held at the hand", [Moment(0.3, -0.18, 0.2, held=True, hand_dist=0.03)], True),
              O("left on the floor", [Moment(*S, hand_dist=0.5)], False)]),
        Case("between shelf and wall", "put the parcel on the floor between the shelf and the wall",
             [O("in the gap", [rest(0.74, -0.02)], True), O("west of the shelf", [rest(0.50, -0.02)], False)]),
        Case("balance on wall", "balance the parcel on top of the wall",
             [O("on the wall's top", [rest(0.80, -0.10, 0.08)], True), O("against the wall, on the floor", [rest(WALL_X - 0.027, -0.10)], False)]),
        Case("spin in place", "spin the parcel 180 degrees without moving it",
             [O("turned around where it was", [rest(0.45, -0.17, yaw=180)], True),
              O("turned but moved 40 cm", [rest(0.45, 0.22, yaw=180)], False), O("not turned", [on_floor], False)]),
        Case("tap then shelf", "tap the bin with the parcel, then leave it on the shelf",
             [O("touched the bin, ended on the shelf", [Moment(0.42, 0.075, 0.03, held=True), carry(*SHELF), on_shelf], True),
              O("straight onto the shelf", [carry(*SHELF), on_shelf], False)]),
        Case("shelf then bin", "first put the parcel on the shelf, then move it into the bin",
             [O("on the shelf, then in the bin", [on_shelf, carry(*BIN), in_bin], True), O("straight into the bin", [carry(*BIN), in_bin], False)]),
        Case("gently", "put the parcel in the bin gently, don't drop it",
             [O("lowered and let go just above the floor", [carry(*BIN, 0.02), in_bin], True),
              O("dropped from 30 cm", [carry(*BIN, 0.3), rest(*BIN, 0.29, vel=(0, 0, -0.5)), in_bin], False)]),
        Case("side by wall", "stand the parcel on its side next to the wall",
             [O("on its side by the wall", [rest(0.74, -0.18, tilt=90)], True), O("upright by the wall", [rest(0.74, -0.18)], False)]),
        Case("out of reach", "push the parcel at least 50 cm away from where it started",
             [O("55 cm away", [rest(0.45, 0.37)], True), O("30 cm away", [rest(0.75, -0.18)], False)]),
        Case("red box", "put the parcel on the red box", impossible=True),
    ]


# ------------------------------------------------------------------ running it
def run(use_llm: bool = True, only: str | None = None, log=print) -> dict:
    from .project import GoalNotPossible, load_project, resolve_goal
    from .reasoning import ReasoningService
    pj = load_project(ROOT / "examples" / "parcel")
    llm = ReasoningService() if use_llm else None
    rows = []
    for case in cases():
        if only and only.lower() not in case.name.lower():
            continue
        row = {"case": case.name, "request": case.request}
        try:
            goal, info = resolve_goal(pj, case.request, None, llm, log=lambda s: None)
            row["goal"] = goal.describe()
            row["from"] = info["goal_from"]
            if case.impossible:
                row["ok"], row["why"] = False, "should have said it's impossible"
            else:
                wrong = [o.what + (" (should pass)" if o.should_pass else " (should fail)")
                         for o in case.outcomes if judge(goal, pj.scene, o) != o.should_pass]
                row["ok"], row["why"] = not wrong, "; ".join(wrong)
        except GoalNotPossible as e:
            row["ok"], row["why"], row["goal"] = case.impossible, "" if case.impossible else f"refused: {e}", []
        except ValueError as e:
            row["ok"], row["why"], row["goal"] = case.impossible, "" if case.impossible else f"error: {e}", []
        rows.append(row)
        log(f"{'PASS' if row['ok'] else 'FAIL'}  {case.name:22s} {' | '.join(row['goal'])[:150]}"
            + (f"\n        wrong on: {row['why']}" if row["why"] else ""))
    n = sum(r["ok"] for r in rows)
    out = {"reader": "nemotron" if use_llm else "rules", "correct": n, "total": len(rows), "rows": rows}
    if llm is not None:
        from .app.compute import nemotron_cost
        out["cost"] = nemotron_cost(llm.records)
    log(f"\n{out['reader']}: {n}/{len(rows)} understood correctly")
    return out
