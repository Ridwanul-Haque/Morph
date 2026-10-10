"""Programs for legged robots: Nemotron writes them from building blocks on top of the robot's learned gait, Morph runs
them, says what happened, and Nemotron fixes them (the same loop as programs.py for arms).

Building blocks (data, never code; Morph checks every argument):
  walk_to(x, y, facing)                    walk there (around the object and the fixtures), facing a heading or the object
  turn_to(facing)                          turn on the spot
  foot_push(direction, distance, height)   stand behind the object and push it with a front foot (low slides it,
                                           high tips it over)
  bump(direction, distance)                walk into the object with the body (a nudge, or knocking it over)
  push_object_to(x, y, facing)             Morph plans a series of foot pushes that bring the object there
  wait(seconds)

Robots: the ones with a walking policy (planner.walk_policy_dir: the Go1, and any legged robot Morph has trained a
gait for, e.g. the Go2).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .goal_terms import History
from .goals import GoalSpec, check
from .programs import ProgramError, _formula, scene_names

LEG_OPS = {"walk_to", "turn_to", "foot_push", "bump", "push_object_to", "wait", "tip_over"}
MAX_STEPS = 16
CACHE = Path(__file__).resolve().parents[1] / "runs" / "programs"
LOW, HIGH = 0.025, 0.07  # foot push heights (m): low on the face slides it, high tips it

BLOCKS_DOC = (
    "walk_to(x, y, facing): the robot walks so its body centre is at (x, y), routing around the object and the "
    "fixtures; facing: a heading in degrees (0 = east, 90 = north) or \"object\" (turned towards the object). "
    "turn_to(facing). foot_push(direction, distance, height): Morph stands the robot behind the object and pushes it "
    "with a front foot towards direction (degrees), about distance (m; one push moves it at most about 7 cm: use "
    "several for more); height \"low\" slides it, \"high\" pushes higher on its face (a tall, narrow object may tip). bump(direction, distance): the robot walks its body into the "
    "object, moving it about distance (m) towards direction: rougher than a foot push (it may knock the object over). "
    "push_object_to(x, y, facing): Morph plans and runs a series of foot pushes that bring the object to (x, y) "
    "(facing: the object's final heading in degrees, or leave it out). tip_over(direction): Morph tries the ways this "
    "robot can knock it over towards direction (high and low foot pushes, body bumps) in simulation and uses the one "
    "that tipped it furthest. wait(seconds). Every number can instead be a "
    "formula (a string) over obj_x, obj_y, obj_z, obj_top, obj_yaw_deg, robot_x, robot_y, robot_facing_deg, and the "
    "scene's names (each fixture's <name>_x, <name>_y, <name>_top, <name>_floor), with + - * / abs min max hypot sqrt."
)


def make_robot(prof, scene):
    """The legged robot with its learned gait in this scene, or None if it has none."""
    from .planner import walk_policy_dir
    pol = walk_policy_dir(prof)
    if pol is None:
        return None
    if pol.name == "go1_walk":
        from .robots.go1_learned import LearnedGo1
        return LearnedGo1(block=scene.block)
    from .robots.learned_walker import LearnedWalker
    return LearnedWalker(pol, block=scene.block)


class LegPlanner:
    """tip_over for a legged robot: the ways it can knock the object over (a high or low foot push, one or two, a body
    bump), each tried in simulation on its own gait from where it stands; the one that tips it furthest is used."""

    def __init__(self, rb, scene, base0, names: dict, log=lambda s: None):
        self.rb, self.scene, self.base0, self.names, self.log, self.memo = rb, scene, base0, names, log, {}

    def candidates(self, d: float) -> list[tuple[str, list[dict]]]:
        fp = lambda h, n=1: [{"do": "foot_push", "direction": d, "distance": 0.07, "height": h}] * n  # noqa: E731
        legs = max(abs(v[1]) for v in self.rb.legs_adapter().legs.nominal.values()) if hasattr(self.rb, "legs_adapter") \
            else 0.13  # how far its feet stand out to the side: walk in that far off-centre and a leg strikes it
        bump = lambda off, dist: [{"do": "bump", "direction": d, "distance": dist, "offset": off}]  # noqa: E731
        out = [("a high foot push", fp("high")), ("two high foot pushes", fp("high", 2)),
               ("a low foot push, then a high one", fp("low") + fp("high")), ("walking into it centred", bump(0.0, 0.3))]
        for k in (0.7, 1.0, 1.3):  # off-centre by about a leg's sideways stance, either side
            out += [(f"walking into it {k * legs * 100:.0f} cm off-centre (left legs)", bump(k * legs, 0.3)),
                    (f"walking into it {k * legs * 100:.0f} cm off-centre (right legs)", bump(-k * legs, 0.3))]
        return out

    def __call__(self, d: float):
        key = round(float(d), 1)
        if key not in self.memo:
            from .goals import GoalSpec
            from .project import _params
            empty = GoalSpec(object=self.scene.object_name, conditions=[])
            from .project import _start_poses
            best = None
            for desc, prog in self.candidates(float(d)):
                steps = check_leg_program(prog, set(self.names))
                tipped, tilts = 0, []
                for i, st in enumerate(_start_poses(self.scene, 3)):  # how often, not once: a lucky try isn't a way
                    self.rb.reset(i, st, _params(i), base_xy_yaw=self.base0)
                    run_leg_program(self.rb, steps, empty, self.scene, self.names)
                    t = math.degrees(self.rb.env.block_pose().tilt) if not self.rb.fallen() else -1.0
                    tilts.append(t)
                    tipped += t > 45
                key_ = (tipped, sum(tilts))
                if best is None or key_ > best[0]:
                    best = (key_, desc, prog, tipped)
            (_, total), desc, prog, tipped = best
            self.log(f"  contact planning: tried {len(self.candidates(d))} ways to knock it over (3 tries each); best: "
                     f"{desc} (tipped it in {tipped}/3)")
            self.memo[key] = (prog, desc, 90.0 * tipped / 3)
        return self.memo[key]


def check_leg_program(steps: list, known: set = frozenset(), planner: "LegPlanner | None" = None) -> list[dict]:
    if not isinstance(steps, list) or not steps:
        raise ProgramError("the program has no steps")
    if len(steps) > MAX_STEPS:
        raise ProgramError(f"too many steps ({len(steps)}; at most {MAX_STEPS})")
    names = set(known) | {"robot_x", "robot_y", "robot_facing_deg"}
    from .programs import normalize_step
    out = []
    for i, st in enumerate(steps, 1):
        st = normalize_step(st)
        if not isinstance(st, dict):
            raise ProgramError(f"step {i} isn't an object")
        op = st.get("do") or st.get("op")
        if op not in LEG_OPS:
            raise ProgramError(f"step {i}: unknown building block {op!r} (known: {', '.join(sorted(LEG_OPS))})")
        if op == "tip_over":  # the way that tipped it furthest when tried (LegPlanner)
            try:
                d = float(st.get("direction"))
            except (TypeError, ValueError):
                raise ProgramError(f"step {i}: tip_over needs direction as a number of degrees") from None
            if planner is None:
                raise ProgramError(f"step {i}: tip_over needs Morph's contact planning (not available here)")
            prog, desc, tilt = planner(d)
            if tilt < 30:  # tipped it in under 1 of 3 tries
                raise ProgramError(f"step {i}: Morph tried {len(planner.candidates(d))} ways for this robot to knock it "
                                   f"over towards {d:.0f}° (high and low foot pushes, walking into it centred and "
                                   "off-centre), 3 tries each, and none tipped it reliably; try another direction or approach")
            out += check_leg_program(prog, known)
            continue
        s = {"do": op}
        if op in ("walk_to", "push_object_to"):
            for k in ("x", "y"):
                if k not in st:
                    raise ProgramError(f"step {i}: {op} needs {k}")
                s[k] = _formula(st[k], f"step {i} {k}", names)
        if op in ("walk_to", "turn_to", "push_object_to"):
            f = st.get("facing", "object" if op != "push_object_to" else None)
            if op == "turn_to" and f is None:
                raise ProgramError(f"step {i}: turn_to needs facing")
            s["facing"] = f if f in (None, "object") else _formula(f, f"step {i} facing", names)
        if op in ("foot_push", "bump"):
            s["direction"] = _formula(st.get("direction", 0.0), f"step {i} direction", names)
            s["distance"] = _formula(st.get("distance", 0.08), f"step {i} distance", names)
        if op == "bump":  # walk in this far to the left (+) or right (-) of the object: a leg strikes it, not the belly
            s["offset"] = _formula(st.get("offset", 0.0), f"step {i} offset", names)
        if op == "foot_push":
            s["height"] = st.get("height", "low")
            if s["height"] not in ("low", "high"):
                raise ProgramError(f"step {i}: height must be low or high")
        if op == "wait":
            s["seconds"] = min(max(float(st.get("seconds", 0.5)), 0.0), 5.0)
        out.append(s)
    return out


def describe(s: dict) -> str:
    if s["do"] == "walk_to":
        return f"walk to ({s['x']}, {s['y']}), facing {s['facing']}"
    if s["do"] == "turn_to":
        return f"turn to face {s['facing']}"
    if s["do"] == "foot_push":
        return f"push it {s['distance']} m towards {s['direction']}° with a front foot, {s['height']}"
    if s["do"] == "bump":
        return f"bump it with the body {s['distance']} m towards {s['direction']}°"
    if s["do"] == "push_object_to":
        return f"push it to ({s['x']}, {s['y']})" + (f", facing {s['facing']}°" if s["facing"] is not None else "")
    return f"wait {s['seconds']:.1f} s"


class _Values:
    def __init__(self, rb, half_z: float, extra: dict):
        p = rb.env.block_pose()
        b, th, _ = rb.base()
        self.v = dict(extra) | {"obj_x": p.x, "obj_y": p.y, "obj_z": p.z, "obj_top": p.z + half_z,
                                "obj_yaw_deg": math.degrees(p.yaw), "robot_x": b[0], "robot_y": b[1],
                                "robot_facing_deg": math.degrees(th)}

    def __call__(self, f) -> float:
        from .programs import _Values
        return _Values.__call__(self, f)


def _face_extent(scene, yaw: float, direction: float) -> float:
    """How far the object's face is from its centre along a direction (its half-size that way)."""
    b = scene.block
    a = direction - yaw
    return abs(b.half_x * math.cos(a)) + abs(b.half_y * math.sin(a))


def run_leg_program(rb, steps: list[dict], goal: GoalSpec, scene, names: dict | None = None, on_step=None,
                    settle_s: float = 1.0) -> dict:
    """Run a checked legged program. Returns success, the checklist results and, per step, what was measured."""
    from .go1_learned_task import Footprint, _route, run_learned_go1_push
    from .robots.go1_control import Go1Controller
    from .robots.quadruped import QuadrupedAdapter
    from .task_model import measure  # noqa: F401  (History uses the scene-level quantities below)
    from .goals import _measure, to_taskspec
    b = scene.block
    hist = History([c.model_dump() for c in goal.conditions], {k: v for k, v in _fixtures(scene).items()},
                   (b.half_x, b.half_y, b.half_z))
    obstacles = [(f.lo, f.hi) for f in scene.fixtures.values() if f.top_z > 0.02]
    generic = hasattr(rb, "legs_adapter")
    qa = rb.legs_adapter() if generic else QuadrupedAdapter(model=rb.model, env=rb.env)
    fp = Footprint(qa.legs, obstacles) if obstacles else None
    body_half = max(abs(v[0]) for v in qa.legs.nominal.values()) + 0.1

    def on_tick(r, *_):
        v = _measure(goal, scene, rb.env.model, rb.env.data, rb.env.block_pose(), True)[0]
        v["held"] = 0.0
        hist.update(v)

    rb.on_tick = on_tick
    report, names = [], names or {}

    def walk(x, y, facing) -> bool:
        p, _, _ = rb.base()
        ok = True
        bp = rb.env.block_pose()
        for wp in _route(rb, np.array([x, y]), np.array([bp.x, bp.y]), fp)[:-1]:
            ok &= rb.go_to(wp[0], wp[1], math.atan2(y - wp[1], x - wp[0]), tol=0.08, yaw_tol=0.6, timeout=10)
        return ok & rb.go_to(x, y, facing, tol=0.03, yaw_tol=0.06, timeout=15)

    for i, st in enumerate(steps, 1):
        val = _Values(rb, b.half_z, names)
        before = rb.env.block_pose()
        p0, th0, _ = rb.base()
        if on_step:
            on_step(i, describe(st))
        note = ""
        if st["do"] in ("walk_to", "turn_to"):
            x, y = (val(st["x"]), val(st["y"])) if st["do"] == "walk_to" else (p0[0], p0[1])
            f = st["facing"]
            yaw = math.atan2(before.y - y, before.x - x) if f == "object" else math.radians(val(f))
            ok = walk(x, y, yaw) if st["do"] == "walk_to" else rb.go_to(x, y, yaw, tol=0.05, yaw_tol=0.06, timeout=8)
            p, th, _ = rb.base()
            note = (f"got there (body at ({p[0]:.2f}, {p[1]:.2f}), facing {math.degrees(th):.0f}°)" if ok else
                    f"didn't quite get there: body at ({p[0]:.2f}, {p[1]:.2f}), facing {math.degrees(th):.0f}°")
        elif st["do"] == "foot_push":
            d = math.radians(val(st["direction"]))
            dist = float(np.clip(val(st["distance"]), 0.01, 0.15))
            h = LOW if st["height"] == "low" else min(HIGH, 2 * b.half_z - 0.008)
            # one push planned and run by the same loop as push_object_to (walk to a pose that keeps the robot clear of
            # fixtures, re-approach, re-step its feet if balancing on three would need a big shift, then push)
            spec = to_taskspec(goal, scene).model_copy(deep=True)
            spec.goal.x.value, spec.goal.y.value = before.x + dist * math.cos(d), before.y + dist * math.sin(d)
            spec.goal.yaw_tol_rad.value, spec.goal.pos_tol_m.value = math.pi, 0.01
            r = run_learned_go1_push(spec, None, None, 0, robot=rb, log=lambda s_: None, obstacles=obstacles,
                                     on_tick=lambda r_: on_tick(rb), reset=False, max_pushes=1, push_height=h)
            rb.on_tick = on_tick
            ev = r.pushes[0]["contact"] if r.pushes else {"skipped": r.stop_reason or "no push possible"}
            note = (f"no push: {ev['skipped']}" if ev.get("skipped") else
                    f"pushed with a front foot for {ev['foot_contact_ticks'] * 0.02:.1f} s (up to {ev['max_force_n']:.0f} N)")
        elif st["do"] == "bump":
            d = math.radians(val(st["direction"]))
            dist = float(np.clip(val(st["distance"]), 0.02, 0.6))
            u = np.array([math.cos(d), math.sin(d)])
            back = body_half + _face_extent(scene, before.yaw, d) + 0.08
            side = float(val(st.get("offset", 0.0)))
            sx, sy = np.array([before.x, before.y]) - u * back + side * np.array([-u[1], u[0]])
            walk(sx, sy, d)
            v = 0.25
            rb.walk((v, 0.0, 0.0), (dist + 0.08) / v)
            rb.walk((-0.2, 0.0, 0.0), 0.8)
            rb.walk((0.0, 0.0, 0.0), 0.4)
            note = "walked into it with the body"
        elif st["do"] == "push_object_to":
            spec = to_taskspec(goal, scene).model_copy(deep=True)
            spec.goal.x.value, spec.goal.y.value = val(st["x"]), val(st["y"])
            if st["facing"] is None:
                spec.goal.yaw_tol_rad.value = math.pi
            else:
                spec.goal.yaw.value = math.radians(val(st["facing"]))
            r = run_learned_go1_push(spec, None, None, 0, robot=rb, log=lambda s: None, obstacles=obstacles,
                                     on_tick=lambda r_: on_tick(rb), reset=False)
            rb.on_tick = on_tick
            note = f"{len(r.pushes)} foot pushes ({r.stop_reason})"
        else:
            rb.walk((0, 0, 0), st["seconds"])
            note = "waited"
        after = rb.env.block_pose()
        moved = math.hypot(after.x - before.x, after.y - before.y)
        obj = f"; the {scene.object_name} is at ({after.x:.2f}, {after.y:.2f})"
        if moved > 0.01:
            obj += f" (moved {moved * 100:.0f} cm)"
        if math.degrees(after.tilt) > 15 and getattr(b, "shape", "box") != "sphere":
            obj += f", tipped {math.degrees(after.tilt):.0f}°"
        if rb.fallen():
            obj += "; THE ROBOT FELL OVER"
        report.append(f"{i}. {describe(st)}: {note}{obj}")
        if rb.fallen():
            break
    rb.walk((0, 0, 0), settle_s) if not rb.fallen() else None
    pose = rb.env.block_pose()
    res = check(goal, scene, rb.env.model, rb.env.data, pose, rb.env.block_speed(), True, history=hist)
    if rb.fallen():
        res["success"] = False
        res["reasons"].insert(0, "the robot fell over")
    from .goals import placement
    pl = placement(goal, scene)
    if pl.get("has_place"):
        miss = math.hypot(pose.x - pl["x"], pose.y - pl["y"])
        if miss > 0.02:
            report.append(f"in the end the {scene.object_name} is at ({pose.x:.2f}, {pose.y:.2f}): {miss * 100:.0f} cm from "
                          f"where it had to end ({pl['x']:.2f}, {pl['y']:.2f})")
    return {"success": res["success"], "results": res["results"], "reasons": res["reasons"], "report": report}


def _fixtures(scene) -> dict:
    from .goal_terms import fixtures_geometry
    return fixtures_geometry(scene)


def solve_legged(pj, goal: GoalSpec, prof, llm, log=print, emit=lambda **e: None, n: int = 5, attempts: int = 4,
                 stand_at=None) -> dict:
    """Nemotron writes a legged program; Morph runs it on the robot's learned gait and reports what happened, until
    the checklist is met; then it's tried from n slightly different starts."""
    from .goals import explain, placement
    from .project import _free_spot, _params, _start_poses
    from .skill_steps import robot_brief
    scene = pj.scene
    rb = make_robot(prof, scene)
    if rb is None:
        return {"success": 0, "n": 0, "setup_error": f"no walking skill for {prof.name} yet (train one with morph walk)"}
    p = placement(goal, scene)
    base0 = tuple(stand_at) if stand_at else _free_spot(pj, scene.start[:2], (p["x"], p["y"]))
    top = max([f.top_z for f in scene.fixtures.values()] + [0.06])
    names = scene_names(scene, top + 0.12, 0.0, (p["x"], p["y"]) if p.get("has_place") else None)
    rb.reset(0, scene.start, _params(0), base_xy_yaw=base0)
    pos, th, _ = rb.base()
    facts = {"robot_starts_at": [round(float(v), 3) for v in pos[:2]], "robot_facing_deg": round(math.degrees(th)),
             "body_length_m": prof.size_m.get("length"), "body_width_m": prof.size_m.get("width"),
             "object_half_size_m": [scene.block.half_x, scene.block.half_y, scene.block.half_z],
             "object_start": [round(float(v), 3) for v in scene.start],
             "where_it_must_end": {k: round(float(p[k]), 3) for k in ("x", "y")} if p.get("has_place") else None,
             "foot_push_heights_m": {"low": LOW, "high": round(min(HIGH, 2 * scene.block.half_z - 0.008), 3)},
             "names_for_formulas": {k: round(v, 3) for k, v in names.items()}}
    checklist = [f"{c.describe()}: {explain(c, scene)}" for c in goal.conditions]
    key = json.dumps([prof.name, [c.model_dump() for c in goal.conditions], str(getattr(scene, "path", "")),
                      list(scene.start), [round(float(v), 3) for v in base0]], sort_keys=True, default=str)
    cache = CACHE / f"legged_{prof.name}__{hashlib.sha1(key.encode()).hexdigest()[:12]}.json"
    tries, steps, ok_run, a, fixes = [], None, None, 0, 0
    planner = LegPlanner(rb, scene, base0, names, log)
    if cache.exists():
        try:
            cached = check_leg_program(json.loads(cache.read_text())["program"], set(names), planner)
            rb.reset(0, scene.start, _params(0), base_xy_yaw=base0)
            run = run_leg_program(rb, cached, goal, scene, names)
            if run["success"]:
                log("  using the program that worked last time (no new Nemotron call)")
                steps, ok_run, attempts = cached, run, 0
        except (ProgramError, KeyError, ValueError):
            pass
    from .programs import feedback
    while a < attempts:
        emit(type="stage", robot=prof.name, stage="trials", text=f"Nemotron is writing program {a + 1}")
        try:
            prop, rec = llm.write_program(robot_brief(prof), checklist, scene.geometry(), facts, BLOCKS_DOC, tries,
                                          body="legged")
        except Exception as e:  # noqa: BLE001
            log(f"  Nemotron's reply couldn't be read ({e}); asking again")
            fixes += 1
            if fixes > 3:
                break
            continue
        try:
            steps = check_leg_program(prop.steps, set(names), planner)
        except ProgramError as e:
            tries.append({"program": prop.steps, "problem": f"Morph couldn't run it: {e}"})
            log(f"  program {a + 1}: can't run it: {e}")
            fixes += 1
            if fixes > 3:
                a += 1
            continue
        log(f"  program {a + 1} ({rec.latency_s}s): " + "; ".join(describe(s) for s in steps))
        emit(type="program", robot=prof.name, attempt=a + 1, steps=[describe(s) for s in steps], rationale=prop.rationale)
        rb.reset(0, scene.start, _params(0), base_xy_yaw=base0)
        run = run_leg_program(rb, steps, goal, scene, names,
                              on_step=lambda i, d, k=a: emit(type="caption", robot=prof.name, text=f"Program {k + 1}, step {i}: {d}"))
        tries.append({"program": steps, "rationale": prop.rationale, "result": feedback(run)})
        emit(type="program_result", robot=prof.name, attempt=a + 1, success=run["success"], report=run["report"],
             reasons=run["reasons"])
        log(f"  program {a + 1}: {'reached the goal' if run['success'] else 'failed: ' + '; '.join(run['reasons'][:2])}")
        if run["success"]:
            ok_run = run
            break
        a += 1
    out = {"program_attempts": tries}
    if ok_run is None:
        return out | {"success": 0, "n": 0, "why_not": f"Nemotron's programs didn't reach the goal in {a} tries"}
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"robot": prof.name, "goal": goal.describe(), "program": steps}, indent=1))
    trials = []
    for i, start in enumerate(_start_poses(scene, n)):
        emit(type="trial_start", robot=prof.name, i=i, n=n)
        rb.reset(i, start, _params(i), base_xy_yaw=base0)
        r = run_leg_program(rb, steps, goal, scene, names)
        trials.append({"success": r["success"], "conditions": r["results"], "reasons": r["reasons"]})
        emit(type="trial", robot=prof.name, i=i, n=n, success=r["success"], conditions=r["results"], reasons=r["reasons"])
        log(f"  trial {i + 1}: {'success' if r['success'] else 'fail: ' + '; '.join(r['reasons'][:2])}")
    ok = sum(t["success"] for t in trials)
    log(f"{prof.name}: {ok}/{n} with Nemotron's program")
    return out | {"success": ok, "n": n, "trials": trials, "program": steps}
