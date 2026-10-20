"""A user's project: their scene and object, their robot, and a goal given as a demonstration, a
description, or both.

    my_project/
      morph.toml        [project] scene, object, robot   [task] description and/or demo
      scene.xml         the object plus bins / shelves / walls (MJCF)
      demo.npz          optional: what happened to the object when the task was done before

`morph do my_project` measures or interprets the goal, shows it for confirmation, then plans for
the robot, sets it up (or finds its trained gait), runs trials from varied start poses and object
physics, and checks every goal condition in the simulation.
"""

from __future__ import annotations

import json
import math
import re
import time
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .goal_sources import goal_from_demo, goal_from_description_rules, load_demo, merge, summarize_demo
from .goals import Condition, GoalSpec, unsatisfiable, check, placement, to_taskspec, validate, warnings
from .user_scene import UserScene, load_scene


# Optional listener for structured progress (the app sets it): called with dicts like
# {"type": "stage", ...}, {"type": "trial", ...}, {"type": "caption", ...}. The terminal ignores it.
LISTENER = None


def emit(**event) -> None:
    if LISTENER is not None:
        LISTENER(event)


class GoalNotPossible(ValueError):
    """The request can't work in this scene (said in physical terms, with a suggestion)."""


@dataclass
class Project:
    root: Path
    name: str
    scene: UserScene
    robot: str | None
    description: str
    demo: str | None


def load_project(root: str | Path) -> Project:
    root = Path(root)
    cfg_path = root / "morph.toml"
    cfg = tomllib.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    pj, task = cfg.get("project", {}), cfg.get("task", {})

    def rel(p):
        return str((root / p).resolve()) if p else None

    scene = load_scene(rel(pj.get("scene", "scene.xml")), pj.get("object"))
    return Project(root=root, name=pj.get("name", root.name), scene=scene, robot=rel(pj.get("robot")),
                   description=task.get("description", ""), demo=rel(task.get("demo")))


def _spins(c) -> bool:
    """A condition that it keeps turning at a real speed: spin_dps (or spin_z_dps) over 60 deg/s, for a stretch of
    time (the last N seconds, or the whole time)."""
    m = re.search(r"\bspin(?:_z)?_dps\s*>=?\s*(\d+(?:\.\d+)?)", c.expr or "")
    return c.kind == "expression" and c.when in ("last", "always") and m is not None and float(m.group(1)) >= 60


_MUST = [  # words in a request -> what its checklist has to contain (checked without the model)
    (r"\b(throw|toss|lob|fling|chuck)\w*\b|(?<!don't )(?<!not )(?<!no )\bdrop(s|ped|ping)?\b(?! it gently)",
     lambda c: c.when == "release",
     "the request is a throw or a drop, so the checklist needs a condition at the moment the robot lets go"),
    (r"\b(first\b.*\bthen|on the way|at some point|along the way|before (putting|placing|moving))\b", lambda c: c.when == "ever",
     "the request has something that happens on the way (first ... then ...), so it needs an 'at some point' condition"),
    (r"\b(never|without (ever )?(lifting|tipping|touching|dropping)|the whole (time|way)|all the way)\b",
     lambda c: c.when == "always", "the request says something must hold the whole time, so it needs a 'the whole time' condition"),
    (r"\b(spin|twirl|twiddle)\w*\b", _spins,
     "the request is to spin it, so the checklist needs it turning at a real speed (spin_dps over 60) for a while"),
    (r"\b(knock|tip|topple|push)\w*\s+(?:\w+\s+){0,2}over\b|\bon its side\b",
     lambda c: c.kind == "orientation" and c.up_face == "side", "the request is to knock it over, so it must end lying on its side"),
]


def missing_intent(description: str, goal: GoalSpec) -> list[str]:
    """What the request clearly asks for that the checklist doesn't have (the model sometimes drops it)."""
    t = description.lower()
    return [why for pat, ok, why in _MUST if re.search(pat, t) and not any(ok(c) for c in goal.conditions)]


def robot_bodies(paths) -> list[dict]:
    """What each robot's body has, for goals about the robot itself: its joints (with ranges, degrees) and end parts."""
    import mujoco
    from .robot_import import import_robot
    out = []
    for p in paths or []:
        try:
            xml, _ = import_robot(p)
            m = mujoco.MjModel.from_xml_path(str(xml))
        except Exception:  # noqa: BLE001  (a robot that doesn't load is reported elsewhere)
            continue
        joints = {}
        for j in range(m.njnt):
            if int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                r = m.jnt_range[j] if m.jnt_limited[j] else (-math.pi, math.pi)
                deg = int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)
                joints[m.joint(j).name] = [round(math.degrees(v) if deg else v, 1 if deg else 3) for v in r]
        has_child = {int(m.body_parentid[b]) for b in range(1, m.nbody)}
        parts = [m.body(b).name for b in range(1, m.nbody) if b not in has_child]
        from .fingers import finger_names
        hands = [{"side": h["side"], "fingers": list(h["fingers"])} for h in finger_names(m)]
        out.append({"robot": Path(xml).stem, "joints": dict(list(joints.items())[:40]), "end_parts": parts[:20]}
                   | ({"hands": hands} if hands else {}))
    return out


def resolve_goal(pj: Project, description: str = "", demo_path: str | None = None, llm=None, log=print,
                 robots: list[str] | None = None) -> tuple[GoalSpec, dict]:
    """Demonstration -> measured conditions; description -> conditions (Nemotron, else rules); both -> merged
    with the description taking precedence. Everything is validated against the scene."""
    info: dict = {}
    demo_goal = demo_summary = None
    if demo_path:
        demo = load_demo(demo_path, pj.scene.object_name)
        demo_summary = summarize_demo(demo, pj.scene)
        demo_goal = goal_from_demo(demo, pj.scene)
        info["demo"] = demo_summary
        log(f"demonstration ({demo_summary['source']}): moved {demo_summary['moved_cm']} cm, turned "
            f"{demo_summary['turned_deg']} deg, lifted to {demo_summary['max_height_cm']} cm; ended "
            + (", ".join(demo_summary["end_relations"]) or "on the floor"))
    text_goal = None
    if description:
        text_goal = goal_from_description_rules(description, pj.scene)
        info["rule_goal"] = [c.model_dump() for c in text_goal.conditions]
    goal, planner = merge(demo_goal, text_goal), "measured" if not description else "rules"
    if llm is not None and llm.available and (description or demo_goal):
        try:
            feedback, total_s = None, 0.0
            bodies = robot_bodies(robots)  # so goals can be about the robots' own bodies (a fist, a raised arm)
            for attempt in range(3):  # a first reading, and up to two corrections from Morph's own checks
                prop, rec = llm.goal_from_request(description, pj.scene.geometry(), demo_summary, feedback, robots=bodies)
                total_s += rec.latency_s or 0.0
                if not prop.feasible:
                    if prop.refusal in ("missing", "contradiction", "no_inside") or attempt == 2:
                        raise GoalNotPossible(prop.why_not + (f" Try: {prop.suggestion}" if prop.suggestion else ""))
                    # merely hard or unusual: Morph tries those (with a warning), so ask again
                    feedback = {"you_answered": {"feasible": False, "why_not": prop.why_not},
                                "morph_says": "Only refuse when the request names something that is not in the scene, "
                                              "contradicts itself, or asks for the inside of a solid thing. This is "
                                              "possible, even if hard: write the conditions and put the risk in warnings."}
                    log("  (Nemotron thought it impossible; asked again: Morph tries hard things)")
                    continue
                conds = [Condition(**({k: v for k, v in c.items() if k in Condition.model_fields} |
                                      {"source": "description" if description else "demo"})) for c in prop.conditions]
                cand = GoalSpec(object=pj.scene.object_name, conditions=conds,
                                notes=[f"Nemotron: {prop.rationale}"] + [f"assumed: {a}" for a in prop.ambiguities],
                                warnings=list(prop.warnings), start="in_hand" if prop.start == "in_hand" else "scene")
                errs = validate(cand, pj.scene) or (missing_intent(description, cand) if description else []) \
                    or unsatisfiable(cand, pj.scene)
                problems = errs
                if not errs and description and attempt < 2:
                    # does the checklist, read back in plain words, mean what was asked? (catches reversals like
                    # "knock it over" -> "stays upright", and conditions nobody asked for)
                    rv, rrec = llm.review_goal(description, [c.describe() for c in cand.conditions]
                                               + ([cand.start_text()] if cand.start_text() else []), pj.scene.geometry(),
                                               [why for pat, _, why in _MUST if re.search(pat, description.lower())])
                    total_s += rrec.latency_s or 0.0
                    problems = [] if rv.ok else (rv.problems or ["the checklist doesn't match the request"])
                if not problems:
                    goal, planner = cand, f"nemotron ({total_s:.1f}s)"
                    break
                info.setdefault("nemotron_corrections", []).append(problems)
                log(f"  (Nemotron's first reading had problems: {'; '.join(problems)[:300]}; asking again)")
                feedback = {"your_conditions": prop.conditions, "problems": problems}
            else:
                log(f"  (Nemotron's goal still had problems; using the {planner} reading)")
        except GoalNotPossible:
            raise
        except Exception as e:  # noqa: BLE001  (no key, network, bad JSON: fall back, say so)
            log(f"  (Nemotron unavailable: {e}; using the {planner} reading)")
    if description and goal is not None and goal is not text_goal and text_goal is not None:
        # the model's reading, still missing something the request clearly says: take that part from the
        # built-in reader rather than drop it
        for pat, ok, why in _MUST:
            if re.search(pat, description.lower()) and not any(ok(c) for c in goal.conditions):
                extra = [c for c in text_goal.conditions if ok(c)]
                if extra:
                    goal.conditions += extra
                    goal.notes.append(f"added from the built-in reader: {why}")
                    log(f"  (added from the built-in reader: {why})")
    if description and goal is not None and text_goal is not None and text_goal.start != goal.start \
            and text_goal.start == "in_hand":
        goal.start = "in_hand"  # "spin it in the fingers" can't start on the floor
        goal.notes.append("starts in the hand (from the built-in reader)")
    if goal is not None and goal.start == "in_hand" and not any(
            c.when == "always" and (c.kind == "height" or "bottom" in (c.expr or "") or "held" in (c.expr or ""))
            for c in goal.conditions):  # handled in the hand: dropping it and rolling it on the floor doesn't count
        goal.conditions.append(Condition(kind="expression", expr="bottom > 0.02", when="always", source="default",
                                         text="it never falls to the floor"))
    if goal is None:
        raise ValueError("no goal: give a description, a demonstration, or both")
    errs = validate(goal, pj.scene)
    if errs:
        raise ValueError("the goal doesn't fit this scene: " + "; ".join(errs))
    measured = warnings(goal, pj.scene)
    # Morph's own (measured) warnings win; Nemotron's are kept only when Morph has none, and never for an
    # object sticking out of a container, which is normal
    llm_w = [w for w in goal.warnings if not re.search(r"protrud|stick(s|ing)? out|above the (bin'?s )?rim|\bis feasible\b|"
                                                       r"comfortabl|remains (physically )?possible|no (real )?risk", w, re.I)]
    goal.warnings = measured or llm_w
    info["goal_from"] = planner
    return goal, info


# ------------------------------------------------------------------ running a goal on a robot
def _start_poses(scene: UserScene, n: int, seed: int = 0) -> list[tuple[float, float, float]]:
    """The scene's start pose, then small variations of it (+/- 2 cm, +/- 10 deg)."""
    rng = np.random.default_rng(seed)
    x, y, yaw = scene.start
    out = [(x, y, yaw)]
    for _ in range(n - 1):
        out.append((x + rng.uniform(-0.02, 0.02), y + rng.uniform(-0.02, 0.02), yaw + math.radians(rng.uniform(-10, 10))))
    return out


def _params(i: int):
    from .learning import sample_scenario
    return sample_scenario(30000 + i).params  # object mass / friction varied per trial


def _recorder(env):
    """Wrap env.advance to log the object's pose every control tick (for --record-demo)."""
    frames = []
    orig = env.advance

    def advance(*a, **k):
        r = orig(*a, **k)
        q = env.data.qpos[env._bq:env._bq + 7]
        frames.append(q.copy())
        return r
    env.advance = advance
    return frames


def run_on_robot(pj: Project, goal: GoalSpec, robot_path: str, n: int = 5, llm=None, log=print,
                 record_demo: str | None = None, stand_at: tuple | None = None) -> dict:
    """stand_at: (x, y, heading) where the user put the robot; None: Morph chooses."""
    from .introspect import profile_robot
    from .goals import needs_learning, reach_problem
    from .planner import body_goal, check_templates, learning_only, plan_context, rule_plan
    from .robot_import import import_robot
    robot_path, notes = import_robot(robot_path)  # URDF -> MJCF (meshes, motors, base) if needed
    robot_path = str(robot_path)
    for note in notes:  # (not n: that's the number of trials)
        log(f"  import: {note}")
    p = placement(goal, pj.scene)
    spec = to_taskspec(goal, pj.scene)
    prof = profile_robot(robot_path)
    checks = body_goal(learning_only(check_templates(prof, spec, pj.scene.block, lift_reason=p["lift_reason"]),
                                     needs_learning(goal), reach_problem(prof, goal, pj.scene)), goal)
    log(f"\n{prof.summary()}")
    for c in checks:
        log(f"  {'ok ' if c.applicable else '-- '}{c.id:20s} " + ("; ".join(c.met) if c.applicable else "missing: " + "; ".join(c.missing)))
    plan = rule_plan(checks) | {"planner": "rule"}
    if llm is not None and llm.available:
        try:
            ctx = plan_context(prof, spec, checks)
            ctx["task"]["conditions"] = goal.describe()
            ctx["task"]["object"] = pj.scene.summary()["object"]
            ctx["task"]["scene"] = pj.scene.summary()["fixtures"]
            pl, rec = llm.plan_for_robot(ctx)
            plan = pl.model_dump() | {"planner": f"nemotron ({rec.latency_s}s)"}
        except Exception as e:  # noqa: BLE001
            log(f"  (Nemotron unavailable for planning: {e}; using the rule)")
    log(f"plan: {plan['plan_id']}  [{plan['planner']}]")
    emit(type="plan", robot=prof.name, plan_id=plan["plan_id"], planner=plan["planner"],
         how=plan.get("how") or plan.get("rationale", ""), what_is_it=plan.get("what_is_it", prof.summary()),
         templates=[{"id": c.id, "ok": c.applicable, "met": c.met, "missing": c.missing} for c in checks])
    if plan.get("how"):
        log(f"  how: {plan['how']}")
    from .learn_task import learned_for
    learned = learned_for(prof.name, goal, robot_path)
    if learned is not None and stand_at is not None:
        mx, my, myaw = json.loads(learned.read_text())["mount"]
        if math.hypot(mx - stand_at[0], my - stand_at[1]) > 0.05 or abs(math.remainder(myaw - stand_at[2], math.tau)) > 0.17:
            log("  (its learned skill was learned standing somewhere else: not used from this spot)")
            learned = None
    if learned is not None:
        plan = {"plan_id": "learned_skill", "planner": "learned", "how": f"use the skill it learned for this goal ({learned.parent.name})"}
        emit(type="plan", robot=prof.name, plan_id="learned_skill", planner="learned", how=plan["how"], what_is_it=prof.summary(),
             templates=[{"id": c.id, "ok": c.applicable, "met": c.met, "missing": c.missing} for c in checks])
    out = {"robot": prof.name, "plan": plan, "placement": p, "templates": [c.__dict__ for c in checks]}
    why = body_problem(goal, robot_path, prof)
    if why and plan["plan_id"] != "learned_skill":
        log(f"  {why}")
        emit(type="robot_done", robot=prof.name, plan_id=plan["plan_id"], success=0, n=0, why_not=why)
        return out | {"success": 0, "n": 0, "why_not": why}
    if plan["plan_id"] == "learned_skill":
        from .learn_task import try_policy
        emit(type="stage", robot=prof.name, stage="trials", text="using its learned skill")
        out |= try_policy(pj, goal, learned, n=n, log=log, emit=emit)
        log(f"{prof.name}: {out['success']}/{out['n']}")
    elif plan["plan_id"] in ("grasp_place_topdown", "legged_foot_push"):
        if plan["plan_id"] == "grasp_place_topdown":
            out |= _run_arm(pj, goal, spec, p, robot_path, prof.name, n, log, record_demo, stand_at)
        else:
            out |= _run_legged(pj, goal, spec, p, prof, n, log, record_demo, stand_at)
        if (out.get("n") and 2 * out.get("success", 0) < out["n"] and llm is not None and llm.available
                and program_body(prof)):
            # the ready-made skill mostly failed here (it's tuned for a 6 cm box; this object may be a pen, a ball):
            # Nemotron writes a program for this object instead, and the better of the two is kept
            log(f"  the ready-made skill worked {out['success']}/{out['n']}: Nemotron writes a program instead")
            first = {k: out[k] for k in ("success", "n", "trials", "plan") if k in out}
            alt = run_program_for(pj, goal, robot_path, prof, llm, log, n, stand_at, out)
            if alt.get("success", 0) > first.get("success", 0):
                out |= alt
                plan = out["plan"]
            else:
                out |= first
    elif plan["plan_id"] == "learn_policy" and llm is not None and llm.available and program_body(prof):
        # no ready-made skill: Nemotron writes a program from building blocks, Morph runs and checks it, and
        # Nemotron fixes it from what happened. Seconds per try, instead of minutes of learning.
        out |= run_program_for(pj, goal, robot_path, prof, llm, log, n, stand_at, out)
        plan = out["plan"]
        if not out.get("success"):
            out["why_not"] = (out.get("why_not") or out.get("setup_error", "")) + \
                "; it can still learn one: open the Robot tab and choose Learn this task"
    else:
        if plan["plan_id"] == "learn_policy":
            grasp = next((c for c in checks if c.id == "grasp_place_topdown"), None)
            reason = needs_learning(goal) or (grasp.missing[0] if grasp and grasp.missing else "none of the ready-made skills fit")
            why = (f"no ready-made skill fits this robot for this goal ({reason}); it can learn one: "
                   "open the Robot tab and choose Learn this task")
        elif p["needs_lift"]:
            why = f"the goal needs the object lifted ({p['lift_reason']}) and this robot can only push"
        else:
            why = plan.get("rationale", "")
        log(f"nothing this robot can run for this goal yet ({plan['plan_id']}): {why}")
        out["success"], out["n"] = 0, 0
        out["why_not"] = why
    if (out.get("setup_error") and not out.get("n") and plan["plan_id"] != "program" and llm is not None and llm.available
            and program_body(prof)):
        # the ready-made skill couldn't be set up here (e.g. out of reach from where it could stand): a program instead
        log(f"  the ready-made skill couldn't be set up ({out['setup_error']}): Nemotron writes a program instead")
        out.pop("setup_error")
        out |= run_program_for(pj, goal, robot_path, prof, llm, log, n, stand_at, out)
        plan = out["plan"]
    if out.get("setup_error") and not out.get("n"):
        out["why_not"] = (f"the ready-made skill couldn't be set up: {out['setup_error']}; it can learn one instead: "
                          "open the Robot tab and choose Learn this task")
        log(f"  {out['why_not']}")
    emit(type="robot_done", robot=prof.name, plan_id=plan["plan_id"], success=out.get("success", 0), n=out.get("n", 0),
         why_not=out.get("why_not", ""))
    return out


def body_problem(goal: GoalSpec, robot_path: str, prof) -> str:
    """Why this robot can't run this goal at all, before anything runs: a finger or joint the goal names that it doesn't
    have, or a goal that starts in the hand. "" if none."""
    import mujoco
    from .fingers import part_names
    from .goal_terms import missing_parts
    try:
        miss = missing_parts([c.model_dump() for c in goal.conditions], part_names(mujoco.MjModel.from_xml_path(robot_path), ""))
    except Exception:  # noqa: BLE001  (a model that won't load is reported elsewhere)
        miss = []
    if miss:
        return f"{prof.name} has nothing called {' or '.join(repr(x) for x in miss)}, which the goal refers to"
    if goal.start == "in_hand":
        if prof.gripper.get("type") != "multi_finger_hand":
            return f"the {goal.object} starts in the hand, and {prof.name} has no hand with fingers to hold it in"
        return ("the goal starts with the object in the hand: programs pick things up from the floor, so this is "
                "learned: open the Robot tab and choose Learn this task")
    return ""


def program_body(prof) -> str | None:
    """Which building blocks Nemotron can write a program from for this body: "arm" (a bolted-down arm with a parallel
    gripper: programs.py), "body" (any other bolted-down robot: a gripperless arm, a hand, a tilted grasp:
    body_programs.py), "legged" (a walking robot with a learned gait: legged_programs.py), or None."""
    from .planner import walk_policy_dir
    if prof.base == "fixed":
        return "arm" if prof.has_gripper else "body"
    if prof.base == "floating" and prof.n_legs >= 4 and walk_policy_dir(prof) is not None:
        return "legged"
    return None


def run_program_for(pj, goal, robot_path, prof, llm, log, n, stand_at, out) -> dict:
    """Nemotron's program for this body (program_body), run and checked; an arm whose hand-down blocks don't work out
    gets a second go with the general blocks (any part, any pointing)."""
    from .learn_task import LEARNED
    kind = program_body(prof)
    how = {"arm": "Nemotron writes a program from building blocks (move the hand, open, close, throw); Morph runs it and "
                  "tells it what happened until the checklist is met",
           "body": "Nemotron writes a program that moves the robot's parts (any part, pointing any way); Morph runs it and "
                   "tells it what happened until the checklist is met",
           "legged": "Nemotron writes a program from walking building blocks (walk to, turn, push with a foot, bump); Morph "
                     "runs it on the robot's learned gait and tells it what happened until the checklist is met"}

    def announce(k):
        plan = {"plan_id": "program", "planner": "nemotron", "how": how[k]}
        emit(type="plan", robot=prof.name, plan_id="program", planner="nemotron", how=plan["how"], what_is_it=prof.summary(),
             templates=[{"id": t["id"], "ok": t["applicable"], "met": t["met"], "missing": t["missing"]}
                        for t in out.get("templates", [])])
        return plan

    res: dict = {}
    from .programs import too_flat_for_hand_down
    if kind == "arm" and too_flat_for_hand_down(pj.scene):
        # a flat object (a pen, a phone): the general blocks grip it from just above the floor (pick) and can throw
        # it too; the hand-down blocks can't get the fingers that low
        kind = "body"
    if kind == "arm":
        from .programs import solve
        plan_ = announce("arm")
        res = solve(pj, goal, robot_path, prof.name, llm, log=log, emit=emit, n=n, stand_at=stand_at)
        res.pop("adapter", None), res.pop("demos", None)  # for learning from; not part of the result
        res["plan"] = plan_
        if res.get("success"):
            return res
        log("  the hand-down building blocks didn't work out: trying the general ones (any part, any pointing)")
        kind = "body"
        first = res
    if kind == "body":
        from .body_programs import solve_body
        from .learn_task import make_task, slug
        folder = LEARNED.parent / "programs" / "tasks" / f"{prof.name}__{slug(' '.join(goal.describe())) or 'goal'}"
        task = make_task(pj, goal, robot_path, plan=False, stand_at=stand_at, folder=folder)
        plan_ = announce("body")
        r = solve_body(pj, goal, task, llm, log=log, emit=emit, n=n, attempts=3 if res else 5)
        r.pop("demos", None)
        r["plan"] = plan_
        if res and not r.get("success") and not r.get("n"):
            return first  # the arm blocks got further: report those
        return r
    from .legged_programs import solve_legged
    plan_ = announce("legged")
    r = solve_legged(pj, goal, prof, llm, log=log, emit=emit, n=n, stand_at=stand_at)
    r["plan"] = plan_
    return r


def _finish_trial(pj, goal, env, released, t0, i, log, robot="", n=0, fell=False) -> dict:
    pose = env.block_pose()
    res = check(goal, pj.scene, env.model, env.data, pose, env.block_speed(), released)
    if fell:
        res["success"] = False
        res["reasons"].append("robot fell")
    emit(type="trial", robot=robot, i=i, n=n, success=res["success"], conditions=res["results"], reasons=res["reasons"])
    log(f"  trial {i + 1}: {'success' if res['success'] else 'fail: ' + '; '.join(res['reasons'][:2])}  "
        f"({time.time() - t0:.0f}s)")
    return {"success": res["success"], "conditions": res["results"], "reasons": res["reasons"],
            "final": pose.as_dict()}


def _save_demo(path: str, frames: list, robot: str, log) -> None:
    arr = np.array(frames)
    np.savez(path, pos=arr[:, :3], quat=arr[:, 3:7], t=np.arange(len(arr)) * 0.02, robot=robot)
    log(f"  recorded the object's path ({len(arr)} frames) to {path}")


def _run_arm(pj, goal, spec, p, robot_path, name, n, log, record_demo, placement=None) -> dict:
    from .arm_skills import expert_action, released, run_episode
    from .robots.auto_arm import NotAGraspingArm, build_auto_arm
    b = pj.scene.block
    # carry high enough that the object's bottom clears what it passes over by 2 cm (the TCP sits
    # ~1.5 cm above the object's centre when grasped)
    min_carry = p["clear_z"] + 0.02 + 2 * b.half_z - 0.015 if p["needs_lift"] else None
    log("\nsetting the arm up by experiment" + (f" (carry height >= {min_carry * 100:.0f} cm to clear "
                                               f"{p['lift_reason']})" if min_carry else "") + ":")
    emit(type="stage", robot=name, stage="setup", text="setting the arm up: mount, carry height, servos, grip")
    try:
        ad, rep = build_auto_arm(robot_path, name, block=b, log=log, task_xy=[pj.scene.start[:2], (p["x"], p["y"])],
                                 min_carry_z=min_carry, start=pj.scene.start,
                                 keepout=[(f.lo, f.hi) for f in pj.scene.fixtures.values()], mount=placement)
    except NotAGraspingArm as e:
        log(f"  can't: {e}")
        return {"success": 0, "n": 0, "setup_error": str(e)}
    if not (ad.reach_ok(*pj.scene.start[:2]) and ad.reach_ok(p["x"], p["y"])):
        log("  the start or the goal is outside this arm's reach from any mount tried")
        return {"success": 0, "n": 0, "setup": rep, "setup_error": "out of reach"}
    emit(type="stage", robot=name, stage="trials", text=f"mount {rep['base_x']:+.2f} m, carry {rep['travel_z'] * 100:.0f} cm, "
                                                   f"grip x{ad.cfg.gripper_gain_scale:g}")
    captions = {"approach": "moving over the object", "descend": "lowering to grasp", "close": "closing the gripper",
                "open": "opening the gripper", "raise": "backing off to try again", "lift": "lifting",
                "transport": "carrying", "lower": "setting it down", "release_retreat": "letting go"}
    last = [None]

    def skill(s):
        a, ph = expert_action(s)
        if ph != last[0]:
            last[0] = ph
            emit(type="caption", robot=name, text=captions.get(ph, ph))
        return a, ph
    from .learn_task import slug
    from .moveit_export import MOVEIT, Recorder, scene_in_base, write_moveit_export
    trials, moveit = [], None
    for i, start in enumerate(_start_poses(pj.scene, n)):
        t0 = time.time()
        ad.reset(i, start, _params(i))
        ad.env.set_goal_marker(p["x"], p["y"], p["yaw"])
        emit(type="trial_start", robot=name, i=i, n=n)
        frames = _recorder(ad.env) if record_demo else None
        rec = Recorder(ad)
        run_episode(ad, spec, skill, max_time=20.0, on_tick=rec)
        trials.append(_finish_trial(pj, goal, ad.env, released(ad), t0, i, log, name, n))
        if record_demo and trials[-1]["success"] and not Path(record_demo).exists():
            _save_demo(record_demo, frames, name, log)
        if trials[-1]["success"] and moveit is None:  # the sequence that worked, for MoveIt
            goal_text = "; ".join(c.describe() for c in goal.conditions)
            try:
                moveit = str(write_moveit_export(name, goal_text, rec.finish(), scene_in_base(ad, pj.scene.fixtures, b), b,
                                                 MOVEIT / f"{name}__{slug(goal_text)}", ad.cfg.tcp_body))
            except Exception as e:  # noqa: BLE001  (an export problem must not fail the run)
                log(f"  MoveIt export skipped: {e}")
    ok = sum(t["success"] for t in trials)
    log(f"{name}: {ok}/{len(trials)}")
    return {"setup": rep, "success": ok, "n": len(trials), "trials": trials, "moveit_export": moveit}


def _free_spot(pj, obj_xy, goal_xy, dist=0.7):
    """Robot start: `dist` behind the object, opposite the goal, rotated until clear of fixtures."""
    d = np.array(goal_xy) - np.array(obj_xy)
    base = math.atan2(-d[1], -d[0]) if np.linalg.norm(d) > 1e-3 else -math.pi / 2
    for k in range(12):
        a = base + (k + 1) // 2 * (0.3 if k % 2 else -0.3)
        xy = np.array(obj_xy) + dist * np.array([math.cos(a), math.sin(a)])
        if all(not (f.lo[0] - 0.4 < xy[0] < f.hi[0] + 0.4 and f.lo[1] - 0.4 < xy[1] < f.hi[1] + 0.4)
               for f in pj.scene.fixtures.values()):
            return float(xy[0]), float(xy[1]), a + math.pi
    return float(obj_xy[0] + dist * math.cos(base)), float(obj_xy[1] + dist * math.sin(base)), base + math.pi


def _run_legged(pj, goal, spec, p, prof, n, log, record_demo, placement=None) -> dict:
    from .go1_learned_task import run_learned_go1_push
    from .planner import walk_policy_dir
    pol = walk_policy_dir(prof)
    if pol.name == "go1_walk":
        from .robots.go1_learned import LearnedGo1
        rb = LearnedGo1(block=pj.scene.block)
    else:
        from .robots.learned_walker import LearnedWalker
        rb = LearnedWalker(pol, block=pj.scene.block)
    touch = [c for c in goal.conditions if c.kind == "touching"]
    if touch and not any(c.kind in ("inside", "on_top_of", "at_position") for c in goal.conditions):
        # aim 1 cm into the fixture: the planner's model has it too, so pushes end with the object against it
        f = pj.scene.fixtures[touch[0].target]
        c = (f.lo[:2] + f.hi[:2]) / 2
        d = c - np.array([p["x"], p["y"]])
        d = d / (np.linalg.norm(d) + 1e-9)
        axis = np.argmax(np.abs(d * (f.hi[:2] - f.lo[:2] < 0.1)))  # push across the fixture's thin side if it has one
        step = np.zeros(2)
        step[axis] = 0.013 * np.sign(d[axis])
        spec = spec.model_copy(deep=True)
        spec.goal.x.value, spec.goal.y.value = p["x"] + step[0], p["y"] + step[1]
    emit(type="stage", robot=prof.name, stage="trials", text=f"learned gait from {pol.name}, three-leg foot push")

    def tick(r):
        mode = getattr(r, "mode", "learned")
        if mode != tick.last:
            tick.last = mode
            emit(type="caption", robot=prof.name, text="walking (learned gait)" if mode == "learned" else "pushing with one foot")
    tick.last = None
    trials = []
    for i, start in enumerate(_start_poses(pj.scene, n)):
        t0 = time.time()
        emit(type="trial_start", robot=prof.name, i=i, n=n)
        base0 = tuple(placement) if placement else _free_spot(pj, start[:2], (p["x"], p["y"]))
        frames = _recorder(rb.env) if record_demo else None
        obstacles = [(f.lo, f.hi) for f in pj.scene.fixtures.values() if f.top_z > 0.02]  # too tall to step over

        def done():
            return check(goal, pj.scene, rb.env.model, rb.env.data, rb.env.block_pose(), rb.env.block_speed(), True)["success"]
        r = run_learned_go1_push(spec, start, _params(i), i, base0=base0, robot=rb, log=lambda s: None,
                                 obstacles=obstacles, done=done, on_tick=tick)
        tr = _finish_trial(pj, goal, rb.env, True, t0, i, log, prof.name, n, fell=r.fell)
        tr |= {"pushes": len(r.pushes), "fell": r.fell, "stop": r.stop_reason}
        trials.append(tr)
        if record_demo and tr["success"] and not Path(record_demo).exists():
            _save_demo(record_demo, frames, prof.name, log)
    ok = sum(t["success"] for t in trials)
    log(f"{prof.name}: {ok}/{len(trials)}")
    return {"success": ok, "n": len(trials), "trials": trials}


def save_result(out_dir: Path, result: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / "result.json"
    f.write_text(json.dumps(result, indent=2, default=str))
    return f
