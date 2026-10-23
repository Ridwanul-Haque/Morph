"""Planning how to make contact: several candidate ways to grip, push or tip the object, each tried in simulation on the
robot's own model (its real fingers, reach and joint limits), and the best kept (grasp planning, as GraspIt! and Dex-Net
do it, here by testing rather than by analysis).

  plan_grasp(rn, scene)        for a gripping part (a parallel gripper, a single jaw): which sides, where along the
                               object, from above or from the side, how high. Each candidate: grip, lift 5 cm, shake;
                               scored on still holding it from two slightly different starts.
  plan_push(rn, scene, d, aim) for any part that can push (a tool flange, a fingertip): which face, how high, how far
                               off-centre, the part pointing down or at the object. Each candidate: push 6 cm towards d;
                               scored on sliding straight along d (aim "slide") or on tipping it over (aim "tip").

Candidates are written as ordinary body-program steps whose numbers are formulas in the object's own frame (cosd / sind
of obj_yaw_deg), so the winner follows the object wherever it is when it runs: the `pick`, `push_object` and `tip_over`
building blocks expand into it.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .body_programs import BodyRunner, check_body_program, run_body_program
from .task_model import contact_held


def _xy(u: float, v: float) -> tuple[str, str]:
    """A point (u, v) metres in the object's frame, as world-frame formulas."""
    return (f"obj_x + ({u:.4f}) * cosd(obj_yaw_deg) - ({v:.4f}) * sind(obj_yaw_deg)",
            f"obj_y + ({u:.4f}) * sind(obj_yaw_deg) + ({v:.4f}) * cosd(obj_yaw_deg)")


# ------------------------------------------------------------------ grasps
def grasp_candidates(rn: BodyRunner, scene, part: str) -> list[dict]:
    b = scene.block
    hx, hy, h = b.half_x, b.half_y, 2 * b.half_z
    radius = 0.04
    cands = []
    # from above: the fingers closing across the object's x or y, at its centre or off-centre along its long side, the
    # fingertips just above the floor (a flat thing), a third of the way up, or two thirds (a tall thing)
    long_ax = 0 if hx >= hy else 1
    offsets = [0.0] + ([0.4, -0.4] if max(hx, hy) > 0.03 else [])
    heights = sorted({round(v, 4) for v in (0.003, 0.35 * h, 0.6 * h) if v < h - 0.004} | {0.003})
    for close_deg in (0, 90):
        for off in offsets:
            u, v = (off * hx, 0.0) if long_ax == 0 else (0.0, off * hy)
            x, y = _xy(u, v)
            for z in heights:
                cands.append({"desc": f"from above, fingers closing along the object's {'x' if close_deg == 0 else 'y'}, "
                                      f"{'at the centre' if off == 0 else f'{abs(off):.0%} towards one end'}, fingertips "
                                      f"{z * 100:.1f} cm up",
                              "steps": [{"do": "grip", "how": "open"},
                                        {"do": "move", "part": part, "x": x, "y": y, "z": "obj_top + 0.06", "point": "down",
                                         "turn": f"obj_yaw_deg + {close_deg}"},
                                        {"do": "move", "part": part, "x": x, "y": y, "z": f"obj_bottom + {z:.4f}",
                                         "point": "down", "speed": "slow"},
                                        {"do": "grip", "how": "close"},
                                        {"do": "move", "part": part, "z": "obj_top + 0.05", "speed": "slow"}]})
    # from the side: the hand comes in level from one of the four sides, fingers closing horizontally across it (or
    # tilted as its wrist holds it, if it can't point level: the Stretch's gripper always points 30 degrees down)
    pz = level_tilt(rn, rn.parts[part])
    c = math.sqrt(max(1.0 - pz * pz, 0.0))
    for side_deg in (0, 90, 180, 270):
        ext = abs(hx * math.cos(math.radians(side_deg))) + abs(hy * math.sin(math.radians(side_deg)))
        for frac in (0.5, 0.35):
            z = f"obj_bottom + {frac * h:.4f}"
            d = f"(obj_yaw_deg + {side_deg})"
            back = ext + radius + 0.04
            cands.append({"desc": f"from the side ({side_deg}° in the object's frame), {frac:.0%} of its height up",
                          "steps": [{"do": "grip", "how": "open"},
                                    {"do": "move", "part": part, "x": f"obj_x - {back:.4f} * cosd{d}",
                                     "y": f"obj_y - {back:.4f} * sind{d}", "z": "obj_top + 0.06",
                                     "point": [f"{c:.4f} * cosd{d}", f"{c:.4f} * sind{d}", f"{pz:.4f}"],
                                     "turn": f"{d} + 90"},
                                    {"do": "move", "part": part, "z": z, "speed": "slow"},
                                    {"do": "move", "part": part, "x": f"obj_x + {0.3 * ext:.4f} * cosd{d}",
                                     "y": f"obj_y + {0.3 * ext:.4f} * sind{d}", "speed": "slow"},
                                    {"do": "grip", "how": "close"},
                                    {"do": "move", "part": part, "z": "obj_top + 0.05", "speed": "slow"}]})
    return cands


def level_tilt(rn: BodyRunner, part) -> float:
    """How far down (the z of its pointing) a hand points when asked to point level: 0 if it can, else as its wrist
    holds it."""
    a = rn.pointing(part)
    h = np.array([a[0], a[1], 0.0])
    if np.linalg.norm(h) < 1e-6:
        h = np.array([1.0, 0.0, 0.0])
    _, _, ae = rn.ik(part, rn.tip(part), point=h / np.linalg.norm(h), iters=120)
    return 0.0 if ae < 0.1 else float(np.clip(a[2], -0.95, 0.95))


SHAKE = [{"do": "move", "part": "{part}", "x": "{part}_x + 0.03", "speed": "fast"},
         {"do": "move", "part": "{part}", "x": "{part}_x - 0.06", "speed": "fast"},
         {"do": "move", "part": "{part}", "x": "{part}_x + 0.03", "speed": "fast"},
         {"do": "wait", "seconds": 0.3}]


GRASP_CACHE = Path(__file__).resolve().parents[1] / "runs" / "programs" / "grasps"


def _grasp_key(rn: BodyRunner, scene, part: str, starts: int) -> Path:
    """The same robot, mounted the same way, gripping the same object from the same start: the same plan."""
    t = rn.task
    key = json.dumps([str(t.get("robot_xml")), [round(float(v), 3) for v in t.get("mount", ())], t.get("block"),
                      [round(float(v), 4) for v in scene.start], part, starts,
                      (t.get("gripper") or {}).get("gain_scale")], sort_keys=True, default=str)
    return GRASP_CACHE / f"{t.get('robot', 'robot')}__{hashlib.sha1(key.encode()).hexdigest()[:12]}.json"


def grasp_geometry(rn: BodyRunner, part: str) -> dict:
    """How the hand holds the object right now, in the object's own frame: where the point between its fingers is,
    which way its fingers close, and which way the hand points (the way it came in). With the bodies and offsets that
    carry them, so the learner can measure the same on its own copy of the robot (train_task: the planned-grasp aim)."""
    p = rn.parts[part]
    R = rn.d.xmat[rn.ix.block].reshape(3, 3)
    bp = rn.d.xpos[rn.ix.block]
    close = rn.finger_axis(p)
    out = {"part": part, "point": (R.T @ (rn.tip(p) - bp)).round(4).tolist(),
           "approach": (R.T @ rn.pointing(p)).round(4).tolist(),
           "close": (R.T @ close).round(4).tolist() if close is not None else None,
           "body": rn.m.body(p.body).name, "offset": np.round(p.offset, 5).tolist(), "axis": np.round(p.axis, 5).tolist()}
    if p.fingers:
        out["fingers"] = [[rn.m.body(int(b)).name, np.round(o, 5).tolist()] for b, o in p.fingers]
    return out


def plan_grasp(rn: BodyRunner, scene, part: str | None = None, starts: int = 2, log=lambda s: None) -> dict | None:
    """The grasp that holds best for this gripper and object: {desc, steps, score, tried, held, geom}. None if it has
    no gripping part. score: per start, 2 for still holding it after the shake (1 if only after the lift). geom: the
    winning grip measured (grasp_geometry), for the learner to aim at. Kept on disk: the same robot, mount and object
    get the same plan without trying the candidates again."""
    part = part or next((n for n, p in rn.parts.items() if p.can_grip), None)
    if part is None:
        return None
    cache = _grasp_key(rn, scene, part, starts)
    if cache.exists():
        try:
            g = json.loads(cache.read_text())
            log(f"  grasp planning: {g['desc']} ({g['score']:.0f}/{g['max']:.0f}; planned before, {g['tried']} ways tried)")
            return g
        except (ValueError, KeyError):
            pass
    g = _plan_grasp(rn, scene, part, starts, log)
    if g is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(g, indent=1))
    return g


def _plan_grasp(rn: BodyRunner, scene, part: str, starts: int, log) -> dict | None:
    from .goals import GoalSpec
    from .project import _params, _start_poses
    shake = [{k: (v.replace("{part}", part) if isinstance(v, str) else v) for k, v in st.items()} for st in SHAKE]
    empty = GoalSpec(object=scene.object_name, conditions=[])
    results = []
    for c in grasp_candidates(rn, scene, part):
        try:
            prog = check_body_program(c["steps"], rn.parts, [])
            test = check_body_program(c["steps"] + shake, rn.parts, [])
        except Exception:  # noqa: BLE001  (a candidate this body can't even express)
            continue
        score, notes = 0.0, []
        for i, st in enumerate(_start_poses(scene, starts)):
            rn.reset(i, st, _params(i))
            run = run_body_program(rn, test, empty, scene, settle_s=0.1)
            held = contact_held(rn.m, rn.d, rn.ix) > 0.5
            lifted = rn.env.block_pose().z > scene.block.half_z + 0.02
            score += 2.0 if held else (1.0 if lifted else 0.0)
            notes.append(run["report"][3][:120] if len(run["report"]) > 3 else "")
        results.append({"desc": c["desc"], "steps": [dict(s) for s in c["steps"]], "score": score, "prog": prog})
    rn.reset(0, scene.start, None)
    if not results:
        return None
    best = max(results, key=lambda r: r["score"])
    top = 2.0 * starts
    geom = None
    if best["score"] > 0:  # the winner once more from the scene's start, measured where it holds the object
        rn.reset(0, scene.start, _params(0))
        run_body_program(rn, best["prog"], empty, scene, settle_s=0.1)
        if contact_held(rn.m, rn.d, rn.ix) > 0.5:
            geom = grasp_geometry(rn, part)
        rn.reset(0, scene.start, None)
    log(f"  grasp planning: tried {len(results)} ways to grip the {scene.object_name}; best: {best['desc']} "
        f"({best['score']:.0f}/{top:.0f})")
    return {"part": part, "desc": best["desc"], "steps": best["steps"], "score": best["score"], "max": top,
            "tried": len(results), "held": best["score"] >= top - 1e-9, "geom": geom,
            "ranking": [(r["desc"], r["score"]) for r in sorted(results, key=lambda r: -r["score"])[:5]]}


# ------------------------------------------------------------------ pushes and tips
def push_candidates(rn: BodyRunner, scene, part: str, direction_deg: float, aim: str) -> list[dict]:
    """Ways to push the object 6 cm towards direction_deg (world): which height on its face, how far off-centre, the
    part pointing down or at the object. The face is the one facing away from the direction."""
    b = scene.block
    h = 2 * b.half_z
    from .body_programs import part_radius
    r = part_radius(rn, rn.parts[part])
    heights = (0.2, 0.5) if aim == "slide" else (0.85, 0.7, 0.5)
    cands = []
    d = f"{direction_deg:.2f}"
    # how far the object's face is from its centre along the direction (its half-size that way)
    ext = f"(abs({b.half_x:.4f} * cosd({d} - obj_yaw_deg)) + abs({b.half_y:.4f} * sind({d} - obj_yaw_deg)))"
    speeds = ("slow",) if aim == "slide" else ("slow", "fast")  # a quick strike near the top tips what a push slides
    for frac in heights:
      for speed in speeds:  # noqa: E111
        for lat in (0.0, 0.5, -0.5):
            for point in ("down", "at"):
                side = min(b.half_x, b.half_y) * lat
                back = f"({ext} + {r:.4f} + 0.03)"
                touch = f"({ext} + {r:.4f} - 0.004)"
                px = lambda dist: f"obj_x - {dist} * cosd({d}) - ({side:.4f}) * sind({d})"  # noqa: E731
                py = lambda dist: f"obj_y - {dist} * sind({d}) + ({side:.4f}) * cosd({d})"  # noqa: E731
                pt = "down" if point == "down" else [f"cosd({d})", f"sind({d})", "0"]
                z = f"obj_bottom + {frac * h:.4f}"
                cands.append({"desc": f"{'low' if frac < 0.4 else 'mid' if frac < 0.65 else 'high'} on its face "
                                      f"({frac:.0%} up), {'centred' if lat == 0 else 'off-centre'}, the {part} pointing "
                                      f"{'down' if point == 'down' else 'at it'}" + (", a quick strike" if speed == "fast" else ""),
                              "steps": [{"do": "move", "part": part, "x": px(back), "y": py(back), "z": "obj_top + 0.05",
                                         "point": pt},
                                        {"do": "move", "part": part, "z": z, "speed": "slow"},
                                        {"do": "move", "part": part, "x": px(touch), "y": py(touch), "speed": "slow",
                                         "until": "touch"},
                                        {"do": "push", "part": part, "direction": float(direction_deg),
                                         "distance": 0.06 if speed == "slow" else 0.1, "speed": speed},
                                        {"do": "move", "part": part, "z": "obj_top + 0.06"}]})
    if aim == "tip":  # a wide flat tool (a flange) can't reach the side high up: catch the top edge with its rim instead
        for overlap in (0.01, 0.02):
            for speed in ("slow", "fast"):
                back = f"({ext} + {r:.4f} - {overlap})"
                cands.append({"desc": f"catching the top edge with the rim of the {part} ({overlap * 100:.0f} cm over it)"
                                      + (", a quick strike" if speed == "fast" else ""),
                              "steps": [{"do": "move", "part": part, "x": f"obj_x - {back} * cosd({d})",
                                         "y": f"obj_y - {back} * sind({d})", "z": "obj_top + 0.04", "point": "down"},
                                        {"do": "move", "part": part, "z": "obj_top - 0.008", "speed": "slow"},
                                        {"do": "push", "part": part, "direction": float(direction_deg), "distance": 0.1,
                                         "speed": speed},
                                        {"do": "move", "part": part, "z": "obj_top + 0.06"}]})
    return cands


def plan_push(rn: BodyRunner, scene, part: str, direction_deg: float, aim: str = "slide", starts: int = 2,
              log=lambda s: None) -> dict | None:
    """The contact that best slides the object straight towards direction_deg (aim "slide"), or best tips it over
    (aim "tip"): {desc, steps, score}."""
    from .goals import GoalSpec
    from .project import _params, _start_poses
    empty = GoalSpec(object=scene.object_name, conditions=[])
    u = np.array([math.cos(math.radians(direction_deg)), math.sin(math.radians(direction_deg))])
    results = []
    for c in push_candidates(rn, scene, part, direction_deg, aim):
        try:
            prog = check_body_program(c["steps"], rn.parts, [])
        except Exception:  # noqa: BLE001
            continue
        score, moved = 0.0, 0.0
        for i, st in enumerate(_start_poses(scene, starts)):
            rn.reset(i, st, _params(i))
            p0 = rn.env.block_pose()
            run_body_program(rn, prog, empty, scene, settle_s=0.5)
            p1 = rn.env.block_pose()
            dv = np.array([p1.x - p0.x, p1.y - p0.y])
            along, sideways = float(dv @ u), float(abs(dv[0] * u[1] - dv[1] * u[0]))
            turned = abs(math.degrees(math.remainder(p1.yaw - p0.yaw, math.tau)))
            moved += along / starts
            if aim == "slide":  # straight along the direction, not sideways, not turned, not tipped
                score += along * 100 - 2 * sideways * 100 - 0.1 * turned - (20 if math.degrees(p1.tilt) > 20 else 0)
            else:  # over onto its side
                score += min(math.degrees(p1.tilt), 90) / 10 + (0 if math.degrees(p1.tilt) > 45 else -5)
        results.append({"desc": c["desc"], "steps": c["steps"], "score": score / starts, "moved": moved})
    rn.reset(0, scene.start, None)
    if not results:
        return None
    best = max(results, key=lambda r: r["score"])
    log(f"  contact planning ({aim}): tried {len(results)} ways to push the {scene.object_name}; best: {best['desc']} "
        f"(score {best['score']:.1f})")
    return {"part": part, "aim": aim, "direction": direction_deg, "desc": best["desc"], "steps": best["steps"],
            "score": best["score"], "tried": len(results), "moved_per_push": max(best["moved"], 0.01),
            "ranking": [(r["desc"], round(r["score"], 1)) for r in sorted(results, key=lambda r: -r["score"])[:5]]}
