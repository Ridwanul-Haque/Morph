"""Held-out evaluation of source->target transfer methods on identical scenarios.

Every trial is scored by the same measured-final-pose test (task.evaluate).
Held-out seeds (>= 10_000) never appear in training.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .arm_skills import expert_action, joint_obs, run_episode, task_obs
from .learning import HELDOUT_SEEDS, Policy, load_policy, sample_scenario
from .robots.arm import ArmAdapter
from .scene import ARM_A, ARM_B
from .sim import SimulationUnstable
from .task import default_spec

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def _policy_fn(policy: Policy, kind: str):
    def f(s):
        return policy.act(task_obs(s) if kind == "task" else joint_obs(s)), "policy"
    return f


def method_list(source: str, targets: list[str], arms: dict, policies: dict) -> dict[str, str]:
    nq = lambda n: len(arms[n].home_q)
    m = {f"expert_on_{source}": "scripted expert on the source arm (reference: source outcome)"}
    if "task" in policies:
        m[f"task_policy_on_{source}"] = "task-space BC/DAgger policy on the arm it was trained on"
    if "joint" in policies:
        m[f"joint_policy_on_{source}"] = "joint-space BC/DAgger policy on the arm it was trained on"
    for t in targets:
        same = nq(t) == nq(source)
        if same:
            m[f"joint_replay_{source}_to_{t}"] = "baseline: replay the source's executed joint targets + open/close on the target"
            if "joint" in policies:
                m[f"joint_policy_{source}_to_{t}"] = "baseline: source-trained joint-space policy applied zero-shot to the target"
        if "task" in policies:
            m[f"task_policy_{source}_to_{t}"] = ("Morph learned transfer: source-trained task-space policy, executed "
                                                 "through the target's own IK adapter")
        m[f"goal_replan_{t}"] = "Morph goal-level transfer: confirmed TaskSpec + the target's own pick-place skill"
    return m


def run_method(method: str, source: str, arms: dict, policies: dict, sc, max_time: float = 14.0, render: bool = False):
    spec = default_spec(*sc.goal)
    kw = dict(max_time=max_time, render_every=3 if render else 0, record=render)
    tgt = method.split("_to_")[-1] if "_to_" in method else None
    if method.startswith("expert_on_"):
        arm = arms[source]; arm.reset(sc.seed, sc.block, sc.params)
        return run_episode(arm, spec, expert_action, **kw)
    if method.startswith("goal_replan_"):
        arm = arms[method[len("goal_replan_"):]]; arm.reset(sc.seed, sc.block, sc.params)
        return run_episode(arm, spec, expert_action, **kw)
    if method.startswith(("task_policy", "joint_policy")):
        kind = "task" if method.startswith("task") else "joint"
        arm = arms[tgt or source]; arm.reset(sc.seed, sc.block, sc.params)
        return run_episode(arm, spec, _policy_fn(policies[kind], kind), mode=kind, stop_when_done=False, **kw)
    if method.startswith("joint_replay_"):
        a = arms[source]; a.reset(sc.seed, sc.block, sc.params)
        src = run_episode(a, spec, expert_action, max_time=max_time, record=True)
        cmds = [(np.array(t["q_cmd"]), t["close"]) for t in src.trace]
        it = iter(cmds)
        b = arms[tgt]; b.reset(sc.seed, sc.block, sc.params, q0=a.home_q)  # same joint start as the source
        return _run_ctrl_sequence(b, spec, lambda s: next(it, cmds[-1]), len(cmds), **kw)
    raise KeyError(method)


def _run_ctrl_sequence(arm, spec, replay, n: int, max_time: float, render_every: int = 0, record: bool = False):
    """Open-loop replay of joint targets and gripper open/close (no feedback, no IK)."""
    from .arm_skills import EpisodeResult, read_state, released
    from .task import evaluate
    arm.env.set_goal_marker(spec.gx, spec.gy, spec.gyaw)
    res = EpisodeResult(outcome=None, sim_time=0.0)  # type: ignore[arg-type]
    for t in range(int(max_time / arm.env.control_dt)):
        s = read_state(arm, spec)
        q, close = replay(s)
        log = arm.env.advance(arm.ctrl(np.clip(q, arm.q_lo, arm.q_hi), close))
        res.max_step_disp = max(res.max_step_disp, log.max_step_disp)
        if log.robot_contacts:
            c = max(log.robot_contacts, key=lambda c: c.normal_force)
            res.contacts.append({"t": round(c.time, 3), "geom": c.geom, "force_n": round(c.normal_force, 3)})
        if record:
            res.trace.append({"t": round(float(arm.env.data.time), 3), "phase": "replay", "q_cmd": [round(float(v), 5) for v in q],
                              "close": bool(close), "tcp": [round(float(v), 5) for v in s.tcp], "block": arm.env.block_pose().as_dict()})
        if render_every and t % render_every == 0:
            res.frames.append(arm.env.render("overview"))
    res.sim_time = float(arm.env.data.time)
    res.outcome = evaluate(spec, arm.env.block_pose(), arm.env.block_speed(), released(arm))
    return res


def evaluate_transfer(n: int = 50, methods: list[str] | None = None, policy_dir: Path = Path("artifacts/policies"),
                      out: Path | None = None, log=print, source: str = "arm_A", targets: list[str] | None = None) -> dict:
    from .robots.registry import make_arm
    targets = targets or (["arm_B"] if source == "arm_A" else ["ufactory_xarm7", "arm_B"])
    arms = {nm: make_arm(nm) for nm in {source, *targets}}
    policies = {}
    for k in ("task", "joint"):
        p = policy_dir / source / f"policy_{k}.pt"
        if p.exists():
            policies[k] = load_policy(p)
    all_methods = method_list(source, targets, arms, policies)
    methods = methods or list(all_methods)
    seeds = list(HELDOUT_SEEDS)[:n]
    results = {"source": source, "targets": targets, "n": n, "seeds": [seeds[0], seeds[-1]], "methods": {},
               "tolerances": {"pos_m": 0.05, "yaw_deg": 15},
               "perturbations": "block mass U(0.1,0.4) kg, block-floor friction U(0.4,0.8), random start/goal poses"}
    for m in methods:
        trials, t0 = [], time.time()
        for sd in seeds:
            sc = sample_scenario(sd)
            try:
                r = run_method(m, source, arms, policies, sc)
                o = r.outcome
                trials.append({"seed": sd, "success": o.success, "pos_err_cm": round(o.pos_err_m * 100, 2),
                               "yaw_err_deg": round(o.yaw_err_deg, 2), "tilt_deg": round(math.degrees(o.tilt_rad), 2),
                               "reasons": o.reasons, "block_start": list(sc.block), "goal": list(sc.goal),
                               "final": o.final_pose, "params": sc.params.as_dict(),
                               "contact_ticks": len(r.contacts), "ik_failures": r.ik_failures,
                               "sim_time_s": round(r.sim_time, 2)})
            except SimulationUnstable as e:
                trials.append({"seed": sd, "success": False, "unstable": True, "reasons": [str(e)]})
        k = sum(t["success"] for t in trials)
        errs = [t["pos_err_cm"] for t in trials if "pos_err_cm" in t]
        yerrs = [t["yaw_err_deg"] for t in trials if "yaw_err_deg" in t]
        results["methods"][m] = {
            "description": all_methods[m], "success": k, "n": len(trials), "rate": round(k / len(trials), 3),
            "wilson95": wilson(k, len(trials)), "median_pos_err_cm": round(float(np.median(errs)), 2) if errs else None,
            "median_yaw_err_deg": round(float(np.median(yerrs)), 2) if yerrs else None,
            "unstable": sum(1 for t in trials if t.get("unstable")), "wall_s": round(time.time() - t0, 1),
            "trials": trials}
        log(f"{m:44s} {k:3d}/{len(trials)}  95% CI {wilson(k, len(trials))}  median pos err "
            f"{results['methods'][m]['median_pos_err_cm']} cm  ({time.time() - t0:.0f}s)")
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"transfer_eval_{source}.json").write_text(json.dumps(results, indent=2))
        (out / f"transfer_eval_{source}.md").write_text(to_markdown(results))
    return results


def to_markdown(res: dict) -> str:
    lines = [f"Held-out seeds {res['seeds'][0]}..{res['seeds'][1]} (n={res['n']}); {res['perturbations']}.",
             f"Success = final block within {res['tolerances']['pos_m'] * 100:.0f} cm and {res['tolerances']['yaw_deg']} deg, "
             "upright, at rest, released.", "",
             "| method | success | rate | 95% CI (Wilson) | median pos err (cm) | median yaw err (deg) | unstable |",
             "|---|---|---|---|---|---|---|"]
    for m, r in res["methods"].items():
        lines.append(f"| `{m}` | {r['success']}/{r['n']} | {r['rate']:.2f} | {r['wilson95'][0]:.2f}-{r['wilson95'][1]:.2f} | "
                     f"{r['median_pos_err_cm']} | {r['median_yaw_err_deg']} | {r['unstable']} |")
    lines.append("")
    for m, r in res["methods"].items():
        lines.append(f"- `{m}`: {r['description']}")
    return "\n".join(lines) + "\n"


def evaluate_pusher(n: int = 30, modes=("mpc", "fixed_direction", "open_loop"), out: Path | None = None, log=print) -> dict:
    """Held-out evaluation of the mobile pusher on the same scenarios as the arms."""
    from .pusher_task import run_pusher
    seeds = list(HELDOUT_SEEDS)[:n]
    res = {"n": n, "seeds": [seeds[0], seeds[-1]], "methods": {}, "tolerances": {"pos_m": 0.05, "yaw_deg": 15},
           "perturbations": "block mass U(0.1,0.4) kg, block-floor friction U(0.4,0.8) (planner assumes 0.2 kg / 0.6)"}
    for mode in modes:
        trials, t0 = [], time.time()
        for sd in seeds:
            sc = sample_scenario(sd)
            try:
                r = run_pusher(default_spec(*sc.goal), sc.block, sc.params, sd, mode=mode, log=lambda s: None)
                o = r.outcome
                trials.append({"seed": sd, "success": o.success, "pos_err_cm": round(o.pos_err_m * 100, 2),
                               "yaw_err_deg": round(o.yaw_err_deg, 2), "reasons": o.reasons, "pushes": len(r.pushes),
                               "tip_contact_ticks": sum(p["contact"]["tip_contact_ticks"] for p in r.pushes),
                               "other_contacts": r.other_contacts, "sim_time_s": round(r.sim_time, 1),
                               "stop": r.stop_reason, "block_start": list(sc.block), "goal": list(sc.goal),
                               "params": sc.params.as_dict(), "final": o.final_pose})
            except SimulationUnstable as e:
                trials.append({"seed": sd, "success": False, "unstable": True, "reasons": [str(e)]})
        k = sum(t["success"] for t in trials)
        errs = [t["pos_err_cm"] for t in trials if "pos_err_cm" in t]
        yerrs = [t["yaw_err_deg"] for t in trials if "yaw_err_deg" in t]
        res["methods"][f"pusher_{mode}"] = {
            "description": {"mpc": "Morph: closed-loop model-predictive pushing (re-plan after each measured push)",
                            "fixed_direction": "baseline: push toward goal through the face centre, position only",
                            "open_loop": "baseline: whole push sequence planned once on the nominal model, not re-measured"}[mode],
            "success": k, "n": len(trials), "rate": round(k / len(trials), 3), "wilson95": wilson(k, len(trials)),
            "median_pos_err_cm": round(float(np.median(errs)), 2) if errs else None,
            "median_yaw_err_deg": round(float(np.median(yerrs)), 2) if yerrs else None,
            "median_pushes": float(np.median([t.get("pushes", 0) for t in trials])),
            "unstable": sum(1 for t in trials if t.get("unstable")), "wall_s": round(time.time() - t0, 1), "trials": trials}
        m = res["methods"][f"pusher_{mode}"]
        log(f"pusher_{mode:16s} {k:3d}/{len(trials)}  95% CI {m['wilson95']}  median pos err {m['median_pos_err_cm']} cm  "
            f"yaw {m['median_yaw_err_deg']} deg  pushes {m['median_pushes']}  ({m['wall_s']}s)")
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "pusher_eval.json").write_text(json.dumps(res, indent=2))
        (out / "pusher_eval.md").write_text(to_markdown(res))
    return res


def evaluate_go1(n: int = 8, out: Path | None = None, log=print) -> dict:
    """Held-out evaluation of the crawl-gait Go1 baseline (same scenarios as the arms)."""
    from .go1_task import push_pose, run_go1_push
    from .push_planner import PushModel, plan_push
    seeds = list(HELDOUT_SEEDS)[:n]
    trials, t0 = [], time.time()
    pm = PushModel()
    for sd in seeds:
        sc = sample_scenario(sd)
        spec = default_spec(*sc.goal)
        heading = math.atan2(sc.goal[1] - sc.block[1], sc.goal[0] - sc.block[0])
        first, _ = plan_push(pm, sc.block, sc.goal, robot_pose=(0, 0, heading), max_heading_change=math.radians(40))
        b0 = push_pose(first.contact_xy, first.direction)
        b0 = (b0[0] - 0.05 * math.cos(b0[2]), b0[1] - 0.05 * math.sin(b0[2]), b0[2])
        r = run_go1_push(spec, sc.block, sc.params, sd, base0=b0, log=lambda s: None)
        o = r.outcome
        start_err = math.hypot(sc.block[0] - sc.goal[0], sc.block[1] - sc.goal[1]) * 100
        trials.append({"seed": sd, "success": o.success, "pos_err_cm": round(o.pos_err_m * 100, 2),
                       "start_pos_err_cm": round(start_err, 2), "yaw_err_deg": round(o.yaw_err_deg, 2),
                       "reasons": o.reasons, "pushes": len(r.pushes), "fell": r.fell, "stop": r.stop_reason,
                       "foot_contact_ticks": sum(p["contact"]["foot_contact_ticks"] for p in r.pushes),
                       "non_foot_block_contacts": r.unintended_contacts, "sim_time_s": round(r.sim_time, 1)})
        log(f"go1 seed {sd}: success={o.success} pushes={len(r.pushes)} fell={r.fell} err {start_err:.1f}->{o.pos_err_m * 100:.1f} cm "
            f"stop='{r.stop_reason}'")
    k = sum(t["success"] for t in trials)
    res = {"n": n, "seeds": [seeds[0], seeds[-1]], "tolerances": {"pos_m": 0.05, "yaw_deg": 15},
           "perturbations": "block mass U(0.1,0.4) kg, friction U(0.4,0.8)",
           "methods": {"go1_foot_push_mpc": {
               "description": "Go1 hand-written crawl + front-foot push, re-planned after each push",
               "success": k, "n": n, "rate": round(k / n, 3), "wilson95": wilson(k, n),
               "median_pos_err_cm": round(float(np.median([t["pos_err_cm"] for t in trials])), 2),
               "median_yaw_err_deg": round(float(np.median([t["yaw_err_deg"] for t in trials])), 2),
               "falls": sum(t["fell"] for t in trials), "median_pushes": float(np.median([t["pushes"] for t in trials])),
               "unstable": 0, "wall_s": round(time.time() - t0, 1), "trials": trials}}}
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "go1_eval.json").write_text(json.dumps(res, indent=2))
        (out / "go1_eval.md").write_text(to_markdown(res))
    return res


def evaluate_go1_learned(n: int = 8, out: Path | None = None, log=print) -> dict:
    """Held-out evaluation of Go1 with learned walking + three-leg foot push (same scenarios as the arms)."""
    from .go1_learned_task import run_learned_go1_push
    from .robots.go1_learned import LearnedGo1
    seeds = list(HELDOUT_SEEDS)[:n]
    robot = LearnedGo1()
    trials, t0 = [], time.time()
    for sd in seeds:
        sc = sample_scenario(sd)
        spec = default_spec(*sc.goal)
        start = math.hypot(sc.block[0] - sc.goal[0], sc.block[1] - sc.goal[1]) * 100
        base0 = (sc.block[0], sc.block[1] - 0.7, math.pi / 2)  # start behind the block, facing it
        r = run_learned_go1_push(spec, sc.block, sc.params, sd, base0=base0, robot=robot, log=lambda s: None)
        o = r.outcome
        trials.append({"seed": sd, "success": o.success, "pos_err_cm": round(o.pos_err_m * 100, 2), "start_pos_err_cm": round(start, 2),
                       "yaw_err_deg": round(o.yaw_err_deg, 2), "reasons": o.reasons, "pushes": len(r.pushes), "fell": r.fell,
                       "stop": r.stop_reason, "foot_contact_ticks": sum(p["contact"]["foot_contact_ticks"] for p in r.pushes),
                       "non_foot_block_contacts": r.unintended_contacts, "sim_time_s": round(r.sim_time, 1),
                       "unstable": r.unstable, "median_walk_base_error_cm": float(np.median([p["base_error_cm"] for p in r.pushes])) if r.pushes else None})
        log(f"go1-learned seed {sd}: success={o.success} pushes={len(r.pushes)} fell={r.fell} err {start:.1f}->{o.pos_err_m * 100:.1f} cm "
            f"yaw {o.yaw_err_deg:.0f} deg  stop='{r.stop_reason}'")
    k = sum(t["success"] for t in trials)
    res = {"n": n, "seeds": [seeds[0], seeds[-1]], "tolerances": {"pos_m": 0.05, "yaw_deg": 15},
           "perturbations": "block mass U(0.1,0.4) kg, friction U(0.4,0.8)",
           "methods": {"go1_learned_walk_foot_push": {
               "description": "Go1: RL-trained walking policy (MuJoCo Playground, trained locally) to reposition + three-leg foot push",
               "success": k, "n": n, "rate": round(k / n, 3), "wilson95": wilson(k, n),
               "median_pos_err_cm": round(float(np.median([t["pos_err_cm"] for t in trials])), 2),
               "median_yaw_err_deg": round(float(np.median([t["yaw_err_deg"] for t in trials])), 2),
               "falls": sum(t["fell"] for t in trials), "median_pushes": float(np.median([t["pushes"] for t in trials])),
               "unstable": sum(t["unstable"] for t in trials), "wall_s": round(time.time() - t0, 1), "trials": trials}}}
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "go1_learned_eval.json").write_text(json.dumps(res, indent=2))
        (out / "go1_learned_eval.md").write_text(to_markdown(res))
    return res
