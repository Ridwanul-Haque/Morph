"""End-to-end transfer run: source demo -> TaskSpec -> target robot. Output goes to runs/<id>/."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .capability import assess
from .learning import load_policy, sample_scenario
from .orchestrator import (RunContext, arm_strategy_options, execute_arm_strategy, infer, rule_select)
from .reasoning import ReasoningError, ReasoningService, ReasoningUnavailable
from .recorder import record_demo
from .robots.arm import ArmAdapter
from .sim import PhysicalParams, SimulationUnstable
from .task import TaskSpec, evaluate
from .viewer import annotate_episode, side_by_side

EVIDENCE_DIR = Path("artifacts/evidence")
POLICY_DIR = Path("artifacts/policies")
ARM_TARGETS = ("arm_B", "ufactory_xarm7", "franka_panda", "arm_A")

REVISION_SPACE = {"strategy_id": {"choices": ["goal_replan", "learned_task_policy", "learned_joint_policy", "joint_replay"]},
                  "time_limit_s": {"range": [8.0, 20.0]}}


def confirm_spec(spec: TaskSpec, ctx: RunContext, edits: dict, interactive: bool) -> TaskSpec:
    """User confirmation step. Edits (CLI flags or prompts) are recorded with provenance 'user'."""
    s = spec.model_copy(deep=True)
    if interactive:
        print("\n--- Proposed TaskSpec (edit or press Enter to accept) ---")
        for key, label, conv, back in [
            ("x", "goal x [m]", float, float), ("y", "goal y [m]", float, float),
            ("yaw", "goal yaw [deg, 90=north]", lambda v: math.radians(float(v)), math.degrees),
            ("pos_tol_m", "position tolerance [cm]", lambda v: float(v) / 100, lambda v: v * 100),
            ("yaw_tol_rad", "yaw tolerance [deg]", lambda v: math.radians(float(v)), math.degrees)]:
            f = getattr(s.goal, key)
            ans = input(f"{label} = {back(float(f.value)):.3f} ({f.provenance}) > ").strip()
            if ans:
                edits[key] = conv(ans)
    for key, val in edits.items():
        f = getattr(s.goal, key)
        f.value = val
        f.provenance = "user"
        f.note = "edited at confirmation"
    s.status = "confirmed"
    ctx.enter("confirm", edits=list(edits) or "none", interactive=interactive)
    return s


def run_arm_transfer(run_dir: Path, instruction: str, demo_block=(0.46, -0.20, 0.4),
                     demo_goal=(0.45, 0.20, math.pi / 2), demo_seed: int = 11, target_seed: int = 12,
                     use_llm: bool = True, edits: dict | None = None, interactive: bool = False,
                     variant: str | None = None, verify_n: int = 10, render: bool = True, log=print,
                     target: str = "ufactory_xarm7", source: str = "franka_panda") -> dict:
    t_start = time.time()
    run_dir.mkdir(parents=True, exist_ok=True)
    llm = ReasoningService(log_path=run_dir / "inference.jsonl") if use_llm else None
    ctx = RunContext(run_dir, llm, log)
    compliance = "live Nemotron via Nebius Token Factory" if (llm and llm.available) else \
        "OFFLINE BASELINE (no Nebius call) - NOT hackathon compliant"
    ctx.enter("start", mode=compliance)

    from .robots.registry import make_arm
    arm_a = make_arm(source)
    # 1. demonstrate (executed, recorded)
    demo = record_demo(arm_a, demo_seed, demo_block, demo_goal, instruction, render=render)
    ctx.save("source/summary.json", demo.summary())
    demo.save(run_dir / "source")
    ctx.enter("demonstrate", robot=source, reached_intended=demo.success_vs_intended,
              observed=demo.summary()["observed_change"])

    # 2. infer
    draft, info = infer(ctx, demo, instruction)
    ctx.save("taskspec_draft.json", draft.model_dump())
    ctx.enter("infer", **info)

    # 3. confirm
    spec = confirm_spec(draft, ctx, dict(edits or {}), interactive)
    if variant == "face_up_flip":
        spec.goal.orientation_mode.value, spec.goal.orientation_mode.provenance = "face_up_flip", "user"
        spec.goal.target_top_face.value, spec.goal.target_top_face.provenance = "+x", "user"
    ctx.save("taskspec_confirmed.json", spec.model_dump())

    # target initial condition: comparable but not identical to the source's
    rng = np.random.default_rng(target_seed)
    block0 = (demo_block[0] + rng.uniform(-0.04, 0.04), demo_block[1] + rng.uniform(-0.04, 0.04),
              demo_block[2] + rng.uniform(-0.8, 0.8))
    params = PhysicalParams(block_mass=float(rng.uniform(0.15, 0.3)), block_friction=float(rng.uniform(0.45, 0.75)))

    if target not in ARM_TARGETS:
        return _run_push_target(ctx, llm, target, spec, demo, block0, params, target_seed, info, compliance,
                                verify_n, render, t_start)

    # 4. assess
    arm_b = make_arm(target)
    caps = arm_b.capabilities()
    cap = assess(spec, caps, block0)
    ctx.save("capability.json", {"target": caps.as_dict(), "report": cap.as_dict()})
    ctx.enter("assess", verdict=cap.verdict)
    result = {"run_dir": str(run_dir), "compliance": compliance, "capability": cap.verdict, "target": caps.robot,
              "demo": demo.summary(),
              "taskspec": spec.model_dump(), "inference": info}
    if cap.verdict == "unsupported":
        result |= {"status": "unsupported", "reasons": [c.evidence for c in cap.checks if c.status == "unsupported"]}
        return _finish(ctx, result, llm, t_start)

    # 5. propose
    ev_path = EVIDENCE_DIR / f"transfer_eval_{source}.json"
    evidence = json.loads(ev_path.read_text()) if ev_path.exists() else {}
    same_nq = len(arm_a.home_q) == len(arm_b.home_q)
    options = arm_strategy_options(evidence, source, target, same_nq)
    policies = {k: load_policy(POLICY_DIR / source / f"policy_{k}.pt") for k in ("task", "joint")
                if (POLICY_DIR / source / f"policy_{k}.pt").exists()}
    options = [o for o in options if not o["id"].startswith("learned_") or o["id"].split("_")[1] in policies]
    choice_info = {"method": "rule_highest_heldout_success", "llm_used": False}
    strategy = rule_select(options)
    if llm and llm.available:
        try:
            ch, rec = llm.select_strategy(_task_brief(spec), caps.as_dict() | {"assessment": cap.as_dict()}, options)
            strategy = ch.strategy_id
            choice_info = {"method": "nemotron", "llm_used": True, "rationale": ch.rationale,
                           "evidence_cited": ch.evidence_cited, "rule_would_choose": rule_select(options)}
        except (ReasoningError, ReasoningUnavailable) as e:
            choice_info["llm_error"] = str(e)[:300]
    ctx.save("strategy.json", {"options": options, "chosen": strategy, **choice_info})
    ctx.enter("propose", strategy=strategy, method=choice_info["method"])

    # 6-8. simulate / inspect / revise (bounded)
    source_ctrls = [(np.array(t["q_cmd"]), t["close"]) for t in demo.trace]
    attempts = []
    tried = []
    for attempt in range(3):
        tried.append(strategy)
        try:
            res = execute_arm_strategy(strategy, arm_b, spec, block0, params, target_seed, source_ctrls, policies, render)
            out = res.outcome
        except SimulationUnstable as e:
            res, out = None, None
            attempts.append({"strategy": strategy, "success": False, "unstable": str(e)})
        if out is not None:
            attempts.append({"strategy": strategy, "success": out.success, "reasons": out.reasons,
                             "pos_err_cm": round(out.pos_err_m * 100, 2), "yaw_err_deg": round(out.yaw_err_deg, 2),
                             "final": out.final_pose, "contact_ticks": len(res.contacts), "sim_time_s": round(res.sim_time, 2)})
            ctx.save(f"target/attempt{attempt}_{strategy}.json", {"trace": res.trace, "contacts": res.contacts,
                                                                   "outcome": out.model_dump(), "phases": res.phases})
        ctx.enter("simulate", attempt=attempt, strategy=strategy, success=attempts[-1]["success"])
        ctx.enter("inspect", reasons=attempts[-1].get("reasons") or attempts[-1].get("unstable"))
        if attempts[-1]["success"]:
            break
        # revise
        report = {"strategy": strategy, "outcome": attempts[-1], "goal": _task_brief(spec),
                  "block_start": block0, "already_tried": tried,
                  "error_over_time": _error_series(res.trace, spec) if res else None}
        remaining = [o["id"] for o in options if o["id"] not in tried]
        space = REVISION_SPACE | {"strategy_id": {"choices": remaining}}
        new = remaining[0] if remaining else None
        diag_info = {"method": "rule_next_best_untried", "llm_used": False}
        if llm and llm.available and remaining:
            try:
                d, rec = llm.diagnose(report, space)
                diag_info = {"method": "nemotron", "llm_used": True, "failure_class": d.failure_class,
                             "evidence": d.evidence, "revision": d.revision, "accepted": rec.accepted,
                             "rejections": rec.rejection_reasons, "rationale": d.rationale}
                if d.give_up:
                    new = None
                elif d.revision.get("strategy_id") in remaining:
                    new = d.revision["strategy_id"]
            except (ReasoningError, ReasoningUnavailable) as e:
                diag_info["llm_error"] = str(e)[:300]
        ctx.save(f"diagnosis{attempt}.json", {"report": report, **diag_info, "next": new})
        ctx.enter("revise", next=new, method=diag_info["method"])
        if new is None:
            break
        strategy = new

    final_ok = attempts[-1]["success"]
    # 9. verify: held-out perturbations with the final strategy
    verify = []
    if final_ok and verify_n:
        for sd in range(20_000, 20_000 + verify_n):
            sc = sample_scenario(sd)
            vb = (sc.block[0], sc.block[1], sc.block[2])
            try:
                r = execute_arm_strategy(strategy, arm_b, spec, vb, sc.params, sd, source_ctrls, policies, render=False)
                verify.append({"seed": sd, "success": r.outcome.success, "pos_err_cm": round(r.outcome.pos_err_m * 100, 2),
                               "yaw_err_deg": round(r.outcome.yaw_err_deg, 2), "params": sc.params.as_dict(),
                               "block_start": list(vb), "reasons": r.outcome.reasons})
            except SimulationUnstable as e:
                verify.append({"seed": sd, "success": False, "unstable": str(e)})
        ctx.enter("verify", success=f"{sum(v['success'] for v in verify)}/{len(verify)}")
    ctx.save("verify.json", verify)

    # 10. export
    plan = {
        "format": "morph.plan/1",
        "valid_for": f"MuJoCo simulation of {target} as defined in this repository (NOT a real-robot controller)",
        "taskspec": spec.model_dump(), "target": caps.as_dict(), "strategy": strategy,
        "strategy_interface": next(o for o in options if o["id"] == strategy),
        "target_initial_condition": {"block": block0, "params": params.as_dict(), "seed": target_seed},
        "joint_command_trajectory": [{"t": t["t"], "ctrl": t["ctrl"]} for t in (res.trace if res else [])],
        "assumptions": spec.assumptions + ["ground-truth object pose from simulator", "control at 50 Hz"],
    }
    ctx.save("export/plan.json", plan)
    if render and res is not None and demo.frames and res.frames:
        g = (spec.gx, spec.gy, spec.gyaw)
        left = annotate_episode(demo.frames, demo.trace, g, f"SOURCE  {source} (demonstration)")
        right = annotate_episode(res.frames, res.trace, g, f"TARGET  {target}  [{strategy}]")
        side_by_side(left, right, run_dir / "replay_side_by_side.mp4")
    result |= {"status": "success" if final_ok else "failed", "strategy": strategy, "strategy_selection": choice_info,
               "attempts": attempts, "verify": {"success": sum(v["success"] for v in verify), "n": len(verify)},
               "target_block_start": block0, "target_params": params.as_dict()}
    return _finish(ctx, result, llm, t_start)


PUSH_EVIDENCE = Path("artifacts/evidence/pusher_eval.json")
GO1_EVIDENCE = Path("artifacts/evidence/go1_eval.json")
GO1_LEARNED_EVIDENCE = Path("artifacts/evidence/go1_learned_eval.json")


def push_strategy_options(target: str) -> list[dict]:
    ev = json.loads(PUSH_EVIDENCE.read_text()).get("methods", {}) if PUSH_EVIDENCE.exists() else {}
    g1 = json.loads(GO1_EVIDENCE.read_text()).get("methods", {}) if GO1_EVIDENCE.exists() else {}
    if GO1_LEARNED_EVIDENCE.exists():
        g1 |= json.loads(GO1_LEARNED_EVIDENCE.read_text()).get("methods", {})

    def stat(src, m):
        r = src.get(m)
        return f"{r['success']}/{r['n']} held-out successes" if r else "not evaluated"

    if target == "go1":
        from .robots.go1_learned import DEFAULT_POLICY
        learned = [] if not DEFAULT_POLICY.exists() else [{"id": "go1_learned_walk_foot_push", "interface": "RL-trained walking policy (velocity commands) to reach "
                 "the push pose, then 3-leg stance and front-right foot push; re-plan after each measured push",
                 "validation": stat(g1, "go1_learned_walk_foot_push"),
                 "assumptions": ["walking policy trained in simulation (MuJoCo Playground), deployed sim-to-sim",
                                 "block must stay upright; no skill to right a tipped block"]}]
        return learned + [
                {"id": "go1_foot_push_mpc", "interface": "hand-written static crawl to push pose, 3-leg stance, "
                 "front-right foot push", "validation": stat(g1, "go1_foot_push_mpc"),
                 "assumptions": ["crawl gait: no side-stepping; repositioning between pushes unreliable"]}]
    return [
        {"id": "closed_loop_push_mpc", "interface": "re-plan the next push with a nominal-physics model after every measured push",
         "validation": stat(ev, "pusher_mpc"), "assumptions": ["object slides on the floor", "planner assumes 0.2 kg, mu 0.6"]},
        {"id": "fixed_direction_push", "interface": "push toward the goal through the face centre; ignores heading",
         "validation": stat(ev, "pusher_fixed_direction"), "assumptions": ["heading does not matter"]},
        {"id": "open_loop_push_plan", "interface": "plan the whole push sequence once, execute without re-measuring",
         "validation": stat(ev, "pusher_open_loop"), "assumptions": ["the nominal model matches reality"]},
    ]


def _llm_push_revise(llm, ctx):
    """Nemotron diagnose() as the bounded revision step between pushes (only after an unproductive push)."""
    from .pusher_task import rule_revise

    def revise(rec, history, space, knobs):
        if len(history) < 2 or rec["pos_err_cm"] <= history[-2]["pos_err_cm"] - 0.5 or rec["yaw_err_deg"] < 3:
            return None  # productive push: no call
        report = {"last_push": rec, "previous_push": history[-2], "current_knobs": knobs,
                  "note": "planner uses nominal mass 0.2 kg and friction 0.6; true values unknown to it"}
        try:
            d, r = llm.diagnose(report, space)
            if d.give_up or not d.revision or not r.accepted:
                ctx.save(f"diagnosis_push{rec['i']}.json", {"report": report, "diagnosis": d.model_dump(),
                                                            "accepted": r.accepted, "rejections": r.rejection_reasons})
                return None
            ctx.save(f"diagnosis_push{rec['i']}.json", {"report": report, "diagnosis": d.model_dump(), "accepted": True})
            return {"apply": d.revision, "method": "nemotron", "failure_class": d.failure_class, "reason": d.rationale[:300]}
        except (ReasoningError, ReasoningUnavailable) as e:
            fb = rule_revise(rec, history, space, knobs)
            return fb | {"llm_error": str(e)[:200]} if fb else None
    return revise


def _run_push_target(ctx, llm, target, spec, demo, block0, params, target_seed, info, compliance, verify_n, render, t_start):
    from .go1_task import run_go1_push
    from .pusher_task import rule_revise, run_pusher
    from .robots.pusher import PusherAdapter
    from .robots.quadruped import QuadrupedAdapter
    caps = (PusherAdapter() if target == "pusher" else QuadrupedAdapter()).capabilities()
    cap = assess(spec, caps, block0)
    ctx.save("capability.json", {"target": caps.as_dict(), "report": cap.as_dict()})
    ctx.enter("assess", target=caps.robot, verdict=cap.verdict)
    result = {"run_dir": str(ctx.run_dir), "compliance": compliance, "capability": cap.verdict, "target": caps.robot,
              "demo": demo.summary(), "taskspec": spec.model_dump(), "inference": info}
    if cap.verdict == "unsupported":
        result |= {"status": "unsupported", "reasons": [c.evidence for c in cap.checks if c.status == "unsupported"]}
        return _finish(ctx, result, llm, t_start)
    options = push_strategy_options(target)
    strategy, choice_info = rule_select(options), {"method": "rule_highest_heldout_success", "llm_used": False}
    if llm and llm.available and len(options) > 1:
        try:
            ch, rec = llm.select_strategy(_task_brief(spec), caps.as_dict() | {"assessment": cap.as_dict()}, options)
            strategy = ch.strategy_id
            choice_info = {"method": "nemotron", "llm_used": True, "rationale": ch.rationale,
                           "evidence_cited": ch.evidence_cited, "rule_would_choose": rule_select(options)}
        except (ReasoningError, ReasoningUnavailable) as e:
            choice_info["llm_error"] = str(e)[:300]
    ctx.save("strategy.json", {"options": options, "chosen": strategy, **choice_info})
    ctx.enter("propose", strategy=strategy, method=choice_info["method"])
    revise = _llm_push_revise(llm, ctx) if (llm and llm.available) else rule_revise
    mode = {"closed_loop_push_mpc": "mpc", "fixed_direction_push": "fixed_direction", "open_loop_push_plan": "open_loop"}
    if target == "pusher":
        r = run_pusher(spec, block0, params, target_seed, mode=mode.get(strategy, "mpc"), render_every=3 if render else 0,
                       revise=revise, log=ctx.log)
        contact_key = "tip_contact_ticks"
    else:
        from .push_planner import PushModel, plan_push
        from .go1_task import push_pose
        heading = math.atan2(spec.gy - block0[1], spec.gx - block0[0])
        first, _ = plan_push(PushModel(), block0, (spec.gx, spec.gy, spec.gyaw), robot_pose=(0, 0, heading),
                             max_heading_change=math.radians(40))
        b0 = push_pose(first.contact_xy, first.direction)
        b0 = (b0[0] - 0.05 * math.cos(b0[2]), b0[1] - 0.05 * math.sin(b0[2]), b0[2])
        if strategy == "go1_learned_walk_foot_push":
            from .go1_learned_task import run_learned_go1_push
            r = run_learned_go1_push(spec, block0, params, target_seed, base0=(block0[0], block0[1] - 0.7, math.pi / 2),
                                     render_every=6 if render else 0, log=ctx.log)
        else:
            r = run_go1_push(spec, block0, params, target_seed, base0=b0, render_every=6 if render else 0, log=ctx.log)
        contact_key = "foot_contact_ticks"
    out = r.outcome
    best_err = min([(p["pos_err_cm"], p["yaw_err_deg"]) for p in r.pushes], default=None)
    attempt = {"strategy": strategy, "success": out.success, "reasons": out.reasons, "pos_err_cm": round(out.pos_err_m * 100, 2),
               "yaw_err_deg": round(out.yaw_err_deg, 2), "final": out.final_pose, "pushes": len(r.pushes),
               "contact_ticks": sum(p["contact"][contact_key] for p in r.pushes), "sim_time_s": round(r.sim_time, 1),
               "stop_reason": r.stop_reason, "best_observed_err": best_err,
               "unintended_contacts": getattr(r, "other_contacts", None) or getattr(r, "unintended_contacts", [])}
    ctx.save("target/pushes.json", {"pushes": r.pushes, "revisions": getattr(r, "revisions", []), "outcome": out.model_dump()})
    ctx.enter("simulate", strategy=strategy, success=out.success, pushes=len(r.pushes))
    ctx.enter("inspect", reasons=out.reasons or "none", stop=r.stop_reason)
    verify = []
    if out.success and verify_n and target == "pusher":
        from .learning import sample_scenario
        for sd in range(20_000, 20_000 + verify_n):
            sc = sample_scenario(sd)
            vr = run_pusher(spec, sc.block, sc.params, sd, mode=mode.get(strategy, "mpc"), log=lambda s: None)
            verify.append({"seed": sd, "success": vr.outcome.success, "pos_err_cm": round(vr.outcome.pos_err_m * 100, 2),
                           "yaw_err_deg": round(vr.outcome.yaw_err_deg, 2), "pushes": len(vr.pushes), "params": sc.params.as_dict()})
        ctx.enter("verify", success=f"{sum(v['success'] for v in verify)}/{len(verify)}")
    ctx.save("verify.json", verify)
    ctx.save("export/plan.json", {
        "format": "morph.plan/1", "valid_for": f"MuJoCo simulation of {caps.robot} in this repository (NOT a real-robot controller)",
        "taskspec": spec.model_dump(), "target": caps.as_dict(), "strategy": strategy,
        "executed_pushes": [{"contact_xy": p["plan"]["contact_xy"], "heading_rad": p["plan"]["direction"],
                             "distance_m": p["plan"]["distance"], "measured_result": p["block_after"]} for p in r.pushes],
        "assumptions": spec.assumptions + ["ground-truth object pose after every push", "flat floor, single object"]})
    if render and r.frames and demo.frames:
        from .viewer import annotate, goal_err_lines
        g = (spec.gx, spec.gy, spec.gyaw)
        left = annotate_episode(demo.frames, demo.trace, g, f"SOURCE  {demo.robot} (demonstration)")
        info_ = getattr(r, "frame_info", None)
        right = []
        for i, f in enumerate(r.frames):
            lines = []
            if info_ and i < len(info_):
                lines = goal_err_lines(info_[i]["block"], g, f"t={info_[i]['t']:.1f}s  pushes done: {info_[i]['push']}")
            right.append(annotate(f, f"TARGET  {caps.robot} (no gripper)  [{strategy}]", lines))
        side_by_side(left, right, ctx.run_dir / "replay_side_by_side.mp4")
    status = "success" if out.success else ("failed_incomplete_experiment" if target == "go1" else "failed")
    result |= {"status": status, "strategy": strategy, "strategy_selection": choice_info, "attempts": [attempt],
               "verify": {"success": sum(v["success"] for v in verify), "n": len(verify)},
               "target_block_start": block0, "target_params": params.as_dict(),
               "revisions": getattr(r, "revisions", [])}
    return _finish(ctx, result, llm, t_start)


def _task_brief(spec: TaskSpec) -> dict:
    return {"goal_x_m": round(spec.gx, 4), "goal_y_m": round(spec.gy, 4), "goal_yaw_deg": round(math.degrees(spec.gyaw), 2),
            "pos_tol_cm": round(float(spec.goal.pos_tol_m.value) * 100, 2),
            "yaw_tol_deg": round(math.degrees(float(spec.goal.yaw_tol_rad.value)), 2),
            "keep_upright": spec.goal.keep_upright.value, "orientation_mode": spec.goal.orientation_mode.value}


def _error_series(trace: list, spec: TaskSpec, n: int = 12) -> list:
    if not trace:
        return []
    idx = np.linspace(0, len(trace) - 1, n).astype(int)
    out = []
    for i in idx:
        b = trace[i]["block"]
        out.append({"t": trace[i]["t"], "pos_err_cm": round(math.hypot(b["x"] - spec.gx, b["y"] - spec.gy) * 100, 2),
                    "yaw_err_deg": round(abs(math.degrees(math.remainder(b["yaw"] - spec.gyaw, 2 * math.pi))), 1),
                    "block_z": b["z"], "tilt_deg": round(math.degrees(b["tilt"]), 1)})
    return out


def _finish(ctx: RunContext, result: dict, llm: ReasoningService | None, t_start: float) -> dict:
    recs = [r.as_dict() for r in (llm.records if llm else [])]
    result["inference_calls"] = [{"call": r["call"], "model": r["model"], "latency_s": r["latency_s"], "usage": r["usage"],
                                  "accepted": r["accepted"], "error": r["error"]} for r in recs]
    result["wall_s"] = round(time.time() - t_start, 1)
    result["state_log"] = ctx.state_log
    ctx.save("result.json", result)
    (ctx.run_dir / "report.md").write_text(render_report(result))
    ctx.enter("export", status=result.get("status"), dir=str(ctx.run_dir))
    return result


def render_report(r: dict) -> str:
    L = [f"# Morph run report", "", f"- Mode: **{r['compliance']}**", f"- Status: **{r.get('status')}**",
         f"- Capability verdict (target): `{r['capability']}`", ""]
    d = r["demo"]
    L += [f"## Source demonstration ({d['robot']})", "",
          f"- Instruction: _{d['instruction']}_", f"- Block start: {d['block_start']}", f"- Block end (measured): {d['block_end']}",
          f"- Observed change: {d['observed_change']}", f"- Gripper contact ticks: {d['contact_ticks']}", ""]
    ts = r["taskspec"]
    L += ["## Confirmed TaskSpec (robot-independent)", "", "| field | value | provenance | note |", "|---|---|---|---|"]
    for k in ("x", "y", "yaw", "pos_tol_m", "yaw_tol_rad", "keep_upright", "orientation_mode", "target_top_face"):
        f = ts["goal"][k]
        v = f["value"]
        if k in ("yaw", "yaw_tol_rad") and isinstance(v, (int, float)):
            v = f"{math.degrees(v):.1f} deg"
        L.append(f"| {k} | {v} | {f['provenance']} | {f['note'][:120]} |")
    L += ["", f"Inference method: `{r['inference'].get('method')}`", ""]
    if r.get("status") == "unsupported":
        L += ["## Unsupported", ""] + [f"- {x}" for x in r["reasons"]]
    else:
        L += ["## Strategy", "", f"- Chosen: `{r['strategy']}` via `{r['strategy_selection']['method']}`"]
        if r["strategy_selection"].get("rationale"):
            L.append(f"- Nemotron rationale: {r['strategy_selection']['rationale']}")
        L += ["", f"## Target trials ({r.get('target', 'target')})", "", "| # | strategy | success | pos err (cm) | yaw err (deg) | reasons |", "|---|---|---|---|---|---|"]
        for i, a in enumerate(r["attempts"]):
            L.append(f"| {i} | {a['strategy']} | {a['success']} | {a.get('pos_err_cm')} | {a.get('yaw_err_deg')} | {'; '.join(a.get('reasons') or [])} |")
        L += ["", f"Verification on held-out perturbations: **{r['verify']['success']}/{r['verify']['n']}**", ""]
    L += ["## Inference calls", "", "| call | model | latency (s) | tokens | accepted | error |", "|---|---|---|---|---|---|"]
    for c in r["inference_calls"]:
        tok = (c["usage"] or {}).get("total_tokens")
        L.append(f"| {c['call']} | {c['model']} | {c['latency_s']} | {tok} | {c['accepted']} | {c['error'] or ''} |")
    if not r["inference_calls"]:
        L.append("| (none: offline baseline) | | | | | |")
    L += ["", f"Wall time: {r['wall_s']} s. All poses are MuJoCo ground truth; no sim-to-real validation.", ""]
    return "\n".join(L)
