"""Command-line entry point: `morph <command>` (or `python -m morph.cli`)."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")


def _run_dir(name: str) -> Path:
    return Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}"


def cmd_check_llm(a) -> int:
    from .reasoning import ReasoningService
    svc = ReasoningService()
    if not svc.available:
        print("NEBIUS_API_KEY is not set (export it or put it in .env). Live Nemotron path unavailable.")
        return 2
    info = svc.verify_model()
    print(json.dumps(info, indent=2))
    return 0 if info["available"] else 1


def cmd_transfer(a) -> int:
    from .pipeline import run_arm_transfer
    edits = {}
    if a.goal_yaw_deg is not None:
        edits["yaw"] = math.radians(a.goal_yaw_deg)
    if a.pos_tol_cm is not None:
        edits["pos_tol_m"] = a.pos_tol_cm / 100
    if a.yaw_tol_deg is not None:
        edits["yaw_tol_rad"] = math.radians(a.yaw_tol_deg)
    run_dir = Path(a.out) if a.out else _run_dir(f"{a.target}_{'flip' if a.flip else 'place'}{'_nollm' if a.no_llm else ''}")
    r = run_arm_transfer(run_dir, a.instruction, use_llm=not a.no_llm, edits=edits,
                         interactive=a.interactive, variant="face_up_flip" if a.flip else None,
                         verify_n=a.verify, render=not a.no_video, target=a.target, source=a.source)
    print(f"\nstatus: {r['status']}   mode: {r['compliance']}\nreport: {run_dir / 'report.md'}")
    return 0 if r["status"] in ("success", "unsupported", "failed_incomplete_experiment") else 1


def cmd_eval(a) -> int:
    from .evaluation import evaluate_transfer
    evaluate_transfer(a.n, a.methods, out=Path(a.out), source=a.source, targets=a.targets)
    print(f"wrote {a.out}/transfer_eval_{a.source}.json and .md")
    return 0


def cmd_train(a) -> int:
    from .learning import train_dagger
    for k in a.kinds:
        train_dagger(k, Path(a.out) / a.source, source=a.source)
    return 0


def cmd_spike(a) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import spike_quadruped
    print(json.dumps(spike_quadruped.main(video=not a.no_video), indent=2))
    return 0


def cmd_view(a) -> int:
    from .live import view
    view(a.robot, seed=a.seed, use_policy=a.policy, speed=a.speed, loop=not a.once)
    sys.stdout.flush()
    os._exit(0)  # skip interpreter teardown: MuJoCo's viewer thread + GL contexts can segfault at exit


def cmd_showcase(a) -> int:
    from .showcase import run_showcase
    r = run_showcase(Path(a.out), policy_source=a.policy_source)
    print(json.dumps(r, indent=2))
    print(f"video: {a.out}/showcase.mp4")
    return 0


def cmd_inspect(a) -> int:
    from .introspect import capabilities_from_profile, profile_robot
    from .robot_import import import_robot
    model, notes = import_robot(a.model)
    for n in notes:
        print(f"import: {n}")
    prof = profile_robot(model, name=a.name)
    caps = capabilities_from_profile(prof)
    print(prof.summary())
    for c in prof.chains:
        if c["role"] != "other":
            print(f"  {c['role']:8s} {c['leaf']:28s} {c['n_joints']:2d} joints ({c['n_actuated']} actuated), "
                  f"{c['length_m']:.2f} m long")
    if prof.gripper:
        print(f"  end effector: {prof.gripper.get('type', '?').replace('_', ' ')}"
              + (f", opens {prof.gripper['max_opening_m'] * 100:.1f} cm" if "max_opening_m" in prof.gripper else ""))
    print(f"  can: {', '.join(caps.contact_modes) or 'nothing with the current skills'}"
          f"  | base: {caps.base_mobility}  | workspace: {caps.object_workspace.get('type')}")
    for n in caps.notes:
        print("  note:", n)
    if a.save:
        out = Path(a.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"profile": prof.as_dict(), "capabilities": caps.as_dict()}, indent=2))
        print(f"saved {out}")
    return 0


def cmd_plan(a) -> int:
    from .introspect import profile_robot
    from .planner import check_templates, plan_context, rule_plan
    from .reasoning import ReasoningError, ReasoningService, ReasoningUnavailable
    from .task import default_spec
    from .robot_import import import_robot
    a.model, notes = import_robot(a.model)
    a.model = str(a.model)
    for n in notes:
        print(f"import: {n}")
    prof = profile_robot(a.model, name=a.name)
    spec = default_spec(a.goal[0], a.goal[1], math.radians(a.goal[2]), provenance="user")
    checks = check_templates(prof, spec)
    print(prof.summary())
    for c in checks:
        mark = "ok " if c.applicable else "-- "
        print(f"  {mark}{c.id:20s} " + ("; ".join(c.met) if c.applicable else "missing: " + "; ".join(c.missing)))
    run_dir = Path(a.out) if a.out else _run_dir(f"plan_{prof.name}")
    plan = rule_plan(checks) | {"planner": "rule"}
    if not a.no_llm:
        try:
            p, rec = ReasoningService(log_path=run_dir / "inference.jsonl").plan_for_robot(plan_context(prof, spec, checks))
            plan = p.model_dump() | {"planner": f"nemotron ({rec.latency_s}s)"}
        except (ReasoningUnavailable, ReasoningError) as e:
            print(f"  (Nemotron unavailable, using the rule: {e})")
    print(f"\nplan: {plan['plan_id']}  [{plan['planner']}]")
    for k in ("what_is_it", "size_and_reach", "goal", "how", "what_to_learn", "rationale"):
        if plan.get(k):
            print(f"  {k.replace('_', ' ')}: {plan[k]}")
    result = {"profile": prof.as_dict(), "templates": [c.__dict__ for c in checks], "plan": plan}
    if a.run and plan["plan_id"] == "grasp_place_topdown":
        result["trials"] = _run_grasp_plan(a.model, prof.name, spec, a.run, run_dir)
    elif a.run and plan["plan_id"] in ("legged_foot_push", "learn_policy") and prof.n_legs == 4:
        result["trials"] = _run_legged_plan(a.model, prof, spec, a.run, train=plan["plan_id"] == "learn_policy")
    elif a.run:
        print(f"  (--run: nothing to execute for {plan['plan_id']} from this command yet)")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plan.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"saved {run_dir / 'plan.json'}")
    return 0


def _run_grasp_plan(model: str, name: str, spec, n: int, out: Path) -> dict:
    """Set the arm up by experiment, try the transferred (Panda-trained) task-space policy, and if a
    short check shows it doesn't work on this body, fine-tune it here (DAgger) before the real trials."""
    import torch
    from .arm_skills import run_episode, task_obs
    from .learning import adapt_policy, load_policy, sample_scenario
    from .robots.auto_arm import build_auto_arm
    print("\nsetting the arm up by experiment:")
    ad, rep = build_auto_arm(model, name, log=print)
    print(f"  tool point {rep['tcp_pos']} on {rep['tcp_body']}, carry height {rep['travel_z']} m, "
          f"servo lead {rep['lead_scale']}, grip gain x{ad.cfg.gripper_gain_scale:g}")
    result = {"setup": rep}
    if not ad.reach_ok(spec.gx, spec.gy):
        print("  the goal is outside this arm's reach; nothing to run")
        return result | {"goal_reachable": False}

    def in_reach(sd):
        sc = sample_scenario(sd)
        return ad.reach_ok(*sc.block[:2]) and ad.reach_ok(*sc.goal[:2])

    def trials(pol, seeds, label):
        ok, runs = 0, []
        for sd in seeds:
            sc = sample_scenario(sd)
            ad.reset(sd, sc.block, sc.params)
            r = run_episode(ad, spec if label == "final" else _spec_for(sc), lambda s: (pol.act(task_obs(s)), "policy"))
            ok += r.outcome.success
            runs.append({"seed": sd, "success": r.outcome.success, "reasons": r.outcome.reasons[:2]})
        return ok, runs

    pol = load_policy(Path(__file__).resolve().parents[1] / "artifacts" / "policies" / "franka_panda" / "policy_task.pt")
    check_seeds = [sd for sd in range(30000, 30400) if in_reach(sd)][:6]
    ok, _ = trials(pol, check_seeds, "check")
    print(f"\ntransferred policy, quick check on this body: {ok}/{len(check_seeds)}")
    result["transfer_check"] = {"success": ok, "n": len(check_seeds)}
    if ok < 0.8 * len(check_seeds):
        print("learning needed: fine-tuning the policy on this body (DAgger, skill layer as teacher)")
        pol, hist = adapt_policy(ad, pol, [sd for sd in range(0, 2000) if in_reach(sd)])
        result["adaptation"] = hist
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"kind": "task", "n_obs": pol.net[0].in_features, "n_act": pol.n_act, "source": f"franka_panda+{name}",
                    "state_dict": pol.state_dict()}, out / "policy_task_adapted.pt")
    seeds = [sd for sd in range(20000, 21000) if ad.reach_ok(*sample_scenario(sd).block[:2])][:n]
    ok, runs = trials(pol, seeds, "final")
    print(f"{'adapted' if 'adaptation' in result else 'transferred'} policy on {name}, goal {spec.gx:.2f}, {spec.gy:.2f}: {ok}/{len(runs)}")
    return result | {"success": ok, "n": len(runs), "runs": runs}


def _run_legged_plan(model: str, prof, spec, n: int, train: bool) -> dict:
    """Train a walking policy for this body if it has none (GPU, separate .venv-rl), check it in Morph's
    scene, then walk + foot-push to the goal from N held-out block positions."""
    import subprocess
    from .go1_learned_task import run_learned_go1_push
    from .learning import HELDOUT_SEEDS, sample_scenario
    from .planner import WALK_DIR, walk_policy_dir
    from .robots.learned_walker import LearnedWalker, walk_check
    from .sim import PhysicalParams  # noqa: F401
    root = Path(__file__).resolve().parents[1]
    out = {}
    if train:
        py = root / ".venv-rl" / "bin" / "python"
        cmd = [str(py), str(root / "scripts" / "train_walk.py"), str(model), "--steps", "300000000",
               "--out", str(WALK_DIR / prof.name)]
        if not py.exists():
            print("learning needed, but the RL environment isn't installed; see requirements-rl.lock, then run:\n  "
                  + " ".join(cmd))
            return {"trained": False}
        print(f"\nlearning to walk: {' '.join(cmd[1:])}  (~35 min on an RTX 5070)")
        t0 = time.time()
        subprocess.run(cmd, check=True, env=dict(os.environ, MORPH_GPU_MEM=os.environ.get("MORPH_GPU_MEM", "0.7")))
        out["train_wall_s"] = round(time.time() - t0)
    pol = walk_policy_dir(prof)
    if pol is None:
        print("no walking policy for this body")
        return out
    if pol.name == "go1_walk":
        from .robots.go1_learned import LearnedGo1
        robot = LearnedGo1()
    else:
        print("\nwalking check in Morph's scene (full collisions, CPU solver):")
        out["walk_check"] = walk_check(pol)
        for r in out["walk_check"]:
            print(f"  {r['test']:15s} fell={r['fell']}  vx {r['mean_vx']:+.2f}  vy {r['mean_vy']:+.2f}  wz {r['mean_wz']:+.2f}")
        robot = LearnedWalker(pol)
    ok, runs = 0, []
    for sd in list(HELDOUT_SEEDS)[:n]:
        sc = sample_scenario(sd)
        r = run_learned_go1_push(spec, sc.block, sc.params, sd, base0=(sc.block[0], sc.block[1] - 0.7, math.pi / 2),
                                 robot=robot, log=lambda s: None)
        ok += r.outcome.success
        runs.append({"seed": sd, "success": r.outcome.success, "pos_err_cm": round(r.outcome.pos_err_m * 100, 1),
                     "pushes": len(r.pushes), "fell": r.fell, "stop": r.stop_reason})
        print(f"  trial {len(runs)}: {'success' if r.outcome.success else 'fail'}  error {runs[-1]['pos_err_cm']} cm, "
              f"{len(r.pushes)} pushes, fell={r.fell}  ({r.stop_reason})")
    print(f"{prof.name}, learned walk + foot push, goal {spec.gx:.2f}, {spec.gy:.2f}: {ok}/{len(runs)}")
    return out | {"success": ok, "n": len(runs), "runs": runs}


def _spec_for(sc):
    from .task import default_spec
    return default_spec(*sc.goal)


def cmd_learn(a) -> int:
    """Learn a task for one robot from scratch on this GPU (no browser needed, e.g. on a server or in CI)."""
    import subprocess
    from .learn_task import export, learn_in_rounds, prepare_learning
    from .project import load_project, resolve_goal
    from .reasoning import ReasoningService
    pj = load_project(a.project)
    llm = None if a.no_llm else ReasoningService()
    goal, _ = resolve_goal(pj, a.description, None, llm)
    print("goal:\n" + "\n".join(f"  - {line}" for line in goal.describe()))
    task, prep = prepare_learning(pj, goal, a.robot, llm=llm)
    t = json.loads(task.read_text())
    print(f"steps ({t.get('steps_from', '')}):")
    for k, st in enumerate(t.get("steps", []) + [t["steps_final"]], 1):
        print(f"  {k}. {st['name']}" + (f"\n       {st['how']}" if st.get("how") else "")
              + (f"\n       done when: {st['done_when']}" if st["done_when"] != "goal" else ""))
    ROOT = Path(__file__).resolve().parents[1]
    py = ROOT / ".venv-rl" / "bin" / "python"
    state = {"envs": a.envs}

    def train(task_path, n_steps, restore) -> bool:
        while True:
            cmd = [str(py), str(ROOT / "scripts" / "train_task.py"), str(task_path), "--steps", str(n_steps),
                   "--envs", str(state["envs"]), "--randomize", a.randomize] + (["--restore", str(restore)] if restore else []) \
                + (["--imitate-only"] if n_steps == 0 else [])
            print(f"training: {' '.join(cmd[1:])}", flush=True)
            r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                               env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
            for line in r.stdout.splitlines():
                if line.startswith('{"step"'):
                    print("  " + line, flush=True)
            oom = "out of memory" in r.stdout + r.stderr or "Failed to allocate" in r.stdout + r.stderr
            if r.returncode == 0 or not oom or state["envs"] <= 128:
                break
            state["envs"] //= 2
            print(f"GPU out of memory: retrying with {state['envs']} copies at once")
        if r.returncode != 0:
            print((r.stderr or r.stdout)[-2000:])
        return r.returncode == 0

    def emit(**e):
        if e.get("type") == "learn_round" and e.get("stage") == "revised":
            print(f"round {e['round']}: Nemotron revised the steps: {e['rationale']}")

    res = learn_in_rounds(pj, goal, task, train, llm=llm, log=print, emit=emit, total_steps=a.steps, rounds=a.rounds,
                          tries=a.tries)
    print(f"learned skill: {res['success']}/{res['n']} tries succeeded")
    if (task.parent / "policy.npz").exists():
        print(f"export: {export(task)}")
    return 0 if res["success"] else 2


def cmd_goal_bench(a) -> int:
    from .goal_bench import run
    out = run(use_llm=not a.no_llm, only=a.only)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    return 0


def cmd_export(a) -> int:
    from .learn_task import LEARNED, export
    folder = Path(a.learned)
    if not folder.exists():
        folder = LEARNED / a.learned
    if not (folder / "policy.npz").exists():
        names = ", ".join(p.name for p in LEARNED.glob("*") if (p / "policy.npz").exists())
        print(f"nothing learned at {a.learned} (learned so far: {names or 'none'})")
        return 1
    zp = export(folder / "task.json", Path(a.out) if a.out else None)
    print(f"{zp}\n  inside: policy.npz, policy.json, model.xml, run_policy.py, ros2/ (a ROS 2 package)")
    return 0


def cmd_do(a) -> int:
    from .project import load_project, resolve_goal, run_on_robot, save_result
    from .reasoning import ReasoningService
    pj = load_project(a.project)
    # command-line task replaces the project's: a --demo alone means "do what the demo did"
    if a.description is not None or a.demo:
        desc, demo = a.description or "", a.demo
    else:
        desc, demo = pj.description, pj.demo
    robots = a.robot or ([pj.robot] if pj.robot else [])
    run_dir = Path(a.out) if a.out else _run_dir(f"do_{pj.name}")
    llm = None if a.no_llm else ReasoningService(log_path=run_dir / "inference.jsonl")
    print(f"project {pj.name}: {pj.scene.object_name} in {Path(pj.scene.path).name}")
    for f in pj.scene.fixtures.values():
        print(f"  {f.describe()}")
    for n in pj.scene.notes:
        print(f"  note: {n}")
    if desc:
        print(f'request: "{desc}"')
    goal, info = resolve_goal(pj, desc, demo, llm)
    print(f"\ngoal ({info['goal_from']}):")
    for line in goal.describe():
        print(f"  - {line}")
    for n in goal.notes:
        print(f"  note: {n}")
    if sys.stdin.isatty() and not a.yes and robots:
        if input("run with this goal? [Y/n] ").strip().lower() in ("n", "no"):
            print("stopped; edit the description or pass --demo and try again")
            return 0
    result = {"project": pj.name, "goal": goal.model_dump(), "goal_info": info, "robots": []}
    for r in robots:
        result["robots"].append(run_on_robot(pj, goal, r, n=a.run, llm=llm, record_demo=a.record_demo))
    if len(result["robots"]) > 1:
        print("\nsummary:")
        for r in result["robots"]:
            print(f"  {r['robot']:14s} {r['plan']['plan_id']:20s} {r.get('success', 0)}/{r.get('n', 0)}")
    print(f"saved {save_result(run_dir, result)}")
    return 0


def cmd_app(a) -> int:
    # the app renders from worker threads: headless EGL unless the user chose a backend themselves
    if os.environ.get("MUJOCO_GL") == "glfw" and "MORPH_KEEP_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
    # Under WSL, Mesa falls back to software rendering (llvmpipe, ~150 ms per frame for the Go1) unless
    # told to use the GPU through D3D12 (~23 ms on an RTX 5070).
    if Path("/usr/lib/wsl/lib/libd3d12.so").exists() and "GALLIUM_DRIVER" not in os.environ:
        os.environ["GALLIUM_DRIVER"] = "d3d12"
        os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")
    from .app.server import serve
    serve(a.port, open_browser=not a.no_browser)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="morph", description="Cross-embodiment task-outcome transfer in MuJoCo")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("plan", help="profile an unseen robot model, check skill templates, and plan the task")
    s.add_argument("model")
    s.add_argument("--name")
    s.add_argument("--goal", nargs=3, type=float, default=(0.40, 0.12, 100.0), metavar=("X", "Y", "YAW_DEG"))
    s.add_argument("--no-llm", action="store_true")
    s.add_argument("--run", type=int, default=0, metavar="N", help="then run N trials of the chosen plan")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_plan)

    s = sub.add_parser("app", help="open Morph in the browser (scene, robots, goal, live simulation)")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(fn=cmd_app)

    s = sub.add_parser("do", help="your scene + a demo and/or a description -> goal -> any robot does it")
    s.add_argument("project", help="folder with morph.toml and the scene")
    s.add_argument("description", nargs="?", default=None, help='e.g. "put the parcel into the bin"')
    s.add_argument("--demo", help="recorded object path (.npz/.csv/.json) or joint log (.npz with qpos + model)")
    s.add_argument("--robot", action="append", help="robot model(s) to use (default: the project's)")
    s.add_argument("--run", type=int, default=5, metavar="N", help="trials per robot")
    s.add_argument("--record-demo", metavar="FILE", help="save the object's path from the first success")
    s.add_argument("--no-llm", action="store_true")
    s.add_argument("--yes", action="store_true", help="don't ask to confirm the goal")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_do)

    s = sub.add_parser("learn", help="learn a goal from scratch for one robot on this GPU, try it, export it")
    s.add_argument("project")
    s.add_argument("description")
    s.add_argument("--robot", required=True, help="MJCF, URDF or xacro")
    s.add_argument("--steps", type=int, default=60_000_000)
    s.add_argument("--envs", type=int, default=1024)
    s.add_argument("--randomize", choices=("off", "light", "full"), default="light")
    s.add_argument("--tries", type=int, default=10)
    s.add_argument("--rounds", type=int, default=4, help="practise in this many rounds; between them Nemotron "
                                                           "revises the steps from what the robot actually did")
    s.add_argument("--no-llm", action="store_true")
    s.set_defaults(fn=cmd_learn)
    s = sub.add_parser("goal-bench", help="does Morph understand varied requests? (judged on example outcomes)")
    s.add_argument("--no-llm", action="store_true", help="the built-in reader instead of Nemotron")
    s.add_argument("--only", help="run the cases whose name contains this")
    s.add_argument("--out", help="save the results as JSON")
    s.set_defaults(fn=cmd_goal_bench)
    s = sub.add_parser("export", help="export a learned policy: runner, model, and a ROS 2 package")
    s.add_argument("learned", help="folder under runs/learned (or its path)")
    s.add_argument("--out", help="export folder (default: <learned>/export)")
    s.set_defaults(fn=cmd_export)
    s = sub.add_parser("check-llm", help="verify NEBIUS_API_KEY and that the configured Nemotron model is served")
    s.set_defaults(fn=cmd_check_llm)

    s = sub.add_parser("transfer", help="demonstrate on arm A, transfer the outcome to arm B")
    s.add_argument("--instruction", default="Place the arrow-marked block inside the green zone with its arrow "
                                            "pointing north. Do not tip it over.")
    s.add_argument("--source", choices=["franka_panda", "arm_A"], default="franka_panda",
                   help="robot that performs the demonstration")
    s.add_argument("--target", choices=["ufactory_xarm7", "arm_B", "pusher", "go1"], default="ufactory_xarm7",
                   help="ufactory_xarm7 / arm_B: different arms; pusher: arm-free mobile pusher; go1: quadruped (experimental)")
    s.add_argument("--no-llm", action="store_true", help="offline deterministic baseline (NOT hackathon compliant)")
    s.add_argument("--interactive", action="store_true", help="confirm/edit the TaskSpec at a prompt")
    s.add_argument("--goal-yaw-deg", type=float)
    s.add_argument("--pos-tol-cm", type=float)
    s.add_argument("--yaw-tol-deg", type=float)
    s.add_argument("--flip", action="store_true", help="request a face-up flip (expected: unsupported)")
    s.add_argument("--verify", type=int, default=10)
    s.add_argument("--no-video", action="store_true")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_transfer)

    s = sub.add_parser("eval", help="held-out transfer evaluation of all methods")
    s.add_argument("-n", type=int, default=50)
    s.add_argument("--methods", nargs="*")
    s.add_argument("--source", default="franka_panda")
    s.add_argument("--targets", nargs="*", help="default: ufactory_xarm7 arm_A arm_B for the Panda; arm_B for arm_A")
    s.add_argument("--out", default="artifacts/evidence")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("train", help="train BC/DAgger policies on arm A")
    s.add_argument("--kinds", nargs="+", default=["task", "joint"])
    s.add_argument("--source", default="franka_panda", help="arm the policies are trained on")
    s.add_argument("--out", default="artifacts/policies")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("spike", help="Go1 foot-contact feasibility spike")
    s.add_argument("--no-video", action="store_true")
    s.set_defaults(fn=cmd_spike)

    s = sub.add_parser("view", help="open the interactive 3D viewer and watch a robot do the task live")
    s.add_argument("--robot", default="franka_panda",
                   choices=["franka_panda", "ufactory_xarm7", "arm_A", "arm_B", "pusher", "go1"])
    s.add_argument("--policy", action="store_true", help="arms: run the Panda-trained task-space policy instead of the scripted skill")
    s.add_argument("--seed", type=int, help="held-out scenario seed (default: the showcase task)")
    s.add_argument("--speed", type=float, default=1.0, help="playback speed vs real time")
    s.add_argument("--once", action="store_true", help="do not repeat")
    s.set_defaults(fn=cmd_view)

    s = sub.add_parser("showcase", help="render the 4-robot showcase video (Panda, xArm7, pusher, Go1)")
    s.add_argument("--out", default="runs/showcase")
    s.add_argument("--policy-source", default="franka_panda")
    s.set_defaults(fn=cmd_showcase)

    s = sub.add_parser("inspect", help="describe any robot from its MJCF/URDF model (type, size, reach, gripper, motors)")
    s.add_argument("model", help="path to the robot's MJCF (.xml), URDF or xacro")
    s.add_argument("--name")
    s.add_argument("--save", help="write the profile + inferred capabilities to this JSON file")
    s.set_defaults(fn=cmd_inspect)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
