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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="morph", description="Cross-embodiment task-outcome transfer in MuJoCo")
    sub = ap.add_subparsers(dest="cmd", required=True)

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

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
