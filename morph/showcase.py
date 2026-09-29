"""Showcase video: the same goal, four different robots.

    morph showcase  ->  runs/showcase/showcase.mp4 (plus per-robot clips and results.json)

Panels: Franka Panda demo (source) | UFactory xArm7 running the Panda-trained policy |
mobile pusher | Unitree Go1 with learned walking + foot pushes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .live import SHOWCASE
from .task import default_spec
from .viewer import annotate, goal_err_lines

W, H = 640, 480


def _panel_frames(frames, infos, title, subtitle, goal, final_line, speedup: int = 1):
    out = []
    for i in range(0, len(frames), speedup):
        f, inf = frames[i], infos[min(i, len(infos) - 1)]
        lines = goal_err_lines(inf["block"], goal, f"t={inf['t']:.1f}s" + (f"  (x{speedup})" if speedup > 1 else ""))
        lines.append(subtitle)
        out.append(annotate(f, title, lines))
    last = annotate(frames[-1], title, [final_line, subtitle], color=(120, 255, 120) if final_line.startswith("SUCCESS")
                    else (255, 200, 120))
    return out + [last] * 25


def run_showcase(out: Path = Path("runs/showcase"), policy_source: str = "franka_panda", log=print) -> dict:
    import imageio.v2 as imageio

    from .arm_skills import expert_action, run_episode, task_obs
    from .go1_task import push_pose, run_go1_push
    from .learning import load_policy
    from .push_planner import PushModel, plan_push
    from .pusher_task import run_pusher
    from .robots.registry import make_arm

    out.mkdir(parents=True, exist_ok=True)
    block, goal, params = SHOWCASE["block"], SHOWCASE["goal"], SHOWCASE["params"]
    spec = default_spec(*goal)
    panels, results = [], {}

    def arm_clip(name, policy, title, subtitle):
        arm = make_arm(name)
        arm.env.track_azimuth = 150
        arm.reset(1, block, params)
        frames, infos = [], []

        def hook(a, phase):
            if a.env.data.time / a.env.control_dt % 3 < 1:
                frames.append(a.env.render("track", W, H))
                infos.append({"t": a.env.data.time, "block": a.env.block_pose().as_dict()})
        r = run_episode(arm, spec, policy or expert_action, mode="task", stop_when_done=policy is None,
                        on_tick=hook, max_time=12.0)
        o = r.outcome
        final = (f"SUCCESS  final error {o.pos_err_m * 100:.1f} cm / {o.yaw_err_deg:.1f} deg" if o.success
                 else "FAILED  " + "; ".join(o.reasons[:2]))
        results[name] = {"success": o.success, "pos_err_cm": round(o.pos_err_m * 100, 2), "yaw_err_deg": round(o.yaw_err_deg, 2),
                         "reasons": o.reasons, "sim_time_s": round(r.sim_time, 1), "gripper_contact_ticks": len(r.contacts)}
        log(f"{name}: {final}")
        return _panel_frames(frames, infos, title, subtitle, goal, final)

    panels.append(arm_clip("franka_panda", None, "SOURCE  Franka Emika Panda", "demonstration (scripted grasp skill)"))
    pol = load_policy(Path("artifacts/policies") / policy_source / "policy_task.pt")
    panels.append(arm_clip("ufactory_xarm7", lambda s: (pol.act(task_obs(s)), "policy"), "TARGET  UFactory xArm7",
                           f"policy learned on {policy_source}, zero-shot via xArm7 IK"))

    # arm-free mobile pusher
    frames, infos = [], []

    def ptick(rb):
        if rb.ticks % 3 == 0:
            frames.append(rb.env.render("track", W, H))
            infos.append({"t": rb.env.data.time, "block": rb.env.block_pose().as_dict()})
    r = run_pusher(spec, block, params, 3, on_tick=ptick, log=lambda s: None)
    o = r.outcome
    final = (f"SUCCESS  {len(r.pushes)} pushes, error {o.pos_err_m * 100:.1f} cm / {o.yaw_err_deg:.1f} deg" if o.success
             else "FAILED  " + "; ".join(o.reasons[:2]))
    results["mobile_pusher"] = {"success": o.success, "pos_err_cm": round(o.pos_err_m * 100, 2), "yaw_err_deg": round(o.yaw_err_deg, 2),
                                "pushes": len(r.pushes), "sim_time_s": round(r.sim_time, 1)}
    log(f"mobile_pusher: {final}")
    panels.append(_panel_frames(frames, infos, "TARGET  mobile pusher (no gripper)", "closed-loop push planning", goal, final,
                                speedup=2))

    # Unitree Go1: learned walking (RL policy trained locally) + three-leg foot push
    from .go1_learned_task import run_learned_go1_push
    frames, infos = [], []

    def gtick(r):
        if r.ticks % 6 == 0 or r.mode == "push" and int(r.env.data.time / 0.02) % 6 == 0:
            frames.append(r.env.render("track", W, H))
            infos.append({"t": r.env.data.time, "block": r.env.block_pose().as_dict()})
    g = run_learned_go1_push(spec, block, params, 3, base0=(block[0], block[1] - 0.7, math.pi / 2), on_tick=gtick,
                             log=lambda s: None)
    o = g.outcome
    final = (f"SUCCESS  {len(g.pushes)} foot pushes, error {o.pos_err_m * 100:.1f} cm / {o.yaw_err_deg:.1f} deg" if o.success
             else "FAILED  " + "; ".join(o.reasons[:2]))
    results["unitree_go1"] = {"success": o.success, "pos_err_cm": round(o.pos_err_m * 100, 2), "yaw_err_deg": round(o.yaw_err_deg, 2),
                              "pushes": len(g.pushes), "fell": g.fell, "stop": g.stop_reason, "sim_time_s": round(g.sim_time, 1),
                              "foot_contact_ticks": sum(p["contact"]["foot_contact_ticks"] for p in g.pushes)}
    log(f"unitree_go1: {final}")
    panels.append(_panel_frames(frames, infos, "TARGET  Unitree Go1 (no arm)", "learned walking (RL) + foot pushes", goal,
                                final, speedup=3))

    n = max(len(p) for p in panels)
    grid = []
    for i in range(n):
        f = [p[min(i, len(p) - 1)] for p in panels]
        grid.append(np.concatenate([np.concatenate(f[:2], 1), np.concatenate(f[2:], 1)], 0))
    imageio.mimsave(out / "showcase.mp4", grid, fps=17, macro_block_size=8)
    for name, p in zip(["panda", "xarm7", "pusher", "go1"], panels):
        imageio.mimsave(out / f"{name}.mp4", p, fps=17, macro_block_size=8)
    imageio.imwrite(out / "showcase_final.png", grid[-1])
    (out / "results.json").write_text(json.dumps(results, indent=2))
    return results
