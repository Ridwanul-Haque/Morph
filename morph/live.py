"""Interactive 3D viewer: watch any robot perform the transfer task live.

    morph view --robot franka_panda            # source arm, scripted skill
    morph view --robot ufactory_xarm7 --policy # Panda-trained task-space policy running on the xArm7
    morph view --robot pusher | go1

Uses MuJoCo's native viewer (mouse: rotate/pan/zoom; double-click to select; Space pauses in
the viewer UI). Needs a display (WSLg on Windows works).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from .learning import sample_scenario
from .sim import PhysicalParams
from .task import default_spec

SHOWCASE = {"block": (0.46, -0.20, 0.4), "goal": (0.45, 0.20, math.pi / 2), "params": PhysicalParams(0.25, 0.55)}


def _pacer(viewer, env, speed: float):
    """Keep the viewer at (a multiple of) real time and stop if the window is closed."""
    t_wall0, t_sim0 = time.time(), env.data.time

    def tick(*_):
        if not viewer.is_running():
            raise KeyboardInterrupt
        viewer.sync()
        ahead = (env.data.time - t_sim0) / speed - (time.time() - t_wall0)
        if ahead > 0:
            time.sleep(ahead)
    return tick


def view(robot: str, seed: int | None = None, use_policy: bool = False, speed: float = 1.0,
         policy_source: str = "franka_panda", loop: bool = True) -> None:
    if seed is None:
        block, goal, params = SHOWCASE["block"], SHOWCASE["goal"], SHOWCASE["params"]
    else:
        sc = sample_scenario(seed)
        block, goal, params = sc.block, sc.goal, sc.params
    spec = default_spec(*goal)
    while True:
        if robot == "pusher":
            from .pusher_task import run_pusher
            from .robots.pusher import PusherAdapter
            ad = PusherAdapter()
            ad.reset(0, block, params)
            with mujoco.viewer.launch_passive(ad.model, ad.env.data) as v:
                v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.6, 150, -30
                tick = _pacer(v, ad.env, speed)
                try:
                    r = run_pusher(spec, block, params, 0, adapter=ad, on_tick=tick, log=print)
                    print("RESULT:", "success" if r.outcome.success else r.outcome.reasons)
                    _linger(v)
                except KeyboardInterrupt:
                    return
        elif robot == "go1":
            from .go1_learned_task import run_learned_go1_push
            from .robots.go1_learned import LearnedGo1
            rb = LearnedGo1()
            base0 = (block[0], block[1] - 0.7, math.pi / 2)
            rb.reset(0, block, params, base_xy_yaw=base0)
            with mujoco.viewer.launch_passive(rb.model, rb.env.data) as v:
                v.cam.distance, v.cam.azimuth, v.cam.elevation = 2.0, 150, -25
                tick = _pacer(v, rb.env, speed)
                try:
                    r = run_learned_go1_push(spec, block, params, 0, base0=base0, robot=rb, on_tick=tick, log=print)
                    print("RESULT:", "success" if r.outcome.success else r.outcome.reasons, "|", r.stop_reason)
                    _linger(v)
                except KeyboardInterrupt:
                    return
        else:
            from .arm_skills import expert_action, run_episode, task_obs
            from .learning import load_policy
            from .robots.registry import make_arm
            arm = make_arm(robot)
            arm.reset(0, block, params)
            policy = None
            if use_policy:
                pol = load_policy(Path("artifacts/policies") / policy_source / "policy_task.pt")
                policy = lambda s: (pol.act(task_obs(s)), "policy")  # noqa: E731
            with mujoco.viewer.launch_passive(arm.model, arm.env.data) as v:
                v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.5, 150, -28
                v.cam.lookat[:] = [0.4, 0.0, 0.15]
                tick = _pacer(v, arm.env, speed)
                try:
                    r = run_episode(arm, spec, policy or expert_action, mode="task", stop_when_done=policy is None,
                                    on_tick=tick, max_time=14.0)
                    print("RESULT:", "success" if r.outcome.success else r.outcome.reasons,
                          f"| final error {r.outcome.pos_err_m * 100:.1f} cm, {r.outcome.yaw_err_deg:.1f} deg")
                    _linger(v)
                except KeyboardInterrupt:
                    return
        if not loop:
            return


def _linger(v, seconds: float = 3.0) -> None:
    t0 = time.time()
    while v.is_running() and time.time() - t0 < seconds:
        v.sync()
        time.sleep(0.02)
