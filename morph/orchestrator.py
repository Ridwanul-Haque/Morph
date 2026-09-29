"""TransferOrchestrator: demonstrate -> infer -> confirm -> assess -> propose ->
simulate -> inspect -> revise -> verify -> export, with persisted artifacts.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .arm_skills import expert_action, run_episode, task_obs
from .capability import assess
from .geometry import wrap_angle
from .reasoning import ReasoningError, ReasoningService, ReasoningUnavailable, TaskInterpretation
from .recorder import Demonstration, infer_taskspec_rule, record_demo
from .robots.arm import ArmAdapter
from .sim import PhysicalParams, SimulationUnstable
from .task import Field_, TaskSpec

STATES = ["demonstrate", "infer", "confirm", "assess", "propose", "simulate", "inspect", "revise", "verify", "export"]

SCENE_ZONES = {"green_zone": {"center_m": [0.45, 0.20], "radius_m": 0.05, "note": "the marked goal zone on the floor"}}


@dataclass
class RunContext:
    run_dir: Path
    llm: ReasoningService | None
    log: Callable[[str], None] = print
    state_log: list = field(default_factory=list)

    def enter(self, state: str, **info) -> None:
        self.state_log.append({"state": state, "t": round(time.time(), 3), **info})
        self.log(f"[{state}] " + ", ".join(f"{k}={v}" for k, v in info.items()))

    def save(self, name: str, obj) -> Path:
        p = self.run_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2, default=_json_default))
        return p


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "model_dump"):
        return o.model_dump()
    if hasattr(o, "as_dict"):
        return o.as_dict()
    raise TypeError(type(o))


# ------------------------------------------------------------------ infer
def spec_from_interpretation(base: TaskSpec, it: TaskInterpretation) -> TaskSpec:
    s = base.model_copy(deep=True)
    note = f"Nemotron ({it.goal_source}): {it.rationale[:300]}"
    s.goal.x = Field_(value=it.goal_x_m, provenance="inferred_llm", note=note)
    s.goal.y = Field_(value=it.goal_y_m, provenance="inferred_llm")
    s.goal.yaw = Field_(value=wrap_angle(math.radians(it.goal_yaw_deg)), provenance="inferred_llm")
    s.goal.pos_tol_m = Field_(value=it.pos_tol_cm / 100, provenance="inferred_llm")
    s.goal.yaw_tol_rad = Field_(value=math.radians(it.yaw_tol_deg), provenance="inferred_llm")
    s.goal.keep_upright = Field_(value=it.keep_upright, provenance="inferred_llm")
    s.goal.orientation_mode = Field_(value=it.orientation_mode, provenance="inferred_llm")
    s.goal.target_top_face = Field_(value=it.target_top_face, provenance="inferred_llm")
    s.permitted_contacts = Field_(value=["any_robot_link"] + [f"exclude: {e}" for e in it.exclusions],
                                  provenance="inferred_llm")
    s.assumptions = base.assumptions + [f"Ambiguity flagged by Nemotron: {a}" for a in it.ambiguities]
    return s


def sanity_check_interpretation(it: TaskInterpretation, demo: Demonstration) -> list[str]:
    """Reject LLM goals that are physically implausible for this scene (not a correctness oracle)."""
    issues = []
    if not (0.0 <= it.goal_x_m <= 1.0 and -0.6 <= it.goal_y_m <= 0.6):
        issues.append(f"goal ({it.goal_x_m:.2f}, {it.goal_y_m:.2f}) outside the modelled floor area")
    be = demo.block_end
    if it.goal_source == "measured_final_pose" and math.hypot(it.goal_x_m - be["x"], it.goal_y_m - be["y"]) > 0.02:
        issues.append("claims to use the measured pose but differs from it by > 2 cm")
    return issues


def infer(ctx: RunContext, demo: Demonstration, instruction: str) -> tuple[TaskSpec, dict]:
    rule = infer_taskspec_rule(demo)
    info = {"method": "rule_measured_final_pose", "llm_used": False}
    if ctx.llm and ctx.llm.available:
        try:
            summary = demo.summary()
            summary.pop("phases", None)
            it, rec = ctx.llm.interpret_task(instruction, summary | {"scene_zones": SCENE_ZONES},
                                             {"name": "arrow_block", "size_m": [0.08, 0.08, 0.06],
                                              "marker": "arrow on top face; red stripe on the face the arrow points to"})
            issues = sanity_check_interpretation(it, demo)
            if issues:
                rec.accepted = False
                rec.rejection_reasons += issues
                info |= {"llm_used": True, "llm_rejected": issues}
            else:
                rec.accepted = True
                return spec_from_interpretation(rule, it), {"method": "nemotron_interpretation", "llm_used": True,
                                                            "latency_s": rec.latency_s, "usage": rec.usage}
        except (ReasoningError, ReasoningUnavailable) as e:
            info |= {"llm_error": str(e)[:300]}
    return rule, info


# ------------------------------------------------------------------ strategies
def arm_strategy_options(evidence: dict, source: str, target: str, same_joint_count: bool) -> list[dict]:
    ev = evidence.get("methods", {})

    def stat(m):
        r = ev.get(m)
        return f"{r['success']}/{r['n']} held-out successes" if r else "not evaluated"

    opts = [
        {"id": "goal_replan", "interface": "TaskSpec goal -> target arm's own IK pick-and-place skill",
         "uses_source_joints": False, "validation": stat(f"goal_replan_{target}"),
         "assumptions": ["target has a validated grasp primitive", "object within target workspace"]},
        {"id": "learned_task_policy", "interface": f"policy trained on {source} in task space, executed via target IK",
         "uses_source_joints": False, "validation": stat(f"task_policy_{source}_to_{target}"),
         "assumptions": ["observation contains only object/TCP relations", "target IK available"]},
    ]
    if same_joint_count:
        opts += [
            {"id": "learned_joint_policy", "interface": f"policy trained on {source} in joint space, applied to target joints",
             "uses_source_joints": True, "validation": stat(f"joint_policy_{source}_to_{target}"),
             "assumptions": ["target kinematics identical to source"]},
            {"id": "joint_replay", "interface": "replay source joint targets and gripper open/close open-loop",
             "uses_source_joints": True, "validation": stat(f"joint_replay_{source}_to_{target}"),
             "assumptions": ["identical kinematics and initial conditions"]},
        ]
    return opts


def rule_select(options: list[dict]) -> str:
    """No-LLM baseline: highest recorded held-out success, ties -> first listed."""
    def score(o):
        v = o["validation"]
        if "/" not in v:
            return -1.0
        k, n = v.split(" ")[0].split("/")
        return int(k) / max(1, int(n))
    return max(options, key=score)["id"]


def execute_arm_strategy(strategy: str, target: ArmAdapter, spec: TaskSpec, block0, params, seed: int,
                         source_ctrls: list | None, policies: dict, render: bool = True):
    target.reset(seed, block0, params)
    if strategy == "goal_replan":
        return run_episode(target, spec, expert_action, record=True, render_every=3 if render else 0)
    if strategy == "learned_task_policy":
        pol = policies["task"]
        return run_episode(target, spec, lambda s: (pol.act(task_obs(s)), "policy"), mode="task", record=True,
                           render_every=3 if render else 0, stop_when_done=False, max_time=12.0)
    if strategy == "learned_joint_policy":
        from .arm_skills import joint_obs
        pol = policies["joint"]
        return run_episode(target, spec, lambda s: (pol.act(joint_obs(s)), "policy"), mode="joint", record=True,
                           render_every=3 if render else 0, stop_when_done=False, max_time=12.0)
    if strategy == "joint_replay":
        from .evaluation import _run_ctrl_sequence
        cmds = source_ctrls or []
        it = iter(cmds)
        return _run_ctrl_sequence(target, spec, lambda s: next(it, cmds[-1]), len(cmds),
                                  max_time=12.0, render_every=3 if render else 0, record=True)
    raise KeyError(strategy)
