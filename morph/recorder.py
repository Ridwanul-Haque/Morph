"""Records a source demonstration and extracts what changed in the world (object
pose before/after). The robot's joints never make it into the TaskSpec.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .arm_skills import expert_action, run_episode
from .geometry import wrap_angle
from .robots.arm import ArmAdapter
from .sim import PhysicalParams
from .task import TaskSpec, default_spec


@dataclass
class Demonstration:
    robot: str
    seed: int
    params: PhysicalParams
    block_start: tuple[float, float, float]
    block_end: dict
    trace: list
    contacts: list
    phases: list
    success_vs_intended: bool
    instruction: str
    sim_time: float
    frames: list

    def summary(self) -> dict:
        be = self.block_end
        dx, dy = be["x"] - self.block_start[0], be["y"] - self.block_start[1]
        return {
            "robot": self.robot,
            "seed": self.seed,
            "physical_params": self.params.as_dict(),
            "block_start": {"x": self.block_start[0], "y": self.block_start[1], "yaw": self.block_start[2]},
            "block_end": be,
            "observed_change": {
                "translation_m": round(math.hypot(dx, dy), 4),
                "yaw_change_deg": round(math.degrees(wrap_angle(be["yaw"] - self.block_start[2])), 2),
                "final_tilt_deg": round(math.degrees(be["tilt"]), 2),
                "lifted_during_demo": max(t["block"]["z"] for t in self.trace) > 0.05 if self.trace else None,
            },
            "contact_ticks": len(self.contacts),
            "contact_bodies": sorted({c["geom"] for c in self.contacts}),
            "phases": self.phases,
            "sim_time_s": round(self.sim_time, 3),
            "instruction": self.instruction,
            "demo_reached_intended_goal": self.success_vs_intended,
        }

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(self.summary(), indent=2))
        (out / "trace.json").write_text(json.dumps({
            "robot": self.robot, "seed": self.seed, "physical_params": self.params.as_dict(),
            "control_dt": 0.02, "timestep": 0.002, "data_source": "MuJoCo ground truth (no perception model)",
            "trace": self.trace, "contacts": self.contacts}))


def record_demo(arm: ArmAdapter, seed: int, block_start: tuple[float, float, float], intended_goal: tuple[float, float, float],
                instruction: str, params: PhysicalParams = PhysicalParams(), render: bool = False) -> Demonstration:
    """Run the scripted source skill towards the operator's intended goal and record it.

    The intended goal is what the demonstrator was aiming for; Morph never reads it
    back when inferring the task -- inference uses only the recorded object motion.
    """
    arm.reset(seed, block_start, params)
    intended = default_spec(*intended_goal)
    t0 = time.time()
    res = run_episode(arm, intended, expert_action, record=True, render_every=3 if render else 0)
    return Demonstration(robot=arm.name, seed=seed, params=params, block_start=block_start,
                         block_end=res.outcome.final_pose, trace=res.trace, contacts=res.contacts,
                         phases=[p for _, p in res.phases], success_vs_intended=res.outcome.success,
                         instruction=instruction, sim_time=res.sim_time, frames=res.frames)


def infer_taskspec_rule(demo: Demonstration) -> TaskSpec:
    """Deterministic (no-LLM) interpretation: the goal is the measured final object pose.

    Everything comes from the object trace; tolerances are defaults the user must confirm.
    """
    be = demo.block_end
    spec = default_spec(be["x"], be["y"], be["yaw"], provenance="measured")
    spec.goal.x.note = "final block x measured in source demo"
    spec.goal.y.note = "final block y measured in source demo"
    spec.goal.yaw.note = "final arrow heading measured in source demo"
    spec.goal.keep_upright.provenance = "inferred_rule"
    spec.goal.keep_upright.note = f"block ended with tilt {math.degrees(be['tilt']):.1f} deg"
    spec.object.mass_kg = demo.params.block_mass
    spec.object.friction = demo.params.block_friction
    spec.instruction = demo.instruction
    spec.assumptions = [
        "Only the final planar pose (x, y, yaw) of the block matters; the path and grasp do not.",
        "Arrow heading is unambiguous (asymmetric marker), so yaw is compared over 360 deg.",
        "Block must rest upright on the floor and be released at the end.",
        "Poses are simulator ground truth; no perception noise is modelled.",
    ]
    return spec


def load_trace(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def np_default(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))
