"""Robot profile + task -> plan.

Morph asks the same questions of every new body: what is it, how big, what does the task need,
which of the skills we have fits it, and if none does, what would have to be learned. The
preconditions of each skill template are checked here in code against the measured profile;
Nemotron then picks among the templates (and explains the choice), and a rule picks when no API
key is set. Nothing is executed here: see `morph plan --run`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .introspect import RobotProfile, capabilities_from_profile
from .scene import BlockSpec
from .task import TaskSpec

GO1_POLICY_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "go1_walk"
WALK_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "walk"  # policies Morph trained itself


def walk_policy_dir(p: RobotProfile) -> Path | None:
    """Where this body's walking policy lives, if one has been trained for it."""
    if _is_go1(p) and (GO1_POLICY_DIR / "go1_walk_policy.npz").exists():
        return GO1_POLICY_DIR
    d = WALK_DIR / p.name
    return d if (d / "walk_policy.npz").exists() else None


@dataclass
class TemplateCheck:
    id: str
    skill: str
    applicable: bool
    implemented: bool
    met: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


def _arm_chain(p: RobotProfile) -> dict | None:
    arms = [c for c in p.chains if c["role"] == "arm"]
    return max(arms, key=lambda c: c["n_actuated"]) if arms else None


def _is_go1(p: RobotProfile) -> bool:
    # The walking policy was trained on the Go1's 12 motors; it only transfers to that exact body.
    names = sorted(a["name"] for a in p.actuators)
    want = sorted(f"{leg}_{j}" for leg in ("FR", "FL", "RR", "RL") for j in ("hip", "thigh", "calf"))
    return p.n_legs == 4 and names == want and 10 < p.mass_kg < 15


def check_templates(p: RobotProfile, spec: TaskSpec, block: BlockSpec | None = None,
                    lift_reason: str = "") -> list[TemplateCheck]:
    """lift_reason: set when the goal needs the object lifted (over a bin wall, onto a shelf)."""
    block = block or BlockSpec()
    size = (2 * block.half_x, 2 * block.half_y, 2 * block.half_z)
    obj = min(size[:2])
    caps = capabilities_from_profile(p, size)
    out: list[TemplateCheck] = []

    # 1. top-down pick and place with a two-finger gripper
    t = TemplateCheck("grasp_place_topdown", "pick the object from above, carry it, set it down at the goal heading",
                      applicable=False, implemented=True,
                      evidence=["set up automatically by morph.robots.auto_arm (mount, carry height, servo lead, grip "
                                "strength found by trials); Panda-trained task-space policy then drives it"])
    arm = _arm_chain(p)
    (t.met if p.base == "fixed" else t.missing).append(f"base is {p.base}" + ("" if p.base == "fixed" else "; template needs a fixed base"))
    if p.gripper.get("type") in ("parallel_gripper", "single_jaw_gripper"):  # a jaw closing on a fixed jaw grips too
        opening = p.gripper.get("max_opening_m", 0.0)
        (t.met if opening >= obj + 0.005 else t.missing).append(
            f"gripper opens {opening * 100:.1f} cm for a {obj * 100:.0f} cm object")
    else:
        t.missing.append(f"end effector is {p.gripper.get('type', 'none')}; template needs a gripper with two sides (two fingers, or a jaw against a fixed jaw)")
    if arm:
        (t.met if arm["n_actuated"] >= 6 else t.missing).append(
            f"arm has {arm['n_actuated']} actuated joints (needs 6 to point down at any heading)")
    else:
        t.missing.append("no arm chain found")
    servo = [a for a in p.actuators if a["mode"] == "position"]
    (t.met if len(servo) >= 6 else t.missing).append(f"{len(servo)} position-controlled actuators")
    t.applicable = not t.missing
    out.append(t)

    # 2. walk to the object and push it with a foot (learned gait)
    t = TemplateCheck("legged_foot_push", "walk with a learned gait, then push the object with one foot while standing on three",
                      applicable=False, implemented=True,
                      evidence=["Go1: 12/16 goals reached, 0 falls, with the PPO walking policy (hand-written crawl: 0/8)",
                                "leg geometry for the foot push is measured from the model (numeric leg IK)"])
    (t.met if p.n_legs == 4 else t.missing).append(f"{p.n_legs} legs (template is for quadrupeds)")
    pol = walk_policy_dir(p)
    (t.met if pol else t.missing).append(
        f"walking policy trained for this body ({pol.relative_to(pol.parents[1])})" if pol else "no walking policy for this body yet")
    if lift_reason:
        t.missing.append(f"the goal needs the object lifted ({lift_reason}); pushing can't do that")
    t.applicable = not t.missing
    out.append(t)

    # 3. planar mobile base that shoves the object with its bumper
    t = TemplateCheck("mobile_push", "drive the base against the object and push it to the goal (model-predictive)",
                      applicable=False, implemented=True,
                      evidence=["built-in holonomic pusher: 26/30 (straight-line and replay baselines 2/30)"])
    (t.met if p.base == "planar_mobile" else t.missing).append(
        f"base is {p.base}" + ("" if p.base == "planar_mobile" else "; template needs x/y/yaw base joints"))
    if lift_reason:
        t.missing.append(f"the goal needs the object lifted ({lift_reason}); pushing can't do that")
    t.applicable = not t.missing
    out.append(t)

    # 4. learn it: reinforcement learning with the task spec as the reward
    legged4 = p.n_legs == 4 and p.base == "floating"
    t = TemplateCheck("learn_policy", "train a policy in simulation (PPO on the local GPU) with reward = progress "
                      "towards the TaskSpec goal, plus a locomotion policy first if the body has to walk",
                      applicable=False, implemented=legged4,
                      evidence=["Go1 walking: 200M steps, ~20 min on an RTX 5070 (MuJoCo Playground + Brax PPO)",
                                "quadrupeds: scripts/train_walk.py builds the walking task from the model itself, then "
                                "legged_foot_push uses the result" if legged4 else
                                "any body: 'Learn this task' (scripts/train_task.py) learns the goal itself from the "
                                "goal checklist; UR5e wall push learned in 60M steps, 7/10 tries"])
    (t.met if p.actuators else t.missing).append(f"{len(p.actuators)} actuators to learn with")
    movers = list(caps.contact_modes) + (["whole body (moving base)"] if p.base != "fixed" else [])
    (t.met if movers else t.missing).append("ways to touch the object: " + (", ".join(movers) or "none found"))
    if lift_reason and not caps.prehensile:
        t.missing.append(f"the goal needs the object lifted ({lift_reason}) and this body has nothing to hold it with")
    t.applicable = not t.missing
    out.append(t)
    return out


PRIORITY = ["grasp_place_topdown", "legged_foot_push", "mobile_push", "learn_policy"]


def learning_only(checks: list[TemplateCheck], reason: str | None, impossible: str | None = None) -> list[TemplateCheck]:
    """reason: the goal asks for something the ready-made skills don't do (goals.needs_learning).
    impossible: this body can't do it at all, learned or not (goals.reach_problem)."""
    for c in checks:
        why = impossible if impossible else (reason if c.id != "learn_policy" else None)
        if why and why not in c.missing:
            c.applicable = False
            c.missing.append(why)
    return checks


def rule_plan(checks: list[TemplateCheck]) -> dict:
    by = {c.id: c for c in checks}
    for tid in PRIORITY:
        c = by[tid]
        if c.applicable and c.implemented:
            return {"plan_id": tid, "rationale": f"first implemented template whose preconditions all hold: {'; '.join(c.met)}"}
    if by["learn_policy"].applicable:
        blockers = {c.id: c.missing for c in checks if c.id != "learn_policy"}
        return {"plan_id": "learn_policy", "rationale": f"no ready-made skill fits: {blockers}"}
    return {"plan_id": "unsupported", "rationale": "; ".join(by["learn_policy"].missing)}


def plan_context(p: RobotProfile, spec: TaskSpec, checks: list[TemplateCheck]) -> dict:
    g = spec.goal
    return {
        "robot": {"summary": p.summary(), "kind": p.kind, "base": p.base, "dof": p.dof, "mass_kg": round(p.mass_kg, 2),
                  "size_m": p.size_m, "legs": p.n_legs, "arms": p.n_arms, "gripper": p.gripper,
                  "reach_m": p.reach_m, "chains": [c for c in p.chains if c["role"] != "other"][:12], "notes": p.notes},
        "task": {"object": "5x5x6 cm block on the floor", "goal_xy_m": [g.x.value, g.y.value],
                 "goal_yaw_deg": round(math.degrees(g.yaw.value), 1), "pos_tol_cm": round(g.pos_tol_m.value * 100, 1),
                 "yaw_tol_deg": round(math.degrees(g.yaw_tol_rad.value), 1), "keep_upright": g.keep_upright.value},
        "templates": [asdict(c) for c in checks],
    }
