"""The steps to go about a task: written per body, measurable, followed in order."""
from pathlib import Path

import numpy as np
import pytest

from morph.goal_sources import goal_from_description_rules
from morph.goal_terms import normalize, parse, quantities
from morph.goals import placement
from morph.introspect import profile_robot
from morph.project import load_project
from morph.skill_steps import GOAL, Step, StepTracker, check_steps, plan_steps

ROOT = Path(__file__).resolve().parents[1]
MEN = ROOT / "third_party" / "mujoco_menagerie"


@pytest.fixture(scope="module")
def pj():
    return load_project(ROOT / "examples" / "parcel")


@pytest.mark.parametrize("robot, goal, expect", [
    ("franka_emika_panda/panda.xml", "put the parcel into the bin", "grip"),
    ("universal_robots_ur5e/ur5e.xml", "push the parcel against the wall", "behind"),
    ("unitree_go2/go2.xml", "push the parcel against the wall", "Walk"),
    ("universal_robots_ur5e/ur5e.xml", "knock the parcel over", "top half"),
])
def test_morphs_own_steps_fit_the_body_and_can_be_measured(pj, robot, goal, expect):
    prof = profile_robot(MEN / robot)
    g = goal_from_description_rules(goal, pj.scene)
    steps, info = plan_steps(prof, g, pj.scene, placement(g, pj.scene), llm=None, log=lambda s: None)
    assert info["from"] == "rules" and not info["notes"]
    assert steps[-1].done_when == GOAL and len(steps) >= 2
    assert any(expect in s.name for s in steps[:-1])
    for s in steps[:-1]:
        parse(s.done_when, list(pj.scene.fixtures))


def test_other_ways_of_writing_formulas_are_understood():
    assert normalize("parcel_x > 0.3 && !held || object_tilt_deg < 5", "parcel") == "x > 0.3  and   not held  or  tilt_deg < 5"
    parse("hand_dist <= 0.04 && held == true", [])


def test_unmeasurable_steps_are_reported_not_kept():
    ok, errors = check_steps([Step(name="a", done_when="hand_dist < 0.1"), Step(name="b", done_when="gripper_force > 3")], [])
    assert [s.name for s in ok] == ["a"] and "gripper_force" in errors[0]


def test_tracker_goes_in_order_and_skips_one_too_strict_step():
    steps = [{"name": "near", "done_when": "hand_dist < 0.1"}, {"name": "exact", "done_when": "hand_dist < 0.0001"},
             {"name": "lifted", "done_when": "z > 0.2"}, {"name": "far", "done_when": "x > 5"}]
    seen = []
    tr = StepTracker(steps, {}, on_step=lambda k, n: seen.append(n))

    def v(z, hand):
        return quantities(np, pos=np.array([0.5, 0, z]), R=np.eye(3), vel=np.zeros(3), start=(0.5, 0), half=(0.025,) * 3,
                          fixtures={}, held=0.0, hand_dist=hand)
    for _ in range(20):
        tr.update(v(0.3, 1.0))  # "lifted" is true, but step 1 isn't done yet: no jumping ahead
    assert tr.reached == 0
    for _ in range(5):
        tr.update(v(0.03, 0.05))  # true for a moment (a bounce): doesn't count
    tr.update(v(0.03, 1.0))
    assert tr.reached == 0
    for _ in range(10):
        tr.update(v(0.03, 0.05))
    assert tr.reached == 1
    for _ in range(10):
        tr.update(v(0.3, 0.05))  # "exact" is impossible, "lifted" holds: skip the too-strict step
    assert tr.reached == 3 and seen == ["near", "lifted"]
    assert "stuck at: far" in tr.summary()


def _pinch_world(on_floor: bool, two_sides: bool = True):
    """A free 5 cm block, two robot 'fingers' pressing on it from either side (or one), in the air or on the floor."""
    import mujoco
    from types import SimpleNamespace
    z = 0.025 if on_floor else 0.5
    right = '<geom type="box" size="0.01 0.02 0.02" pos="0.034 0 0"/>' if two_sides else ""
    xml = f"""<mujoco><option gravity="0 0 {-9.81 if on_floor else 0}"/><worldbody>
      <geom name="floor" type="plane" size="1 1 0.1"/>
      <body name="block" pos="0 0 {z}"><freejoint name="block_free"/><geom name="block_geom" type="box" size="0.025 0.025 0.025" mass="0.2"/></body>
      <body name="robot_fingers" pos="0 0 {z}"><geom type="box" size="0.01 0.02 0.02" pos="-0.034 0 0"/>{right}</body>
    </worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    for _ in range(20):
        mujoco.mj_step(m, d)
    blk = m.body("block").id
    ix = SimpleNamespace(block=blk, robot_geoms=np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("robot_fingers").id]),
                         block_geoms=np.array([m.geom("block_geom").id]))
    return m, d, ix


def test_held_means_a_real_two_sided_grip_with_nothing_else_carrying_it():
    from morph.task_model import contact_held
    assert contact_held(*_pinch_world(on_floor=False)) == 1.0  # pinched from both sides, in the air
    assert contact_held(*_pinch_world(on_floor=True)) == 0.0  # fingers on it, but the floor carries it
    assert contact_held(*_pinch_world(on_floor=False, two_sides=False)) == 0.0  # touched on one side only


def test_a_loosely_written_step_is_still_a_step():
    """keep: false, a word for body_parts, a name a little too long: a model's reply shouldn't lose a whole plan."""
    from morph.skill_steps import Step
    s = Step(name="x" * 120, done_when="touch_dist < 0.02", keep=False, body_parts="gripper", how=None)
    assert s.keep == "" and s.body_parts == ["gripper"] and len(s.name) == 80 and s.how == ""
