"""Programs for any body: the general building blocks (move any part, pointing any way, grip, push) on the learner's own
model, and the legged blocks on a learned gait. Checked before they run; run with measured reports."""
from pathlib import Path

import numpy as np
import pytest

from morph.body_programs import BodyRunner, calibrate_grip, check_body_program, run_body_program
from morph.goal_sources import goal_from_description_rules
from morph.programs import ProgramError, scene_names
from morph.project import _params, load_project
from morph.task_model import load_task

ROOT = Path(__file__).resolve().parents[1]
PANDA = ROOT / "third_party/mujoco_menagerie/franka_emika_panda/panda.xml"


@pytest.fixture(scope="module")
def panda(tmp_path_factory):
    from morph.learn_task import make_task
    pj = load_project(ROOT / "examples" / "parcel")
    goal = goal_from_description_rules("lift the parcel 10 cm off the ground and hold it there", pj.scene)
    task = make_task(pj, goal, str(PANDA), plan=False, folder=tmp_path_factory.mktemp("panda"))
    rn = BodyRunner(load_task(task))
    calibrate_grip(rn, pj.scene, log=lambda s: None)
    return pj, goal, rn, scene_names(pj.scene, 0.2, 0.01)


def test_the_parts_of_an_arm_with_a_gripper_are_found(panda):
    _, _, rn, _ = panda
    hand = rn.parts["hand"]
    assert hand.can_grip and len(hand.joints) == 7  # the wrist chain; the fingers move with the gripper
    assert rn.pointing(hand)[2] < -0.98  # after the grip test it points (nearly) straight down


@pytest.mark.parametrize("bad, says", [
    ([{"do": "teleport"}], "unknown building block"),
    ([{"do": "move", "part": "tail", "x": 0.4}], "part must be one of"),
    ([{"do": "move", "part": "hand", "x": "__import__('os')"}], "can be called"),
    ([{"do": "move", "part": "hand", "x": 0.4, "point": "sideways"}], "point must be"),
    ([{"do": "grip", "how": "squeeze"}], "close"),
    ([{"do": "joints", "joints": {"elbow_of_doom": 1.0}}], "unknown joint"),
])
def test_body_programs_are_checked_before_they_run(panda, bad, says):
    _, _, rn, names = panda
    with pytest.raises(ProgramError, match=says):
        check_body_program(bad, rn.parts, ["joint1"], set(names))


def test_a_general_program_lifts_the_parcel_and_reports_each_step(panda):
    pj, goal, rn, names = panda
    prog = check_body_program([
        {"do": "grip", "how": "open"},
        {"do": "move", "part": "hand", "x": "obj_x", "y": "obj_y", "z": "obj_top + 0.08", "point": "down", "turn": "object"},
        {"do": "move", "part": "hand", "z": "obj_z", "speed": "slow"},
        {"do": "grip", "how": "close"},
        {"do": "move", "part": "hand", "z": "obj_z + 0.15", "speed": "slow"},
        {"do": "wait", "seconds": 1.0}], rn.parts, [], set(names))
    rn.reset(0, pj.scene.start, _params(0))
    rec = []
    run = run_body_program(rn, prog, goal, pj.scene, names=names, record=rec)
    assert run["success"], run["report"]
    assert "the fingers are on the object" in run["report"][3]
    assert len(rec) > 100 and len(rec[0]) == 3  # recorded on the learner's model: (qpos, qvel, ctrl) per tick


def test_pointing_a_part_another_way(panda):
    pj, goal, rn, names = panda
    prog = check_body_program([{"do": "move", "part": "hand", "x": 0.35, "y": -0.17, "z": 0.25, "point": "east"}],
                              rn.parts, [], set(names))
    rn.reset(0, pj.scene.start, _params(0))
    run = run_body_program(rn, prog, goal, pj.scene, names=names)
    assert "points" not in run["report"][0], run["report"][0]  # within 10° of east
    assert np.dot(rn.pointing(rn.parts["hand"]), [1, 0, 0]) > 0.95


def test_legged_programs_are_checked():
    from morph.legged_programs import check_leg_program
    with pytest.raises(ProgramError, match="needs x"):
        check_leg_program([{"do": "walk_to", "y": 0.2}])
    with pytest.raises(ProgramError, match="height"):
        check_leg_program([{"do": "foot_push", "direction": 0, "distance": 0.1, "height": "medium"}])
    s = check_leg_program([{"do": "walk_to", "x": "obj_x - 0.5", "y": "obj_y"}, {"do": "foot_push", "direction": 90}])
    assert s[0]["facing"] == "object" and s[1]["height"] == "low"


def test_a_legged_program_walks_and_pushes_with_a_foot():
    from morph.introspect import profile_robot
    from morph.legged_programs import check_leg_program, make_robot, run_leg_program
    pj = load_project(ROOT / "examples" / "parcel")
    prof = profile_robot(ROOT / "third_party/mujoco_menagerie/unitree_go1/go1.xml")
    rb = make_robot(prof, pj.scene)
    if rb is None:
        pytest.skip("no Go1 walking policy in this checkout")
    goal = goal_from_description_rules("push the parcel 5 cm east", pj.scene)
    rb.reset(0, pj.scene.start, _params(0), base_xy_yaw=(-0.25, -0.18, 0.0))
    run = run_leg_program(rb, check_leg_program([{"do": "foot_push", "direction": 0, "distance": 0.07}]), goal, pj.scene)
    assert not rb.fallen()
    assert "pushed with a front foot" in run["report"][0], run["report"]
    assert run["success"], run["report"]


def test_repeat_if_goes_back_while_the_condition_holds(panda):
    """A reactive step: nothing is held, so it goes back (at most `times`), then carries on."""
    pj, goal, rn, names = panda
    prog = check_body_program([{"do": "wait", "seconds": 0.1},
                               {"do": "repeat_if", "condition": "held == 0", "back_to": 1, "times": 2}],
                              rn.parts, [], set(names), rn.task["fixtures"])
    rn.reset(0, pj.scene.start, _params(0))
    rep = run_body_program(rn, prog, goal, pj.scene, names=names)["report"]
    assert [r.split(":")[0] for r in rep] == ["1. wait 0.1 s", "2. if held == 0", "1. wait 0.1 s", "2. if held == 0",
                                              "1. wait 0.1 s", "2. if held == 0"]
    assert "already repeated 2 times" in rep[-1]
    with pytest.raises(ProgramError, match="back_to"):
        check_body_program([{"do": "repeat_if", "condition": "held == 0", "back_to": 3}], rn.parts, [], set(names))


def test_steps_written_other_common_ways_are_understood():
    from morph.programs import normalize_step
    assert normalize_step({"do": "grip", "args": ["open"]}) == {"do": "grip", "how": "open"}
    assert normalize_step({"do": "grip", "args": "close"})["how"] == "close"
    s = normalize_step({"do": "move", "args": {"part": "hand", "x": "where_it_must_end.x", "z": "bin.top + 0.02"}})
    assert s == {"do": "move", "part": "hand", "x": "goal_x", "z": "bin_top + 0.02"}
    assert normalize_step({"op": "wait", "seconds": 1})["do"] == "wait"
