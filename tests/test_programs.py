"""Robot programs: checked before they run, run with measured step-by-step reports, judged by the checklist."""
from pathlib import Path

import pytest

from morph.goal_sources import goal_from_description_rules
from morph.programs import ProgramError, check_program, run_program, scene_names
from morph.project import _params, load_project

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def setup():
    from morph.robots.auto_arm import build_auto_arm
    pj = load_project(ROOT / "examples" / "parcel")
    goal = goal_from_description_rules("put the parcel on the shelf", pj.scene)
    ad, rep = build_auto_arm(ROOT / "third_party/mujoco_menagerie/franka_emika_panda/panda.xml", None, block=pj.scene.block,
                             log=lambda s: None, task_xy=[pj.scene.start[:2], (0.62, -0.02)], min_carry_z=0.105,
                             start=pj.scene.start, keepout=[(f.lo, f.hi) for f in pj.scene.fixtures.values()])
    return pj, goal, ad, scene_names(pj.scene, rep["travel_z"], 0.025)


@pytest.mark.parametrize("bad, says", [
    ([{"do": "teleport"}], "unknown building block"),
    ([{"do": "move_hand", "x": "__import__('os')"}], "can be called"),
    ([{"do": "move_hand", "x": "obj_q"}], "unknown name"),
    ([{"do": "move_hand", "x": 0.4, "speed": "warp"}], "speed"),
    ([], "no steps"),
])
def test_programs_are_checked_before_they_run(bad, says):
    with pytest.raises(ProgramError, match=says):
        check_program(bad)


def test_throw_to_becomes_a_wind_up_and_a_swing_that_lets_go():
    steps = check_program([{"do": "close_hand"}, {"do": "throw_to", "x": "bin_x", "y": "bin_y"}], {"bin_x", "bin_y"})
    assert [s["do"] for s in steps] == ["close_hand", "move_hand", "move_hand"]
    assert steps[2]["speed"] == "fast" and 0 < steps[2]["release_at"] < 1
    assert check_program([{"do": "move_hand", "z": 0.2}])[0]["x"] == "hand_x"  # left out: stay where the hand is


def test_a_sensible_program_reaches_the_goal(setup):
    pj, goal, ad, names = setup
    prog = check_program([
        {"do": "open_hand"}, {"do": "move_hand", "x": "obj_x", "y": "obj_y", "z": "obj_top + 0.06", "yaw": "object"},
        {"do": "move_hand", "x": "obj_x", "y": "obj_y", "z": "obj_z", "speed": "slow"}, {"do": "close_hand"},
        {"do": "move_hand", "z": 0.2}, {"do": "move_hand", "x": "shelf_x", "y": "shelf_y", "z": 0.2},
        {"do": "move_hand", "x": "shelf_x", "y": "shelf_y", "z": "shelf_top + obj_z - obj_bottom + 0.005", "speed": "slow"},
        {"do": "open_hand"}, {"do": "move_hand", "z": 0.2}], set(names))
    ad.reset(0, pj.scene.start, _params(0))
    run = run_program(ad, prog, goal, pj.scene, names=names)
    assert run["success"], run["reasons"]
    assert "holding the object" in run["report"][3]


def test_failures_are_reported_in_words_that_say_why(setup):
    pj, goal, ad, names = setup
    prog = check_program([{"do": "open_hand"}, {"do": "move_hand", "x": "obj_x", "y": "obj_y", "z": "obj_top + 0.06"},
                          {"do": "close_hand"}, {"do": "move_hand", "x": 3.0, "y": 0.0, "z": 0.2}], set(names))
    ad.reset(0, pj.scene.start, _params(0))
    run = run_program(ad, prog, goal, pj.scene, names=names)
    assert not run["success"]
    assert "closed on nothing" in run["report"][2]  # it never came down to the parcel
    assert "can't reach" in run["report"][3]  # 3 m away
