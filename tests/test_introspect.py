from pathlib import Path

import pytest

from morph.introspect import capabilities_from_profile, profile_robot

MEN = Path(__file__).resolve().parents[1] / "third_party" / "mujoco_menagerie"


def _model(rel):
    p = MEN / rel
    if not p.exists():
        pytest.skip(f"{rel} not fetched (scripts/fetch_assets.sh [--extra])")
    return p


def test_panda_is_a_fixed_arm_with_a_gripper():
    p = profile_robot(_model("franka_emika_panda/panda.xml"))
    assert p.kind == "fixed_arm" and p.base == "fixed" and p.n_arms == 1 and p.n_legs == 0
    assert p.gripper["type"] == "parallel_gripper"
    assert p.gripper["max_opening_m"] == pytest.approx(0.08, abs=0.005)  # Franka hand: 80 mm
    reach = max(v["max_horizontal"] for v in p.reach_m.values())
    assert 0.8 < reach < 0.95  # spec 855 mm to the flange
    caps = capabilities_from_profile(p)
    assert "grasp" in caps.contact_modes and caps.object_workspace["type"] == "annulus"


def test_xarm7_linkage_gripper_is_a_gripper_not_a_hand():
    p = profile_robot(_model("ufactory_xarm7/xarm7.xml"))
    assert p.kind == "fixed_arm" and p.gripper["type"] == "parallel_gripper"
    assert 0.07 < p.gripper["max_opening_m"] < 0.10  # spec ~85 mm


def test_go1_is_a_quadruped_that_can_only_push():
    p = profile_robot(_model("unitree_go1/go1.xml"))
    assert p.kind == "quadruped" and p.base == "floating" and p.n_legs == 4 and not p.has_gripper
    caps = capabilities_from_profile(p)
    assert caps.contact_modes == ["push"] and not caps.prehensile


@pytest.mark.parametrize("rel,kind", [
    ("unitree_h1/h1.xml", "humanoid"),
    ("boston_dynamics_spot/spot_arm.xml", "legged_manipulator"),
    ("skydio_x2/x2.xml", "aerial"),
    ("hello_robot_stretch/stretch.xml", "mobile_manipulator"),
    ("unitree_go2/go2.xml", "quadruped"),
    ("agilex_piper/piper.xml", "fixed_arm"),
])
def test_unseen_robots_are_classified(rel, kind):
    assert profile_robot(_model(rel)).kind == kind


def test_small_gripper_cannot_grasp_the_block():
    p = profile_robot(_model("ufactory_lite6/lite6_gripper_wide.xml"))
    caps = capabilities_from_profile(p)  # 1.8 cm opening vs a 5 cm block
    assert not caps.prehensile and any("smaller than the object" in n for n in caps.notes)
