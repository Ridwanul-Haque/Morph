from pathlib import Path

import pytest

from morph.arm_skills import expert_action, run_episode
from morph.learning import sample_scenario
from morph.robots.auto_arm import NotAGraspingArm, auto_arm_config, build_auto_arm
from morph.task import default_spec

MEN = Path(__file__).resolve().parents[1] / "third_party" / "mujoco_menagerie"


def _model(rel):
    p = MEN / rel
    if not p.exists():
        pytest.skip(f"{rel} not fetched (scripts/fetch_assets.sh [--extra])")
    return p


def test_panda_config_is_derived_from_the_model():
    cfg, rep = auto_arm_config(_model("franka_emika_panda/panda.xml"))
    assert len(cfg.arm_joints) == 7 and rep["gripper"] == "parallel_gripper"
    assert rep["opening_m"] == pytest.approx(0.08, abs=0.005)
    # grasp point between the finger pads, below the hand body (hand-written config: 0.103 = fingertips)
    assert cfg.tcp_body == "hand" and 0.07 < rep["tcp_pos"][2] < 0.11


def test_arm_without_gripper_is_rejected():
    with pytest.raises(NotAGraspingArm):
        auto_arm_config(_model("universal_robots_ur5e/ur5e.xml"))


def test_unseen_small_arm_picks_and_places():
    """The Piper was never configured by hand: mount, carry height, servo lead and grip strength all
    come from experiments in build_auto_arm."""
    ad, rep = build_auto_arm(_model("agilex_piper/piper.xml"))
    assert rep["travel_z"] < 0.15  # small arm: lower carry height than the default
    ok = n = 0
    for i in range(40):
        sc = sample_scenario(10000 + i)
        if not (ad.reach_ok(*sc.block[:2]) and ad.reach_ok(*sc.goal[:2])):
            continue
        ad.reset(i, sc.block, sc.params)
        ok += run_episode(ad, default_spec(*sc.goal), expert_action).outcome.success
        n += 1
        if n == 5:
            break
    assert n == 5 and ok >= 4
