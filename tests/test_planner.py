from pathlib import Path

import pytest

from morph.introspect import profile_robot
from morph.planner import check_templates, plan_context, rule_plan
from morph.task import default_spec

MEN = Path(__file__).resolve().parents[1] / "third_party" / "mujoco_menagerie"
SPEC = default_spec(0.40, 0.12, 1.8)


@pytest.fixture(autouse=True)
def _no_trained_walkers(tmp_path, monkeypatch):
    # plans shouldn't depend on which walking policies happen to be in artifacts/walk
    import morph.planner
    monkeypatch.setattr(morph.planner, "WALK_DIR", tmp_path)


def _plan(rel):
    p = MEN / rel
    if not p.exists():
        pytest.skip(f"{rel} not fetched (scripts/fetch_assets.sh [--extra])")
    prof = profile_robot(p)
    checks = check_templates(prof, SPEC)
    return prof, checks, rule_plan(checks)


@pytest.mark.parametrize("rel,plan", [
    ("franka_emika_panda/panda.xml", "grasp_place_topdown"),
    ("agilex_piper/piper.xml", "grasp_place_topdown"),
    ("unitree_go1/go1.xml", "legged_foot_push"),
    ("unitree_go2/go2.xml", "learn_policy"),  # same shape as the Go1, but the gait policy isn't its own
    ("trs_so_arm100/so_arm100.xml", "grasp_place_topdown"),  # a jaw closing on a fixed jaw grips too
    ("skydio_x2/x2.xml", "learn_policy"),  # no gripper or legs, but it can still move into the block
])
def test_rule_plan(rel, plan):
    assert _plan(rel)[2]["plan_id"] == plan


def test_unmet_preconditions_are_named():
    _, checks, _ = _plan("unitree_go2/go2.xml")
    legged = next(c for c in checks if c.id == "legged_foot_push")
    assert not legged.applicable and any("walking policy" in m for m in legged.missing)


def test_trained_walker_switches_the_plan(tmp_path):
    (tmp_path / "go2").mkdir()
    (tmp_path / "go2" / "walk_policy.npz").write_bytes(b"")
    assert _plan("unitree_go2/go2.xml")[2]["plan_id"] == "legged_foot_push"


def test_plan_context_is_serialisable():
    import json
    prof, checks, _ = _plan("agilex_piper/piper.xml")
    ctx = plan_context(prof, SPEC, checks)
    assert json.loads(json.dumps(ctx, default=str))["task"]["goal_yaw_deg"] == pytest.approx(103.1, abs=0.1)
