import math
from pathlib import Path

import numpy as np
import pytest

from morph.goal_sources import goal_from_demo, goal_from_description_rules, load_demo, merge
from morph.goals import check, placement, validate
from morph.user_scene import load_scene

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "examples" / "parcel" / "scene.xml"


@pytest.fixture(scope="module")
def scene():
    return load_scene(SCENE)


def test_fixtures_are_classified_by_shape_not_name(scene):
    f = scene.fixtures
    assert f["bin"].kind == "container" and f["bin"].rest_z == pytest.approx(0.01)
    assert f["bin"].inner_hi[0] - f["bin"].inner_lo[0] == pytest.approx(0.14, abs=1e-3)
    assert f["shelf"].kind == "surface" and f["shelf"].rest_z == pytest.approx(0.05)
    assert f["wall"].kind == "obstacle"
    b = scene.block
    assert (2 * b.half_x, 2 * b.half_z) == pytest.approx((0.05, 0.06)) and b.mass == pytest.approx(0.2)
    assert '<body name="bin">' in b.fixtures_xml


@pytest.mark.parametrize("text,kinds", [
    ("put the parcel into the bin", {("inside", "bin")}),
    ("place it on the shelf facing north", {("on_top_of", "shelf"), ("heading", 90.0)}),
    ("push the parcel against the wall", {("touching", "wall")}),
    ("put the parcel next to the shelf", {("near", "shelf")}),
    ("put the parcel north of the shelf", {("beside", "shelf")}),
    ("move it to the bin", {("inside", "bin")}),  # no preposition: what a container is for
    ("put it on the bin", {("inside", "bin")}),  # "on" a container means in it
    ("put the box at (0.40, 0.10) within 3 cm", {("at_position", None)}),
])
def test_description_rules(scene, text, kinds):
    g = goal_from_description_rules(text, scene)
    got = {(c.kind, c.target if c.kind != "heading" else c.yaw_deg) for c in g.conditions if c.kind != "upright"}
    assert got == kinds
    assert not validate(g, scene)


def test_impossible_goals_are_explained_physically(scene):
    from morph.goals import Condition, GoalSpec
    g = goal_from_description_rules("put the parcel in the shelf", scene)
    assert any("no inside" in e for e in validate(g, scene))
    from morph.goals import warnings
    wall = GoalSpec(object="parcel", conditions=[Condition(kind="on_top_of", target="wall")])
    assert not validate(wall, scene)  # possible, just very unlikely: attempted with a warning
    assert any("only 1 cm wide" in w and "still try" in w for w in warnings(wall, scene))
    rim = GoalSpec(object="parcel", conditions=[Condition(kind="on_top_of", target="bin")])
    assert any("say \"in\"" in w for w in warnings(rim, scene))


def test_near_and_beside_are_placed_where_they_are_checked(scene):
    from morph.goals import Condition, GoalSpec
    from morph.sim import BlockPose
    for c in (Condition(kind="near", target="shelf", tol=0.02), Condition(kind="beside", target="shelf", side="south"),
              Condition(kind="near", target="bin", tol=0.1)):
        g = GoalSpec(object="parcel", conditions=[c])
        p = placement(g, scene)
        r = check(g, scene, None, None, BlockPose(p["x"], p["y"], 0.03, 0.0, 0.0), 0.0, True)
        assert r["success"], (c, p, r)


def test_placement_knows_what_has_to_be_lifted(scene):
    p = placement(goal_from_description_rules("put the parcel into the bin", scene), scene)
    assert (p["x"], p["y"]) == pytest.approx((0.42, 0.18)) and p["support_z"] == pytest.approx(0.01)
    assert p["needs_lift"] and p["clear_z"] == pytest.approx(0.05)
    p = placement(goal_from_description_rules("push the parcel against the wall", scene), scene)
    assert not p["needs_lift"] and p["x"] == pytest.approx(0.795 - 0.025 - 0.003) and p["y"] == pytest.approx(-0.18)


def _path(end_xyz, end_yaw_deg, start=(0.45, -0.18, 0.03)):
    pos = np.linspace(start, end_xyz, 20)
    pos[5:15, 2] += 0.08  # lifted on the way
    yaw = np.radians(np.linspace(0, end_yaw_deg, 20))
    quat = np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], 1)
    return {"pos": pos, "quat": quat, "source": "test"}


def test_demo_is_measured_into_conditions(scene):
    g = goal_from_demo(_path((0.42, 0.18, 0.04), 180), scene)
    kinds = {c.kind for c in g.conditions}
    assert {"inside", "heading", "upright"} <= kinds and "at_position" not in kinds
    g = goal_from_demo(_path((0.30, 0.00, 0.03), 5), scene)  # ends on the floor, barely turned
    assert {c.kind for c in g.conditions} == {"at_position", "upright"}


def test_description_overrides_the_demo(scene):
    demo = goal_from_demo(_path((0.42, 0.18, 0.04), 180), scene)
    text = goal_from_description_rules("put it on the shelf", scene)
    kinds = {(c.kind, c.target) for c in merge(demo, text).conditions}
    assert ("on_top_of", "shelf") in kinds and ("inside", "bin") not in kinds and ("heading", None) in kinds


def test_joint_log_demo_is_replayed(tmp_path, scene):
    d = _path((0.42, 0.18, 0.04), 90)
    f = tmp_path / "log.npz"
    np.savez(f, qpos=np.hstack([d["pos"], d["quat"]]), model=str(SCENE), object="parcel")
    demo = load_demo(f)
    assert demo["pos"][-1] == pytest.approx([0.42, 0.18, 0.04], abs=1e-6)
    assert any(c.kind == "inside" for c in goal_from_demo(demo, scene).conditions)


def test_check_measures_each_condition(scene):
    from morph.sim import BlockPose
    g = goal_from_description_rules("put the parcel into the bin facing west", scene)
    inside = BlockPose(0.42, 0.19, 0.04, math.pi, 0.0)
    r = check(g, scene, None, None, inside, 0.0, True)
    assert r["success"], r["reasons"]
    outside = BlockPose(0.42, 0.30, 0.03, math.pi, 0.0)
    r = check(g, scene, None, None, outside, 0.0, True)
    assert not r["success"] and any("inside the bin" in x for x in r["reasons"])


def test_morph_do_puts_the_parcel_in_the_bin(tmp_path):
    panda = ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"
    if not panda.exists():
        pytest.skip("panda not fetched")
    from morph.project import load_project, resolve_goal, run_on_robot
    pj = load_project(ROOT / "examples" / "parcel")
    goal, _ = resolve_goal(pj, "put the parcel into the bin", None, None, log=lambda s: None)
    r = run_on_robot(pj, goal, str(panda), n=2, log=lambda s: None)
    assert r["plan"]["plan_id"] == "grasp_place_topdown" and r["success"] >= 1


def test_push_only_robot_refuses_a_lifting_goal():
    go2 = ROOT / "third_party" / "mujoco_menagerie" / "unitree_go2" / "go2.xml"
    if not go2.exists():
        pytest.skip("go2 not fetched")
    from morph.project import load_project, resolve_goal, run_on_robot
    pj = load_project(ROOT / "examples" / "parcel")
    goal, _ = resolve_goal(pj, "put the parcel on the shelf", None, None, log=lambda s: None)
    r = run_on_robot(pj, goal, str(go2), n=1, log=lambda s: None)
    assert r["plan"]["plan_id"] == "unsupported" and r["n"] == 0
