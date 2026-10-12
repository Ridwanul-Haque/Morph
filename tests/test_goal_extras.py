"""Object shapes, spin, "keep it up for the last N seconds", and Nemotron's reward terms (checked formulas)."""
from pathlib import Path

import numpy as np

from morph.goal_terms import History, parse, term_value, violation
from morph.goals import Condition
from morph.project import load_project
from morph.skill_steps import check_terms

ROOT = Path(__file__).resolve().parents[1]


def test_a_ball_is_simulated_as_a_ball():
    from morph.learning import sample_scenario
    from morph.scene import _block_xml  # noqa: F401
    from morph.sim import SimulationEnvironment
    from morph.task_model import write_task, build_task_model
    pj = load_project(ROOT / "examples" / "ball")
    b = pj.scene.block
    assert b.shape == "sphere" and b.half_x == b.half_z == 0.03
    assert pj.scene.geometry()["object"]["shape"] == "sphere"
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        task = write_task(Path(td) / "task.json", robot_xml=str(ROOT / "tests" / "data" / "tiny_arm.urdf"), block=b,
                          start=pj.scene.start, target={"x": 0, "y": 0, "z": 0, "yaw": 0}, mount=(0, 0, 0), floating=False,
                          q0=[], conditions=[], scene_path=pj.scene.path, robot_name="t")
        assert task["block"]["shape"] == "sphere"
    import mujoco
    from morph.robot_import import import_robot
    xml, _ = import_robot(str(ROOT / "tests" / "data" / "tiny_arm.urdf"))
    task.update(robot_xml=str(xml))
    m = build_task_model(task)
    assert int(m.geom_type[m.geom("block_geom").id]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
    env = SimulationEnvironment(m)
    env.reset(0, pj.scene.start, sample_scenario(0).params)
    assert abs(env.data.qpos[m.joint("block_free").qposadr[0] + 2] - 0.03) < 1e-9  # resting on the floor, not sunk in it


def test_keep_it_up_for_the_last_seconds():
    c = Condition(kind="expression", expr="spin_dps > 180", text="spinning", when="last", seconds=1.0).model_dump()
    h = History([c], {}, (0.03, 0.03, 0.03))
    base = {"x": 0.0, "y": 0.0, "z": 0.03, "held": 0.0, "R": np.eye(3), "dist": {}}
    for k in range(200):  # spins for the first 3 s, then slows down for the last 1 s
        h.update(base | {"spin_dps": 360.0 if k < 150 else 90.0})
    assert h.result()[0] > 0  # it stopped keeping it up
    h = History([c], {}, (0.03, 0.03, 0.03))
    for k in range(200):  # slow at first, spinning for the whole last second
        h.update(base | {"spin_dps": 90.0 if k < 100 else 360.0})
    assert h.result()[0] == 0
    assert Condition(**c).describe().startswith("for the last 1 s")


def test_reward_terms_are_numbers_or_graded_conditions():
    v = {"touch_dist": 0.01, "closure": 0.8, "speed": 2.0}
    assert term_value(np, parse("closure * (touch_dist < 0.03)", {}), v) == 0.8
    assert term_value(np, parse("-speed", {}), v) == -2.0
    assert term_value(np, parse("speed < 1", {}), v) < 0  # a condition: minus how far from true
    terms, errors = check_terms([{"name": "squeeze at it", "formula": "closure * (touch_dist < 0.03)", "weight": 9},
                                 {"name": "bad", "formula": "__import__('os')"}, {"formula": "wobble > 2"}], {})
    assert [t["name"] for t in terms] == ["squeeze at it"] and terms[0]["weight"] == 5.0  # weight clipped to 5
    assert len(errors) == 2


def test_goals_can_be_about_the_robots_own_body():
    """'make a fist', 'raise the arm': joint_frac / part_z over the robot's joints and parts, not the object."""
    from morph.goal_terms import violation
    from morph.goals import GoalSpec, about_body, unsatisfiable
    v = {"joints": {"index_pip": 0.9, "middle_pip": 0.95, "index_mcp_roll": 0.5}, "parts": {"index_tip": np.array([0, 0, 0.3])}}
    assert violation(np, parse("joint_frac_min('pip') > 0.85", {}), v) == 0
    assert violation(np, parse("joint_frac('pip') > 0.95", {}), v) > 0
    assert violation(np, parse("part_z('tip') > 0.25", {}), v) == 0
    assert np.isnan(violation(np, parse("joint_frac('elbow') > 0.5", {}), v))  # this robot has no elbow: not "met"
    c = Condition(kind="expression", expr="joint_frac_min('pip') > 0.85", text="a fist")
    assert about_body(c)
    pj = load_project(ROOT / "examples" / "parcel")
    assert unsatisfiable(GoalSpec(object="parcel", conditions=[c]), pj.scene) == []  # not judged from the scene


def test_a_body_goal_is_checked_on_the_robots_real_joints(tmp_path, monkeypatch):
    import mujoco
    import morph.learn_task as lt
    from morph.goals import GoalSpec, check
    from morph.learn_task import make_task
    from morph.task_model import load_task, build_task_model
    monkeypatch.setattr(lt, "LEARNED", tmp_path)
    pj = load_project(ROOT / "examples" / "parcel")
    goal = GoalSpec(object="parcel", conditions=[Condition(kind="expression", expr="joint_frac('joint') > 0.9",
                                                           text="every joint near its upper limit")])
    task = load_task(make_task(pj, goal, str(ROOT / "tests" / "data" / "tiny_arm.urdf"), plan=False))
    m = build_task_model(task)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    from morph.sim import BlockPose  # noqa: F401
    pose = type("P", (), {"x": 0.45, "y": -0.18, "z": 0.03, "yaw": 0.0, "tilt": 0.0})()
    r = check(goal, pj.scene, m, d, pose, 0.0, True)
    assert not r["success"] and "joint_frac('joint')" in r["results"][0]["measured"]
    for j in range(m.njnt):
        if m.joint(j).name.startswith("robot_") and m.jnt_limited[j]:
            d.qpos[m.jnt_qposadr[j]] = m.jnt_range[j][1]
    mujoco.mj_forward(m, d)
    assert check(goal, pj.scene, m, d, pose, 0.0, True)["success"]


def test_body_patterns_take_words_or_wildcards():
    from morph.goal_terms import body_names
    names = ["index_mcp_pitch", "index_mcp_roll", "index_pip", "thumb_ip"]
    assert body_names("pip|thumb", names) == ["index_pip", "thumb_ip"]
    assert body_names("*_mcp_pitch|*_ip", names) == ["index_mcp_pitch", "thumb_ip"]
    assert body_names("index_*", names) == ["index_mcp_pitch", "index_mcp_roll", "index_pip"]


def test_body_patterns_can_be_regular_expressions():
    from morph.goal_terms import body_names
    names = ["thumb_mcp", "index_mcp_roll", "index_mcp_pitch", "index_pip", "middle_mcp_roll"]
    assert body_names("(index|middle)_(mcp_pitch|pip)|thumb_mcp", names) == ["thumb_mcp", "index_mcp_pitch", "index_pip"]


def _box_stl(path: Path, size_mm=(100.0, 20.0, 10.0)):
    """A binary STL box (millimetres, as CAD tools export), standing on its long end."""
    import struct
    sx, sy, sz = size_mm[2], size_mm[1], size_mm[0]  # long side along z in the file: Morph must lay it down
    v = [(x, y, z) for x in (0, sx) for y in (0, sy) for z in (0, sz)]
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4),
             (1, 5, 7), (1, 7, 3)]
    with open(path, "wb") as f:
        f.write(b"\0" * 80 + struct.pack("<I", len(faces)))
        for a, b, c in faces:
            f.write(struct.pack("<3f", 0, 0, 0) + b"".join(struct.pack("<3f", *v[i]) for i in (a, b, c)) + b"\0\0")


def test_an_object_from_cad_comes_in_lying_flat_in_metres(tmp_path):
    from morph.object_import import object_scene
    from morph.sim import PhysicalParams, SimulationEnvironment
    from morph.app.preview import _scene_spec
    _box_stl(tmp_path / "Bar.STL")
    info = object_scene(tmp_path / "Bar.STL", out=tmp_path / "scene")
    assert info["units"] == "mm" and info["object"] == "bar"
    assert info["size_cm"] == [10.0, 2.0, 1.0]  # longest along x, shortest up: the way it rests
    assert abs(info["mass_kg"] - 0.02) < 0.002  # 20 cm3 at the density of water
    pj = load_project(tmp_path / "scene")
    b = pj.scene.block
    assert b.shape == "mesh" and abs(b.half_x - 0.05) < 1e-4 and abs(b.half_z - 0.005) < 1e-4
    env = SimulationEnvironment(_scene_spec(b).compile())
    env.reset(0, pj.scene.start, PhysicalParams())
    for _ in range(100):
        env.advance()
    p = env.block_pose()
    assert abs(p.z - 0.005) < 0.002 and env.block_speed() < 0.01  # resting on the floor, still
    assert abs(float(env.model.body_mass[env._block_body]) - b.mass) < 1e-6  # its own mass, not the parcel's 0.2 kg


def test_moment_snapshots_keep_body_measurements_without_crashing():
    """'held at some point' snapshots the measurements; the robot's joints / parts (collections) mustn't break it."""
    from morph.goal_terms import History
    c = Condition(kind="expression", expr="held == 1", text="held at some point", when="ever").model_dump()
    h = History([c], {}, (0.03, 0.03, 0.03))
    v = {"x": 0.4, "y": 0.0, "z": 0.1, "held": 1.0, "R": np.eye(3), "dist": {"bin": 0.2},
         "joints": {"j1": 0.5}, "parts": {"hand": np.array([0.4, 0.0, 0.1])}}
    h.update(v)
    assert h.result()[0] == 0 and h.snap[0]["joints"] == {"j1": 0.5} and "parts" not in h.snap[0]


def test_formulas_take_powers_maths_and_scene_names():
    """What Nemotron writes for shaping (all once rejected, each throwing away a whole plan): squared errors, exp / tanh,
    clip, angles, the goal's and fixtures' positions, roll and pitch."""
    import math
    fx = {"bin": {"lo": [0.34, 0.10, 0.0], "hi": [0.5, 0.26, 0.05], "top_z": 0.05}}
    from morph.goal_terms import quantities
    v = quantities(np, pos=np.array([0.30, 0.0, 0.05]), R=np.eye(3), vel=np.array([0.6, 0.8, 0.0]), start=(0.45, -0.18),
                   half=(0.05, 0.01, 0.01), fixtures=fx, held=0.0, hand_dist=np.asarray(0.1), goal=(0.42, 0.18, 0.02))
    val = lambda f: float(term_value(np, parse(f, fx), v))  # noqa: E731
    assert abs(val("(0.42 - x)**2") - 0.0144) < 1e-9 and abs(val("(hand_dist - 0.05)^2") - 0.0025) < 1e-9
    assert abs(val("exp(-hand_dist / 0.05)") - math.exp(-2)) < 1e-6 and val("clipped(speed, 0, 0.5)") == 0.5
    assert abs(val("bin_x") - 0.42) < 1e-9 and val("target_y") == 0.18 and abs(val("cosd(60)") - 0.5) < 1e-9
    assert val("roll_deg") == 0.0 and np.isfinite(val("log(0)")) and np.isfinite(val("hand_dist ** -4"))
    for bad in ("x ** y", "x ** 9", "eval(1)", "exp(x, y)"):
        try:
            parse(bad, fx)
        except ValueError:
            continue
        raise AssertionError(f"{bad} was accepted")
    # a goal with no place: goal_x isn't a number, so a condition on it isn't met (rather than crashing)
    v2 = quantities(np, pos=np.zeros(3), R=np.eye(3), vel=np.zeros(3), start=(0, 0), half=(0.03, 0.03, 0.03), fixtures={},
                    held=0.0)
    assert not violation(np, parse("abs(x - goal_x) < 0.05", {}), v2) == 0


def test_a_plan_is_kept_without_the_reward_term_morph_cant_measure():
    """One bad reward term used to throw a whole plan away (three times in one Stretch run)."""
    terms, errors = check_terms([{"name": "ok", "formula": "-(hand_dist - 0.05) ** 2", "weight": 1},
                                 {"name": "bad", "formula": "wobble_factor * 2", "weight": 1}], ["bin"])
    assert [t["name"] for t in terms] == ["ok"] and len(errors) == 1 and "wobble_factor" in errors[0]
