"""Object shapes, spin, "keep it up for the last N seconds", and Nemotron's reward terms (checked formulas)."""
from pathlib import Path

import numpy as np

from morph.goal_terms import History, parse, term_value
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
