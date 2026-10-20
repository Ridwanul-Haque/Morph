import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from morph.fingers import finger_names, hands, part_names, touching
from morph.goal_sources import goal_from_description_rules
from morph.goal_terms import describe_formula, missing_parts, parse, violation
from morph.user_scene import load_scene

MENAGERIE = Path(__file__).resolve().parents[1] / "third_party" / "mujoco_menagerie"


def _finger(name: str, pos: str, axis: str = "0 1 0") -> str:
    return f"""
      <body name="{name}_1" pos="{pos}">
        <joint name="{name}_j1" axis="{axis}" range="0 1.6"/>
        <geom type="capsule" fromto="0 0 0 0.04 0 0" size="0.008"/>
        <body name="{name}_2" pos="0.04 0 0">
          <joint name="{name}_j2" axis="{axis}" range="0 1.6"/>
          <geom type="capsule" fromto="0 0 0 0.03 0 0" size="0.008"/>
        </body>
      </body>"""


def _hand(names=("f1", "f2", "f3", "f4")) -> str:
    # four fingers side by side along the palm's edge, the last one (the thumb) well apart from them
    spots = ["0.05 -0.03 0", "0.05 0 0", "0.05 0.03 0", "0.0 -0.07 0"]
    fingers = "".join(_finger(n, p) for n, p in zip(names, spots))
    return f"""<mujoco><worldbody><geom name="floor" type="plane" size="1 1 0.1"/>
      <body name="palm" pos="0 0 0.3"><geom type="box" size="0.05 0.05 0.01"/>{fingers}</body>
      <body name="block" pos="0.07 0 0.315"><freejoint/><geom type="box" size="0.01 0.01 0.01"/></body>
    </worldbody></mujoco>"""


def test_fingers_found_and_labelled_by_name():
    m = mujoco.MjModel.from_xml_string(_hand(("ff", "mf", "rf", "th")))
    (h,) = finger_names(m)
    assert h["palm"] == "palm"
    assert {k: v[-1] for k, v in h["fingers"].items()} == {"index": "ff_2", "middle": "mf_2", "ring": "rf_2", "thumb": "th_2"}


def test_thumb_found_by_shape_when_names_dont_say():
    m = mujoco.MjModel.from_xml_string(_hand())
    (h,) = finger_names(m)
    assert h["fingers"]["thumb"][0] == "f4_1"  # the one set apart
    assert h["fingers"]["index"][0] == "f1_1"  # the nearest to the thumb


def test_a_parallel_gripper_is_not_a_hand():
    xml = """<mujoco><worldbody><body name="hand">
      <body name="left"><joint type="slide" axis="0 1 0" range="0 0.04"/><geom type="box" size="0.01 0.01 0.03"/></body>
      <body name="right"><joint type="slide" axis="0 -1 0" range="0 0.04"/><geom type="box" size="0.01 0.01 0.03"/></body>
    </body></worldbody></mujoco>"""
    assert hands(mujoco.MjModel.from_xml_string(xml)) == []


def test_touching_reads_the_contacts_per_finger():
    m = mujoco.MjModel.from_xml_string(_hand(("ff", "mf", "rf", "th")))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    t = touching(m, d)
    hit = {k for k, v in t.items() if v and k != "fingers_touching"}
    assert "middle" in hit and "thumb" not in hit
    assert t["fingers_touching"] == len(hit & {"thumb", "index", "middle", "ring"})


def test_touching_in_formulas():
    v = {"touch": {"thumb": 0.0, "index": 1.0, "fingers_touching": 1.0}}
    assert float(violation(np, parse("touching('thumb') == 0", {}), v)) == 0.0
    assert float(violation(np, parse("touching('thumb')", {}), v)) > 0
    assert float(violation(np, parse("touching('index|middle')", {}), v)) == 0.0
    assert math.isnan(float(violation(np, parse("touching('little')", {}), v)))  # not on this robot: never met
    assert describe_formula("touching('thumb') == 0", "pencil") == "the thumb not touching the pencil"


def test_goal_names_a_finger_the_robot_lacks():
    names = part_names(mujoco.MjModel.from_xml_string(_hand(("ff", "mf", "rf", "th"))), "")
    conds = [{"kind": "expression", "expr": "touching('thumb') == 0"}, {"kind": "expression", "expr": "touching('little')"}]
    assert missing_parts(conds, names) == ["little"]


def test_in_hand_requests_start_in_the_hand():
    scene = load_scene(Path(__file__).resolve().parents[1] / "examples" / "pencil" / "scene.xml")
    g = goal_from_description_rules("spin the pencil around the fingers without using the thumb", scene)
    assert g.start == "in_hand"
    assert {c.expr for c in g.conditions} == {"spin_dps > 180", "touching('thumb') == 0"}
    assert [c.when for c in g.conditions if "thumb" in c.expr] == ["always"]
    assert goal_from_description_rules("push the pencil 10 cm east", scene).start == "scene"


@pytest.mark.skipif(not (MENAGERIE / "shadow_hand").exists(), reason="scripts/fetch_assets.sh --extra")
def test_shadow_hand_is_a_hand_and_holds_a_pencil_without_the_thumb(tmp_path):
    from morph.introspect import profile_robot
    from morph.learn_task import make_task
    from morph.project import load_project
    xml = str(MENAGERIE / "shadow_hand" / "right_hand.xml")
    prof = profile_robot(xml)
    assert prof.kind == "hand" and prof.n_arms == 0
    assert set(prof.gripper["finger_names"]) == {"thumb", "index", "middle", "ring", "little"}
    pj = load_project(Path(__file__).resolve().parents[1] / "examples" / "pencil")
    goal = goal_from_description_rules("spin the pencil around the fingers without using the thumb", pj.scene)
    task = make_task(pj, goal, xml, plan=False, folder=tmp_path / "shadow")
    import json
    t = json.loads(task.read_text())
    assert t["start_state"] and t["mount_pose"]
    from morph.task_model import build_task_model, contact_touch, indices
    m = build_task_model(t, accurate=True)
    ix = indices(m, t)
    d = mujoco.MjData(m)
    d.qpos[:] = t["start_state"]["qpos"]
    d.ctrl[:] = t["start_state"]["ctrl"]
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    assert d.xpos[ix.block][2] > 0.1  # still up in the hand
    assert contact_touch(m, d, ix)["thumb"] == 0.0


def test_a_segment_called_middle_doesnt_make_every_finger_the_middle_one():
    # a LinkerHand names each finger's middle segment "<finger>_middle"
    xml = _hand(("index", "middle", "ring", "thumb")).replace('name="index_2"', 'name="index_middle"') \
        .replace('name="ring_2"', 'name="ring_middle"')
    (h,) = finger_names(mujoco.MjModel.from_xml_string(xml))
    assert {k: v[0] for k, v in h["fingers"].items()} == {"index": "index_1", "middle": "middle_1", "ring": "ring_1",
                                                         "thumb": "thumb_1"}


def test_spinning_the_whole_time_is_impossible_from_rest():
    from morph.goal_terms import needs_motion_at_rest
    assert needs_motion_at_rest({"kind": "expression", "expr": "spin_dps > 60"})
    assert not needs_motion_at_rest({"kind": "expression", "expr": "touching('thumb') == 0"})
    assert not needs_motion_at_rest({"kind": "expression", "expr": "speed < 0.1"})
