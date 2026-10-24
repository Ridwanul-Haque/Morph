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


def test_a_request_can_say_which_fingers_it_starts_on():
    scene = load_scene(Path(__file__).resolve().parents[1] / "examples" / "pencil" / "scene.xml")
    g = goal_from_description_rules("pass the pencil between the fingers. it starts between the index and middle "
                                    "fingers and ends between the middle and ring fingers", scene)
    assert g.start == "in_hand" and g.start_fingers == ["index", "middle"]
    assert "index and middle" in g.start_text()
    assert goal_from_description_rules("spin the pencil around the fingers", scene).start_fingers == []
    from morph.goal_sources import start_fingers
    assert start_fingers("start with it on the ring and pinky fingers") == ["ring", "little"]


def test_in_the_hand_isnt_gripped():
    """A pass or a roll loosens the pinch for a moment: "never falls out of the hand" as held == 1 the whole time
    punished the LinkerHand for the very motion it had to learn. in_hand only asks that nothing but the robot keeps it
    up."""
    from morph.goal_terms import quantities
    half = (0.07, 0.004, 0.004)
    q = lambda z, held: quantities(np, pos=np.array([0.0, 0.0, z]), R=np.eye(3), vel=np.zeros(3), start=(0, 0),  # noqa: E731
                                   half=half, fixtures={}, held=held)
    up, floor = q(0.25, 0.0), q(0.004, 0.0)
    assert up["in_hand"] == 1.0 and up["held"] == 0.0  # rolling between the fingers: in the hand, not pinched
    assert floor["in_hand"] == 0.0  # fallen


@pytest.mark.skipif(not (MENAGERIE / "shadow_hand").exists(), reason="scripts/fetch_assets.sh --extra")
def test_it_starts_on_the_fingers_the_request_names(tmp_path):
    from morph.goals import Condition, GoalSpec
    from morph.learn_task import make_task
    from morph.project import load_project
    from morph.task_model import build_task_model, contact_touch, indices
    import json
    pj = load_project(Path(__file__).resolve().parents[1] / "examples" / "pencil")
    goal = GoalSpec(object=pj.scene.object_name, start="in_hand", start_fingers=["index", "middle"],
                    conditions=[Condition(kind="expression", expr="in_hand == 1", when="always")])
    t = json.loads(make_task(pj, goal, str(MENAGERIE / "shadow_hand" / "right_hand.xml"), plan=False,
                             folder=tmp_path / "shadow").read_text())
    m = build_task_model(t, accurate=True)
    ix = indices(m, t)
    d = mujoco.MjData(m)
    d.qpos[:] = t["start_state"]["qpos"]
    d.ctrl[:] = t["start_state"]["ctrl"]
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    touch = contact_touch(m, d, ix)
    assert touch["index"] == 1.0 and touch["middle"] == 1.0  # (with others too, if only they hold it still)


def test_it_starts_where_the_goal_first_needs_it():
    """"Lying across the fingers" named none, so the LinkerHand started in a near-fist over the palm and never worked
    the pen onto the index and middle fingers the goal asked for next."""
    from morph.in_hand import fingers_needed_first
    labels = ["thumb", "index", "middle", "ring", "little"]
    conds = [{"when": "always", "expr": "in_hand == 1"}, {"when": "always", "expr": "touching('thumb') == 0"},
             {"when": "ever", "expr": "spin_z_dps > 90"},
             {"when": "ever", "expr": "touching('index') == 1 && touching('middle') == 1", "text": "between index and middle"},
             {"when": "last", "expr": "touching('middle') == 1 and touching('ring') == 1"}]
    assert fingers_needed_first(conds, labels) == (["index", "middle"], "between index and middle")
    assert fingers_needed_first(conds[:3] + conds[4:], labels) == ([], "")  # not the end's: holding still would pass it
    assert fingers_needed_first([conds[1], {"when": "ever", "expr": "touching('thumb') == 1 and touching('pinky') > 0.5"}],
                                labels)[0] == ["little"]  # never a finger kept off it the whole time
    assert fingers_needed_first([{"when": "ever", "expr": "touching('ring') == 1 or touching('index') == 1"}], labels)[0] == []


def test_a_pen_kept_in_the_hand_isnt_judged_as_set_down():
    """With "never falls out of the hand" as in_hand, no condition said held, so Morph took it the hand had to let go:
    6 of 10 LinkerHand tries met every condition and failed for the pen still wobbling between the fingers."""
    from morph.goals import Condition, GoalSpec, lets_go
    keep = Condition(kind="expression", expr="in_hand == 1", when="always")
    assert not lets_go(GoalSpec(object="pen", conditions=[keep], start="in_hand"))
    assert lets_go(GoalSpec(object="pen", conditions=[Condition(kind="inside", target="bin")]))


def test_a_spin_written_any_way_is_a_spin():
    """Nemotron's "abs(spin_z_dps) > 180 for the last 4 s" was turned down three times as not a spin, and the built-in
    reading that replaced it had no "stays in the hand", so the pen was also judged as set down: it had to be both
    spinning and still at the end."""
    from morph.goal_sources import goal_from_description_rules
    from morph.goals import Condition, GoalSpec
    from morph.project import load_project, missing_intent
    req = ("Spin the pen in the hand. Every try starts with the pen already in the hand. The pen must "
           "never fall out of the hand. For the last 4 seconds it turns faster than 180 degrees per second.")
    for e in ("abs(spin_z_dps) > 180", "spin_z_dps > 180", "spin_dps >= 90"):
        g = GoalSpec(object="pen", conditions=[Condition(kind="expression", when="always", expr="in_hand == 1"),
                                               Condition(kind="expression", when="last", seconds=4, expr=e)])
        assert missing_intent(req, g) == []
    pj = load_project(Path(__file__).resolve().parents[1] / "examples" / "parcel")
    rules = goal_from_description_rules(req.replace("pen", "parcel"), pj.scene)
    assert ("always", "in_hand == 1") in [(c.when, c.expr) for c in rules.conditions]
