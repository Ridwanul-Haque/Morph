import json

import mujoco
import numpy as np

from morph.goal_terms import History, parse, violation
from morph.learn_task import _CondTrace, _condition_report, _ease, _handoffs, senses_for
from morph.task_model import finger_gaps, finger_points

from test_fingers import _hand


def test_spin_and_yes_no_have_their_own_units():
    v = {"touch": {"thumb": 1.0, "fingers_touching": 1.0}, "spin_dps": 30.0, "held": 0.0}
    assert round(float(violation(np, parse("spin_dps > 120", {}), v)), 3) == 1.0  # 90 deg/s short is 1, not 300
    assert round(float(violation(np, parse("touching('thumb') == 0", {}), v)), 2) == 0.99
    assert round(float(violation(np, parse("held == 1", {}), v)), 2) == 0.98


def test_a_contact_flicker_isnt_a_break():
    c = {"kind": "expression", "expr": "held == 1", "when": "always"}
    base = {"x": 0.0, "y": 0.0, "z": 0.1, "bottom": 0.1, "R": np.eye(3), "dist": {}}
    for gap, broken in ((2, False), (6, True)):
        h = History([c], {}, (0.01, 0.01, 0.01))
        for held in [1.0] * 5 + [0.0] * gap + [1.0] * 5:
            h.update({**base, "held": held})
        assert (h.result()[0] > 0) == broken


def test_condition_report_says_what_broke_and_when():
    c = {"kind": "expression", "expr": "bottom > 0.02", "when": "always", "text": "it never falls"}
    trs = []
    for fall_at in (10, 30):
        t = _CondTrace(c)
        for k in range(50):
            t.update({"bottom": 0.1 if k < fall_at else 0.0}, {}, (0.01, 0.01, 0.01))
        trs.append(t)

    class W:
        def __init__(self, t):
            self.cond = [t]
    (r,) = _condition_report([W(t) for t in trs])
    assert r["met_in"] == "0/2 replays"
    assert [b["at_s"] for b in r["first_broken"]] == [0.2, 0.6]
    assert r["values_seen (min, max)"]["bottom"] == (0.0, 0.1)


def test_easing_loosens_distances_not_rules(tmp_path):
    conds = [{"kind": "expression", "expr": "spin_dps > 180", "when": "last", "text": "it spins"},
             {"kind": "expression", "expr": "touching('thumb') == 0", "when": "always", "text": "no thumb"}]
    f = tmp_path / "task.json"
    f.write_text(json.dumps({"conditions": conds, "fixtures": {}}))
    diag = {"goal_conditions": [{"miss_each_replay (by its rule, same units)": [0.8, 1.0, 1.2]},
                                {"miss_each_replay (by its rule, same units)": [0.99, 0.99, 0.99]}]}
    msg = _ease(f, diag, {"success": 0}, log=lambda s: None)
    assert json.loads(f.read_text())["ease"] == {"slack": {"0": 1.0}}
    assert "90 deg/s" in msg and "thumb" not in msg
    _ease(f, None, {"success": 0}, log=lambda s: None)  # it improved: tighter
    assert json.loads(f.read_text())["ease"] == {"slack": {"0": 0.5}}
    _ease(f, None, {"success": 1}, log=lambda s: None)  # a real success: the real goal
    assert json.loads(f.read_text())["ease"] == {}


def test_a_step_that_ends_where_the_next_cant_continue_is_flagged():
    steps = [{"name": "grip it", "done_when": "held"}, {"name": "turn it", "done_when": "spin_dps > 60"}]

    class Tr:
        reached = 1

    class W:
        tracker, handoff, first, best = Tr(), {1: (None, 0)}, [None, 0.5], [0.0, 0.5]
    (h,) = _handoffs([W(), W()], steps)
    assert h["next_step_done_from_here"] == "0/2" and "step 1 ends somewhere" in h["verdict"]


def test_whole_finger_touch_and_senses():
    m = mujoco.MjModel.from_xml_string(_hand(("ff", "mf", "rf", "th")))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    body, off, rad, of, labels = finger_points(m)

    class Ix:
        finger_body, finger_off, finger_rad, finger_of, finger_labels = body, off, rad, of, labels
    b = m.body("block").id
    gaps = dict(zip(labels, finger_gaps(np, d.xpos, d.xmat, d.xpos[b], d.xmat[b].reshape(3, 3), (0.01, 0.01, 0.01), Ix)))
    assert gaps["middle"] < 0.006 < gaps["thumb"]  # the middle finger lies under the block; the thumb is well away


def test_senses_follow_the_goal(tmp_path):
    xml = tmp_path / "hand.xml"
    xml.write_text(_hand(("ff", "mf", "rf", "th")))
    task = {"robot_xml": str(xml), "conditions": [{"kind": "expression", "expr": "spin_dps > 60"},
                                                  {"kind": "expression", "expr": "touching('thumb') == 0"}]}
    assert senses_for(task) == ["spin", "touch"]
    assert senses_for({"robot_xml": str(xml), "conditions": [{"kind": "height", "min_z": 0.1}]}) == []
