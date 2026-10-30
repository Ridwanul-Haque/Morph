"""Plans and goals that don't fit: rules that can never hold, steps for parts the robot doesn't have."""
from types import SimpleNamespace

from morph.goal_terms import never_holds
from morph.learn_task import _hard
from morph.skill_steps import Step, check_steps, finger_joints, has_part, parts_it_lacks


def _hand():
    """A five-finger hand fixed in place, as introspect profiles the LinkerHand."""
    acts = []
    for f in ("thumb", "index", "middle", "ring", "pinky"):
        acts += [{"name": f"{f}_mcp_roll", "target": f"{f}_mcp_roll", "ctrlrange": [-0.2, 0.2]},
                 {"name": f"{f}_mcp_pitch", "target": f"{f}_mcp_pitch", "ctrlrange": [0.0, 1.2]}]
    chains = [{"role": "finger", "leaf": f"{f}_distal", "bodies": [f"{f}_proximal", f"{f}_distal"]}
              for f in ("thumb", "index", "middle", "ring", "pinky")]
    return SimpleNamespace(actuators=acts, chains=chains, n_arms=0, n_legs=0, base="fixed")


def test_rules_that_can_never_hold():
    assert never_holds("abs(yaw_deg) >= 360") and "change(yaw_deg" in never_holds("abs(yaw_deg) >= 360")[0]
    assert never_holds("in_hand == 2") and never_holds("speed < 0 and z > 0.1")
    assert never_holds("mean(in_hand) > 1")
    assert not never_holds("yaw_deg > 200 or z > 0.1")  # one side of an "or" can still hold
    assert not never_holds("abs(change(yaw_deg, 4)) >= 720")  # a total turn has no limit
    assert not never_holds("spin_dps > 180") and not never_holds("touching('thumb') >= 1")


def test_a_step_that_can_never_be_done_is_sent_back():
    ok, errors = check_steps([Step(name="turn", done_when="abs(yaw_deg) >= 360"), Step(name="keep", done_when="in_hand == 1")],
                             {})
    assert len(ok) == 1 and "never hold" in errors[0]


def test_a_hand_fixed_in_place_has_no_wrist():
    hand = _hand()
    assert set(finger_joints(hand)) == {"thumb", "index", "middle", "ring", "little"}
    assert "swings sideways" in finger_joints(hand)["index"][0]
    assert not has_part(hand, "wrist") and has_part(hand, "thumb") and has_part(hand, "palm")
    wrist = Step(name="spin pen vertically", how="Rotate the wrist joint rapidly about the vertical axis",
                 body_parts=["wrist", "fingertips"], done_when="abs(spin_z_dps) > 180")
    assert parts_it_lacks(wrist, hand) == ["wrist"]
    fingers = Step(name="roll", how="Swing the index finger sideways without the wrist moving",
                   body_parts=["index finger", "thumb"], done_when="in_hand == 1")
    assert parts_it_lacks(fingers, hand) == []
    _, errors = check_steps([wrist], {}, "pen", hand)
    assert "wrist" in errors[0] and "doesn't have" in errors[0]


def test_which_goals_get_a_warm_up():
    assert _hard({"start_state": {"qpos": [0]}, "conditions": []})
    assert _hard({"conditions": [{"expr": "abs(change(yaw_deg, 4)) >= 720", "when": "end"}]})
    assert not _hard({"conditions": [{"kind": "inside", "when": "end"}]})


def test_a_warm_up_turn_counts_since_the_try_began():
    from morph.learn_task import since_start
    assert since_start("abs(change(yaw_deg, 2)) >= 180 and spin_z_dps * change(yaw_deg, 2) > 0") == \
        "abs(change(yaw_deg)) >= 180 and spin_z_dps * change(yaw_deg) > 0"
    assert since_start("mean(in_hand, 2) > 0.9") == "mean(in_hand, 2) > 0.9"


def test_exploring_more_stops_at_a_limit():
    from morph.learn_task import EXPLORE, _nudge
    more = dict(EXPLORE)["the current steps, exploring more"]
    assert _nudge({}, more) == {"entropy": 2.0}
    assert _nudge({"entropy": 2.0}, more) == {"entropy": 2.0}  # it compounded to 9 once
    assert _nudge({"entropy": 9.0}, {"entropy": 0.5}) == {"entropy": 2.0}


def test_moving_targets_follow_how_the_goal_counts_it():
    from morph.goal_terms import milestones_for

    def ms(expr):
        return milestones_for({"conditions": [{"kind": "expression", "expr": expr}]})
    assert ms("abs(spin_z_dps) > 180 and abs(change(yaw_deg, 4)) >= 720") == [{"of": "abs(change(yaw_deg))", "every": 30.0}]
    assert ms("change(x, 2) < -0.2") == [{"of": "-change(x)", "every": 0.025}]  # one way only: backwards
    assert ms("travelled >= 1.0") == [{"of": "travelled", "every": 0.025}]
    assert ms("abs(change(yaw_deg)) >= 40") == [{"of": "abs(change(yaw_deg))", "every": 20.0}]  # at least two
    assert ms("in_hand == 1") == [] and ms("change(spin_dps) > 9") == [] and ms("change(x) < 0.3") == []


def test_practice_is_plain_until_it_succeeds(tmp_path):
    import json
    from morph.learn_task import _vary_now, practice_aids
    task = {"conditions": [{"kind": "expression", "expr": "abs(change(yaw_deg, 4)) >= 720", "when": "last", "seconds": 4}]}
    practice_aids(task, log=lambda s: None)
    assert task["plain_practice"] and task["adaptive_lr"] and task["critic_milestones"] and task["milestones"]
    p = tmp_path / "task.json"
    p.write_text(json.dumps(task))
    assert _vary_now(p, log=lambda s: None) and not json.loads(p.read_text())["plain_practice"]
    assert not _vary_now(p, log=lambda s: None)  # only once


def test_between_the_fingers():
    import numpy as np
    from morph.task_model import between_terms
    pen, pair = (0.05, 0.006, 0.006), np.array([[0, 1]])

    def between(a, b):
        return float(between_terms(np, np.array([a, b], float), np.zeros(3), np.eye(3), pen, pair)[0])
    assert between([0.03, -0.02, 0], [0.03, 0.02, 0])  # either side, off its middle
    assert not between([0, 0.02, 0], [0, 0.06, 0])  # both on one side
    assert not between([0, 0.007, 0], [0, 0.05, 0.03])  # one touching it, the other away
    assert not between([0, 0.025, 0], [0.04, 0.025, 0])  # beside it: near isn't between


def test_the_step_before_the_grasp_needs_it_between_the_fingers():
    from morph.plan_check import between_first, require_grip
    task = {"gripper": {"actuator": "grip"}, "steps": [
        {"name": "down", "done_when": "touch_dist < 0.02 and closure < 0.1"},
        {"name": "close", "done_when": "grip_gap < 0.04 and held == 1"}]}
    assert between_first(task["steps"], task)
    assert "between" in require_grip(task) and task["steps"][0]["done_when"].endswith("between == 1")
    assert not between_first(task["steps"], task) and require_grip(task) is None


def test_rounds_are_ranked_by_the_steps_reached_too(tmp_path):
    import json
    from morph.learn_task import _fitness
    steps = [{"done_when": "hand_dist < 0.1"}, {"done_when": "held == 1"}]
    for name, done, dist in (("idle", 0.0, 148.0), ("grasping", 1.5, 157.0)):  # (nudging it left it further away)
        (tmp_path / name).mkdir()
        (tmp_path / name / "task.json").write_text(json.dumps({"steps": steps}))
        (tmp_path / name / "train_meta.json").write_text(json.dumps({"kept": {"steps_done": done, "dist": dist}}))
    res = {"success": 0}
    assert _fitness(tmp_path / "grasping", res) > _fitness(tmp_path / "idle", res)


def test_a_closed_gripper_beside_it_is_not_between():
    import numpy as np
    from morph.task_model import between_terms
    pen, pair = (0.0525, 0.00625, 0.00625), np.array([[0, 1]])

    def between(a, b):
        return float(between_terms(np, np.array([a, b], float) / 100, np.zeros(3), np.eye(3), pen, pair)[0])
    assert not between([3.5, 1.6, 0.2], [2.8, -0.3, 5.5])  # one tip on the floor beside it, the other above (cm)
    assert between([3.0, -3.0, 0.2], [3.0, 3.0, 0.2])  # the jaws open around it


def test_a_wobble_is_not_getting_better():
    from morph.learn_task import _improved
    last = (0, 0.0, 0.0, 0.17, -155.4)
    assert not _improved((0, 0.0, 0.0, 0.18, -153.9), last)  # 0.70 of a step against 0.68, 1.5 cm closer
    assert _improved((0, 0.0, 0.0, 0.3, -160.0), last)  # half a step further on
    assert _improved((1, 0.0, 0.0, 0.0, -170.0), last) and _improved((0, 0.0, 0.0, 0.17, -140.0), last)
    assert not _improved((0, 0.0, 0.0, 0.05, -100.0), last)  # closer, but by giving up steps
