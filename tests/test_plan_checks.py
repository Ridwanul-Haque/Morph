"""Checks on plans and rewards: a plan that stops short of the goal, a keep rule broken where its step starts, a reward
part the learner trades the task for, and a revision that has to practise better than the plan it replaces."""
import json
from pathlib import Path

import numpy as np

from morph.goal_sources import goal_from_description_rules
from morph.learn_task import make_task
from morph.plan_check import KeepCheck, unfinished
from morph.project import load_project
from morph.reward_parts import loophole
from morph.task_model import build_task_model, indices, load_task, obs_size

ROOT = Path(__file__).resolve().parents[1]
TINY = str(ROOT / "tests" / "data" / "tiny_arm.urdf")


def _policy(folder: Path, n_obs: int, n_act: int):
    rng = np.random.default_rng(0)
    np.savez(folder / "policy.npz", obs_mean=np.zeros(n_obs), obs_std=np.ones(n_obs),
             W0=rng.normal(0, 0.1, (n_obs, 2 * n_act)), b0=np.zeros(2 * n_act))


def test_a_plan_that_stops_short_of_the_goal():
    conds = [{"kind": "inside", "target": "bin", "when": "end"}, {"kind": "expression", "expr": "held == 0", "when": "end"},
             {"kind": "expression", "expr": "held == 1", "when": "ever"}]
    short = [{"name": "grab", "done_when": "held == 1"}, {"name": "lift", "done_when": "grip >= 0.35 and lifted >= 0.02"}]
    (p,) = unfinished(short, conds)
    assert "lift" in p and "held == 0" in p  # names the step and what it leaves out
    whole = short + [{"name": "drop it in", "done_when": "held == 0 and x > bin_x - 0.05 and x < bin_x + 0.05"}]
    assert unfinished(whole, conds) == []


def test_a_keep_rule_broken_where_its_step_starts(tmp_path, monkeypatch):
    import morph.learn_task as lt
    monkeypatch.setattr(lt, "LEARNED", tmp_path)
    pj = load_project(ROOT / "examples" / "parcel")
    goal = goal_from_description_rules("push the parcel against the wall", pj.scene)
    task = load_task(make_task(pj, goal, TINY, plan=False))
    check = KeepCheck(task)
    tight = [{"name": "come in close", "done_when": "hand_dist < 0.02", "keep": "hand_dist < 0.01"}]
    (p,) = check(tight)
    assert "come in close" in p and "hand_dist" in p
    fine = [{"name": "come in close", "done_when": "hand_dist < 0.02", "keep": "speed < 5"}]
    assert check(fine) == []


def test_a_part_paying_more_as_the_task_gets_worse():
    """The Stretch's round 3: the score fell, but one term rose from 1 to 7 while the time in the goal collapsed."""
    task = {"reward_terms": [{"name": "center", "formula": "x", "weight": 1.5}]}
    rec = lambda s, succ, term: {"step": s, "reward": 26 - s, "success": succ, "steps_done": 0.1,  # noqa: E731
                                 "dist": 70.0, "parts": {"success": succ / 16, "reach": 2.0}, "term_totals": [term]}
    hole = loophole([rec(0, 167, 1.0), rec(5, 60, 3.0), rec(10, 6, 7.3)], task)
    assert hole and hole["part"] == "term: center"
    assert loophole([rec(0, 167, 1.0), rec(10, 170, 7.3)], task) is None  # better at the task too: fine


def test_a_revision_is_kept_only_if_it_practises_better(tmp_path, monkeypatch):
    import morph.learn_task as lt
    monkeypatch.setattr(lt, "LEARNED", tmp_path)
    pj = load_project(ROOT / "examples" / "parcel")
    goal = goal_from_description_rules("push the parcel against the wall", pj.scene)
    tp = make_task(pj, goal, TINY, plan=False)
    m = build_task_model(load_task(tp))
    n_obs, n_act = obs_size(indices(m, load_task(tp))), m.nu
    _policy(tp.parent, n_obs, n_act)
    (tp.parent / "params.pkl").write_bytes(b"x")
    t = json.loads(tp.read_text())
    t["steps"] = [{"name": "touch", "done_when": "touch_dist < 0.05"}]
    tp.write_text(json.dumps(t))
    before = tp.read_text()

    def race(score_new):
        def train(p, n, restore):
            new = json.loads(p.read_text())["steps"][0]["done_when"] == "hand_x < x - 0.03"
            _policy(p.parent, n_obs, n_act)
            (p.parent / "train_meta.json").write_text(json.dumps({"curve": [{"success": score_new if new else 2.0,
                                                                              "dist": 1.0}]}))
            return True
        revised = dict(json.loads(before), steps=[{"name": "behind", "done_when": "hand_x < x - 0.03"}],
                       steps_from="nemotron, revised after round 1")
        tp.write_text(json.dumps(revised))
        return lt._tested_revision(pj, goal, tp, before, {"rationale": "get behind it"}, train, 1000, 1, [{"round": 1}],
                                   log=lambda s: None)

    assert not race(1.0)  # no better: the current plan stays
    assert load_task(tp)["steps"][0]["done_when"] == "touch_dist < 0.05"
    assert race(9.0)
    t = load_task(tp)
    assert t["steps"][0]["done_when"] == "hand_x < x - 0.03" and t["steps_from"] == "nemotron, revised after round 1"


def test_the_object_isnt_asked_to_be_at_the_goal_before_it_is_picked_up():
    """A revision of the Stretch's pen plan began "align above the pen: abs(x - goal_x) <= 0.02" with the gripper open:
    x is the pen's, so the pen had to be at the bin before it was picked up."""
    from morph.plan_check import before_the_grasp
    task = {"target": {"lift": True}, "fixtures": {"bin": {}}}
    bad = [{"name": "align", "done_when": "abs(x - goal_x) <= 0.02 and abs(y - goal_y) <= 0.02"},
           {"name": "grab", "done_when": "held == 1"}]
    (p,) = before_the_grasp(bad, task)
    assert "align" in p and "hand_x" in p
    good = [{"name": "reach", "done_when": "hypot(hand_x - x, hand_y - y) < 0.02"}, {"name": "grab", "done_when": "held == 1"},
            {"name": "carry", "done_when": "abs(x - goal_x) < 0.02 and held == 1"}]
    assert before_the_grasp(good, task) == []
    assert before_the_grasp(bad, {"target": {"lift": False}}) == []  # pushing it there is fine when it isn't carried


def test_a_place_it_must_be_carried_into_isnt_eased(tmp_path):
    """Missed by 29 cm, "in the bin" was met by nudging the pen 7 cm along the floor: practice learned to push it."""
    from morph.learn_task import _ease
    f = tmp_path / "task.json"
    conds = [{"kind": "inside", "target": "bin", "when": "end"}, {"kind": "expression", "expr": "spin_dps > 60", "when": "end"}]
    diag = {"goal_conditions": [{"miss_each_replay (by its rule, same units)": [0.97, 0.98]},
                                {"miss_each_replay (by its rule, same units)": [0.5, 0.6]}]}
    f.write_text(json.dumps({"conditions": conds, "fixtures": {}, "target": {"lift": True}}))
    _ease(f, diag, {"success": 0}, log=lambda s: None)
    assert list(json.loads(f.read_text())["ease"]["slack"]) == ["1"]  # the spin is eased, the bin isn't
    f.write_text(json.dumps({"conditions": conds, "fixtures": {}, "target": {"lift": False}}))
    _ease(f, diag, {"success": 0}, log=lambda s: None)
    assert set(json.loads(f.read_text())["ease"]["slack"]) == {"0", "1"}


def test_skills_for_different_objects_keep_their_own_folders(tmp_path, monkeypatch):
    """"Put it in the bin" for a pen and for a parcel once shared a folder: learning the pen overwrote the parcel."""
    import morph.learn_task as lt
    from morph.goals import Condition, GoalSpec
    monkeypatch.setattr(lt, "LEARNED", tmp_path)
    c = [Condition(kind="inside", target="bin")]
    pen, parcel = GoalSpec(object="pen", conditions=c), GoalSpec(object="parcel", conditions=c)
    assert lt.task_dir("stretch", pen) != lt.task_dir("stretch", parcel)
    old = tmp_path / f"stretch__{lt._goal_slug(parcel)}"  # learned before folders named the object
    old.mkdir()
    (old / "task.json").write_text(json.dumps({"scene": str(ROOT / "examples" / "parcel" / "scene.xml")}))
    assert lt._legacy_dir("stretch", parcel) == old
    assert lt._legacy_dir("stretch", pen) is None  # the parcel's skill isn't the pen's


def test_the_step_before_the_grasp_needs_the_hand_where_it_can_grip():
    """The Stretch's step before closing was "hand_dist <= 0.05 ...": done with its hand resting on top of the parcel,
    so practice learned to rest it there and closing never gripped anything."""
    from morph.plan_check import grip_first, require_grip
    steps = [{"name": "approach", "done_when": "hand_dist <= 0.06 and held == 0"},  # "not held" isn't the grasp
             {"name": "prepare", "done_when": "hand_dist <= 0.05 and closure < 0.2"},
             {"name": "close", "done_when": "held == 1"}, {"name": "lift", "done_when": "lifted > 0.05 and held"}]
    assert grip_first(steps, {}) == []  # no grip of Morph's to measure against
    task = {"grasp": {"body": "hand", "point": [0, 0, 0]}, "steps": [dict(s) for s in steps]}
    (p,) = grip_first(steps, task)
    assert "prepare" in p and "grip_gap" in p
    assert "prepare" in require_grip(task)
    assert task["steps"][1]["done_when"] == "(hand_dist <= 0.05 and closure < 0.2) and grip_gap < 0.03"
    assert task["steps"][0]["done_when"] == steps[0]["done_when"]
    assert require_grip(task) is None and grip_first(task["steps"], task) == []  # once is enough


def test_conditions_written_with_and_and_signs_can_be_explained():
    """Nemotron wrote "touching('index') == 1 && touching('middle') == 1": practice ran, then judging the tries crashed
    explaining what was measured, and the app stopped after round 1."""
    from morph.goal_terms import uses
    from morph.plan_check import _names
    e = "touching('index') == 1 && touching('middle') == 1 || !held"
    assert uses(e) == {"touching", "held"} and _names(e) == {"touching", "held"}


def test_goals_that_begin_alike_keep_their_own_skills(tmp_path, monkeypatch):
    """Folder names were cut to 40 letters: a thumb spin and a finger pass both began "the whole time: pen never falls
    out of the hand", and the spin was answered "learned" with the pass's skill. Skills from before keep working."""
    import morph.learn_task as lt
    from morph.goals import Condition, GoalSpec
    monkeypatch.setattr(lt, "LEARNED", tmp_path)
    keep = Condition(kind="expression", when="always", expr="in_hand == 1", text="pen never falls out of the hand")
    pass_ = GoalSpec(object="pen", conditions=[keep, Condition(kind="expression", when="ever", expr="touching('ring') == 1")])
    spin = GoalSpec(object="pen", conditions=[keep, Condition(kind="expression", when="ever", expr="spin_z_dps > 360")])
    assert lt.task_dir("hand", pass_) != lt.task_dir("hand", spin)
    old = tmp_path / f"hand__pen__{lt._goal_slug(pass_)}"  # named the old way
    old.mkdir()
    (old / "task.json").write_text(json.dumps({"conditions": [c.model_dump() for c in pass_.conditions]}))
    assert lt._goal_slug(pass_) == lt._goal_slug(spin)  # the old names were the same
    assert lt.skill_dir("hand", pass_) == old and lt.skill_dir("hand", spin) == lt.task_dir("hand", spin)


def test_a_plan_in_the_hand_asks_for_it_kept_up_not_pinched():
    """The LinkerHand's pen-spin plan began "secure grip on pen: held == 1 and grip_gap < 0.03" with the pen lying
    across the fingers, never pinched: no try got past step 1, so the steps never helped."""
    from morph.plan_check import in_hand_steps
    keep = {"kind": "expression", "expr": "in_hand == 1", "when": "always"}
    task = {"start_state": {"qpos": [0.0]}, "conditions": [keep], "steps": [
        {"name": "secure", "done_when": "held == 1 and grip_gap < 0.03", "keep": "grip_gap < 0.05"},
        {"name": "spin", "done_when": "held && spin_z_dps > 180", "keep": "held == 1 and bottom > 0.02"},
        {"name": "let it roll", "done_when": "held == 0 and spin_z_dps > 180", "keep": "not held"}]}
    assert "secure" in in_hand_steps(task)
    assert [(s["done_when"], s["keep"]) for s in task["steps"]] == [
        ("in_hand == 1", ""), ("in_hand and spin_z_dps > 180", "in_hand == 1 and bottom > 0.02"),
        ("held == 0 and spin_z_dps > 180", "not held")]  # not being pinched is fine in the hand
    assert in_hand_steps(task) is None  # once is enough
    firm = {"start_state": {"qpos": [0.0]}, "steps": [{"name": "a", "done_when": "held == 1"}],
            "conditions": [{"kind": "expression", "expr": "held == 1", "when": "always"}]}
    assert in_hand_steps(firm) is None  # the goal asks for the pinch: the plan keeps it
    assert in_hand_steps({"steps": [{"name": "a", "done_when": "held == 1"}], "conditions": []}) is None  # from the floor
