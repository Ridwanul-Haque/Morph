"""Goals beyond placing things: how they're read, measured, and routed to learning."""
from pathlib import Path

import numpy as np
import pytest

from morph.goal_sources import goal_from_description_rules
from morph.goal_terms import ExprError, condition_violation, parse, quantities, violation
from morph.goals import Condition, GoalSpec, needs_learning, validate
from morph.project import load_project

ROOT = Path(__file__).resolve().parents[1]
FX = {"wall": {"lo": [0.8, -0.3, 0.0], "hi": [0.81, 0.1, 0.08]}}
HALF = (0.025, 0.025, 0.03)


def _v(pos=(0.5, 0.0, 0.03), R=np.eye(3), held=0.0):
    return quantities(np, pos=np.array(pos), R=np.asarray(R, float), vel=np.zeros(3), start=(0.5, 0.0), half=HALF,
                      fixtures=FX, held=held)


@pytest.fixture(scope="module")
def scene():
    return load_project(ROOT / "examples" / "parcel").scene


@pytest.mark.parametrize("text, kind, extra", [
    ("knock the parcel over", "orientation", {"up_face": "side"}),
    ("flip the parcel upside down", "orientation", {"up_face": "bottom"}),
    ("lift the parcel 20 cm and hold it", "height", {"min_z": 0.2, "hold": True}),
    ("take the parcel out of the bin", "inside", {"negate": True}),
    ("push the parcel off the shelf", "on_top_of", {"negate": True}),
    ("keep the parcel away from the wall", "away_from", {"target": "wall"}),
    ("move the parcel 30 cm north", "moved", {"side": "north"}),
])
def test_rule_reader_understands_more_than_places(scene, text, kind, extra):
    g = goal_from_description_rules(text, scene)
    c = next(c for c in g.conditions if c.kind == kind)
    for k, v in extra.items():
        assert getattr(c, k) == pytest.approx(v) if isinstance(v, float) else getattr(c, k) == v
    assert not validate(g, scene)
    assert needs_learning(g)  # no scripted skill does these


def test_upright_only_when_asked(scene):
    assert not any(c.kind == "upright" for c in goal_from_description_rules("put the parcel into the bin", scene).conditions)
    assert any(c.kind == "upright" for c in goal_from_description_rules("carefully put the parcel on the shelf", scene).conditions)


def test_formulas_measure_how_far_from_true():
    v = _v()
    assert violation(np, parse("z > 0.3", FX), v) == pytest.approx(0.27 / 0.3)
    assert violation(np, parse("not (x < 0.4)", FX), v) == 0
    assert violation(np, parse("dist_to('wall') > 0.4 or held", FX), v) == pytest.approx((0.4 - 0.275) / 0.3)
    assert violation(np, parse("held", FX), _v(held=1.0)) == 0


@pytest.mark.parametrize("bad", ["__import__('os')", "open('x')", "x.real", "[1, 2]", "lambda: 1", "dist_to('moon') > 1", "foo > 1"])
def test_formulas_only_allow_measured_quantities(bad):
    with pytest.raises(ExprError):
        parse(bad, FX)


def test_orientation_from_the_object_frame():
    on_side = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    side = {"kind": "orientation", "up_face": "side"}
    assert condition_violation(np, side, _v(R=on_side), FX, HALF) == 0
    assert condition_violation(np, side, _v(), FX, HALF) > 0
    assert condition_violation(np, {"kind": "orientation", "up_face": "top"}, _v(), FX, HALF) == 0


def test_invalid_formula_is_reported_in_words(scene):
    g = GoalSpec(object="parcel", conditions=[Condition(kind="expression", expr="z >> 2")])
    assert any("formula" in e for e in validate(g, scene))


def test_fixed_arms_rest_on_the_floor():
    from morph.introspect import floor_offset
    menagerie = ROOT / "third_party" / "mujoco_menagerie"
    assert floor_offset(menagerie / "ufactory_xarm7" / "xarm7.xml") == pytest.approx(-0.12, abs=0.005)  # its base is 12 cm up in the file
    assert floor_offset(menagerie / "franka_emika_panda" / "panda.xml") == pytest.approx(0.0, abs=0.002)


def test_a_checklist_that_can_never_be_met_is_caught(scene):
    from morph.goals import unsatisfiable
    z_for_bottom = GoalSpec(object="parcel", conditions=[Condition(kind="expression", expr="x > 0.68 and abs(z) < 0.02")])
    assert "can never be true" in unsatisfiable(z_for_bottom, scene)[0]  # z is the centre: 3 cm up on the floor
    two_places = GoalSpec(object="parcel", conditions=[Condition(kind="inside", target="bin"),
                                                       Condition(kind="touching", target="wall")])
    assert "together" in unsatisfiable(two_places, scene)[0]
    fine = GoalSpec(object="parcel", conditions=[Condition(kind="inside", target="bin"), Condition(kind="upright", tol=10)])
    assert unsatisfiable(fine, scene) == []


def test_a_place_on_the_way_is_checked_at_some_point(scene):
    from morph.goal_bench import Moment, Outcome, judge
    g = GoalSpec(object="parcel", conditions=[Condition(kind="on_top_of", target="shelf", when="ever"),
                                              Condition(kind="inside", target="bin")])
    assert not validate(g, scene)
    shelf, bin_ = Moment(0.62, -0.02, 0.05), Moment(0.42, 0.18, 0.01)
    assert judge(g, scene, Outcome("via the shelf", [shelf, bin_], True))
    assert not judge(g, scene, Outcome("straight in", [bin_], False))


def test_the_goal_benchmark_outcomes_are_consistent(scene):
    """Every 'must pass' outcome passes the intended reading (written by hand here), so the benchmark is fair."""
    from morph.goal_bench import cases, judge
    intended = {
        "put in bin": [Condition(kind="inside", target="bin")],
        "knock over": [Condition(kind="orientation", up_face="side", tol=30)],
        "throw 1 m north": [Condition(kind="inside", target="bin"),
                            Condition(kind="expression", expr="y > 1.26", when="release")],
        "shelf then bin": [Condition(kind="on_top_of", target="shelf", when="ever"), Condition(kind="inside", target="bin")],
        "slide no lift": [Condition(kind="touching", target="wall"), Condition(kind="expression", expr="bottom < 0.02", when="always")],
    }
    for case in cases():
        if case.name in intended:
            g = GoalSpec(object="parcel", conditions=intended[case.name])
            for o in case.outcomes:
                assert judge(g, scene, o) == o.should_pass, (case.name, o.what)
