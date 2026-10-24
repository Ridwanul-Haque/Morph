"""Measures over time: change(q, s) and mean(q, s), followed through a run."""
from pathlib import Path

import numpy as np
import pytest

from morph.goal_terms import ExprError, Windows, _unit, describe_formula, parse, violation

PEN = Path(__file__).resolve().parents[1] / "runs" / "app_uploads" / "scenes" / "pen"


def _run(expr: str, frames: list[dict]) -> tuple[dict, float]:
    t = parse(expr, {})
    w = Windows([t])
    for f in frames:
        v = w.feed(dict(f))
    return v["window"], float(violation(np, t, v))


def _yaw(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def test_a_spin_one_way_adds_up_and_a_shake_does_not():
    spin = [{"yaw_deg": _yaw(k * 0.02 * 270)} for k in range(250)]  # 270 deg/s for 5 s, read as a heading (+-180)
    win, miss = _run("abs(change(yaw_deg, 4)) >= 720", spin)
    assert list(win.values())[0] == pytest.approx(1080, abs=1)  # the last 4 s: three turns
    assert miss == 0
    shake = [{"yaw_deg": 30.0 * np.sin(k * np.pi / 2)} for k in range(250)]  # fast back and forth
    win, miss = _run("abs(change(yaw_deg, 4)) >= 720", shake)
    assert abs(list(win.values())[0]) < 31 and miss > 1
    win, _ = _run("change(yaw_deg)", spin)  # since the try began
    assert list(win.values())[0] == pytest.approx(249 * 0.02 * 270, abs=1)


def test_a_share_of_the_time_and_an_average():
    frames = [{"in_hand": 1.0 if k >= 10 else 0.0, "spin_z_dps": 200.0 if k >= 50 else 0.0} for k in range(100)]
    win, _ = _run("mean(in_hand) >= 0.9", frames)
    assert list(win.values())[0] == pytest.approx(0.9)
    win, _ = _run("mean(spin_z_dps > 100, 2) >= 0.5", frames)
    assert list(win.values())[0] == pytest.approx(0.5)
    win, _ = _run("mean(spin_z_dps, 1) > 150", frames)
    assert list(win.values())[0] == pytest.approx(200)
    assert _unit(parse("mean(spin_z_dps > 100, 2) >= 0.5", {}).body) == 1.0  # a share, not deg/s
    assert _unit(parse("abs(change(yaw_deg, 4)) >= 720", {}).body) == 360.0  # a turn short counts as 1


def test_without_the_run_it_is_not_met():
    assert np.isnan(float(violation(np, parse("change(yaw_deg, 4) > 90", {}), {"yaw_deg": 0.0})))


@pytest.mark.parametrize("bad", ["change(change(yaw_deg))", "change(yaw_deg, 0)", "mean(yaw_deg, 99)",
                                 "change(yaw_deg, x)", "change()", "mean(z, 1, 2)"])
def test_bad_ones_are_turned_down(bad):
    with pytest.raises(ExprError):
        parse(bad, {})


def test_in_plain_words():
    assert describe_formula("abs(change(yaw_deg, 4)) >= 720", "pen") == \
        "how far the pen turned about the vertical in the last 4 s (either way) at least 720°"
    assert "share of the time over the last 2 s with the thumb touching the pen" in \
        describe_formula("mean(touching('thumb'), 2) > 0.5", "pen")


@pytest.mark.skipif(not PEN.exists(), reason="the pen scene from the app isn't here")
def test_round_and_round_needs_how_far_it_turned():
    from morph import project as P
    from morph.goal_sources import goal_from_description_rules
    from morph.goals import Condition, GoalSpec
    from morph.project import missing_intent
    pj = P.load_project(str(PEN))
    req = ("Spin the pen round and round in the hand. The pen must never fall out of the hand. For the last 4 seconds "
           "it keeps turning about the vertical in the same direction, faster than 180 degrees per second.")
    keep = Condition(kind="expression", expr="in_hand == 1", when="always")
    speed = Condition(kind="expression", expr="abs(spin_z_dps) > 180", when="last", seconds=4)
    turned = Condition(kind="expression", expr="abs(change(yaw_deg, 4)) >= 720")
    assert any("how far it turned" in m for m in missing_intent(req, GoalSpec(object="pen", conditions=[keep, speed])))
    assert not missing_intent(req, GoalSpec(object="pen", conditions=[keep, speed, turned]))
    assert not missing_intent(req, goal_from_description_rules(req, pj.scene))  # the rule reader writes one too


def test_the_learner_counts_the_same():
    jax = pytest.importorskip("jax")
    jp = jax.numpy
    t = parse("abs(change(yaw_deg, 4)) >= 720", {})
    w = Windows([t])
    ys = jp.array([_yaw(k * 0.02 * 270) for k in range(250)])

    @jax.jit
    def run(ys):
        st = w.start(jp, {"yaw_deg": ys[0]})
        st, _ = jax.lax.scan(lambda s, y: (w.step(jp, s, {"yaw_deg": y}), None), st, ys[1:])
        return w.values(jp, st)
    assert float(list(run(ys).values())[0]) == pytest.approx(1080, abs=1)
