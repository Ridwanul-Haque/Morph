import math

from morph.push_planner import PushCandidate, PushModel, plan_push
from morph.pusher_task import run_pusher
from morph.sim import PhysicalParams
from morph.task import default_spec


def test_push_model_is_physical_and_directional():
    pm = PushModel()
    x, y, yaw = pm.predict(0.5, 0.0, 0.0, PushCandidate(face=2, offset=0.0, distance=0.1))  # push the -x face
    assert 0.08 < x - 0.5 < 0.11 and abs(y) < 0.01 and abs(yaw) < math.radians(3)
    _, _, yaw2 = pm.predict(0.5, 0.0, 0.0, PushCandidate(face=2, offset=0.03, distance=0.1))
    assert abs(yaw2) > math.radians(15)  # an off-centre push rotates the block


def test_planner_prefers_progress():
    best, ranked = plan_push(PushModel(), (0.45, -0.2, 0.0), (0.45, 0.2, 0.0))
    assert math.hypot(best.predicted[0] - 0.45, best.predicted[1] - 0.2) < 0.4 - 0.05
    assert ranked[0].cost <= ranked[-1].cost


def test_pusher_reaches_arm_goal_by_contact_only():
    spec = default_spec(0.45, 0.20, math.pi / 2)
    r = run_pusher(spec, (0.46, -0.20, 0.4), PhysicalParams(0.25, 0.55), 3, log=lambda s: None)
    assert r.outcome.success, r.outcome.reasons
    assert all(p["contact"]["tip_contact_ticks"] > 0 for p in r.pushes if p["measured_move_cm"] > 0.5)
    assert r.other_contacts == []  # only the tip ever touched the block
