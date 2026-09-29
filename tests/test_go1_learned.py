import math

import numpy as np
import pytest

from morph.robots.go1_learned import DEFAULT_POLICY, LearnedGo1
from morph.sim import PhysicalParams

pytestmark = pytest.mark.skipif(not DEFAULT_POLICY.exists(), reason="learned Go1 policy not trained/committed")


def test_learned_policy_walks_and_turns_upright_in_morph_scene():
    g = LearnedGo1()
    for cmd in [(0.5, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, 0.0, -0.8)]:
        g.reset(0, (5, 5, 0), PhysicalParams())
        p0, _, _ = g.base()
        g.walk(cmd, 3.0)
        assert not g.fallen(), cmd
        p1, _, tilt = g.base()
        assert tilt < math.radians(10)
        if cmd[0] or cmd[1]:
            assert np.linalg.norm(p1[:2] - p0[:2]) > 0.3  # it actually moved


def test_learned_go1_pushes_block_to_goal_by_foot_contact():
    from morph.go1_learned_task import run_learned_go1_push
    from morph.task import default_spec
    r = run_learned_go1_push(default_spec(0.45, 0.20, math.pi / 2), (0.46, -0.20, 0.4), PhysicalParams(0.25, 0.55), 3,
                             base0=(0.46, -0.9, math.pi / 2), log=lambda s: None)
    assert r.outcome.success, (r.outcome.reasons, r.stop_reason)
    assert not r.fell and r.unintended_contacts == []
    assert all(p["contact"]["foot_contact_ticks"] > 0 for p in r.pushes if p["measured_move_cm"] > 1)
