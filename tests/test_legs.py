from pathlib import Path

import mujoco
import numpy as np
import pytest

from morph.introspect import home_qpos
from morph.robots.legged_model import build_model, legged_setup
from morph.robots.legs import measure_legs
from morph.robots.quadruped import leg_ik

MEN = Path(__file__).resolve().parents[1] / "third_party" / "mujoco_menagerie"


def _measured(rel):
    p = MEN / rel
    if not p.exists():
        pytest.skip(f"{rel} not fetched")
    s = legged_setup(p)
    m = build_model(s, training=False)
    feet = [dict(f) for f in s.feet]
    return s, m, measure_legs(m, s.torso, feet, s.joints, home_qpos(m))


def test_measured_go1_legs_match_the_hand_written_ones():
    """Numeric IK from the model vs the closed-form IK derived by hand for the Go1."""
    s, m, g = _measured("unitree_go1/go1.xml")
    assert set(g.legs) == {"FR", "FL", "RR", "RL"}
    rng = np.random.default_rng(0)
    checked = 0
    for leg in g.legs:
        for _ in range(20):
            p = np.array([*g.nominal[leg], -0.27]) + rng.uniform([-0.06, -0.03, -0.02], [0.06, 0.03, 0.08])
            qa, qn = leg_ik(leg, p), g.ik(leg, p)
            if qa is None:
                continue
            assert qn is not None
            assert np.allclose(qa, qn, atol=2e-2), (leg, p, qa, qn)
            checked += 1
    assert checked > 50


def test_unseen_quadruped_legs_are_measured():
    s, m, g = _measured("unitree_go2/go2.xml")
    assert g.legs == ("FR", "FL", "RR", "RL")
    assert g.nominal["FL"][0] > 0 and g.nominal["FL"][1] > 0 and g.side["RR"] == -1
    q = g.ik("FR", np.array([*g.nominal["FR"], -0.25]))
    assert q is not None and len(q) == 3 and len(g.ctrl_idx["FR"]) == 3
