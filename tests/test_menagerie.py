import math

import numpy as np

from morph.arm_skills import expert_action, run_episode
from morph.robots.menagerie_arm import PANDA, XARM7, MenagerieArmAdapter
from morph.sim import PhysicalParams
from morph.task import default_spec


def test_real_arm_ik_round_trip():
    rng = np.random.default_rng(0)
    for cfg in (PANDA, XARM7):
        arm = MenagerieArmAdapter(cfg)
        arm.reset(0, (0.45, -0.2, 0.0), PhysicalParams())
        solved = 0
        for _ in range(15):
            tgt = np.array([rng.uniform(0.38, 0.55), rng.uniform(-0.25, 0.25), rng.uniform(0.04, 0.2)])
            q = arm.ik(tgt, rng.uniform(-1.5, 1.5), q0=arm.home_q)
            if q is None:
                continue
            p, R = arm.fk(q)
            assert np.linalg.norm(p - tgt) < 2e-3 and R[2, 2] < -0.99  # tool points down
            solved += 1
        assert solved >= 10, (cfg.name, solved)


def test_real_arms_pick_and_place_by_physical_grasp():
    for cfg in (PANDA, XARM7):
        arm = MenagerieArmAdapter(cfg)
        arm.reset(1, (0.46, -0.20, 0.4), PhysicalParams(0.25, 0.55))
        r = run_episode(arm, default_spec(0.45, 0.20, math.pi / 2), expert_action, record=True)
        assert r.outcome.success, (cfg.name, r.outcome.reasons)
        assert max(t["block"]["z"] for t in r.trace) > 0.08  # it was lifted, i.e. actually grasped
        assert any(t["gripping"] for t in r.trace) and r.max_step_disp < 0.01
