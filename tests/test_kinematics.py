import mujoco
import numpy as np

from morph.geometry import wrap_angle
from morph.robots.arm import ArmAdapter
from morph.robots.quadruped import LEGS, QuadrupedAdapter, home_feet
from morph.scene import ARM_A, ARM_B
from morph.sim import PhysicalParams


def test_arm_ik_matches_forward_kinematics():
    rng = np.random.default_rng(0)
    for g in (ARM_A, ARM_B):
        arm = ArmAdapter(g)
        checked = 0
        for _ in range(100):
            tgt = np.array([rng.uniform(0.25, 0.6), rng.uniform(-0.35, 0.35), rng.uniform(0.03, 0.2)])
            yaw = rng.uniform(-3, 3)
            q = arm.ik(tgt, yaw)
            if q is None:
                continue
            arm.reset(0, (0.9, 0.9, 0), PhysicalParams(), q0=q)
            assert np.allclose(arm.tcp(), tgt, atol=1e-9)
            assert abs(wrap_angle(arm.hand_yaw() - yaw)) < 1e-9
            checked += 1
        assert checked > 80


def test_arms_are_kinematically_different():
    a, b = ArmAdapter(ARM_A), ArmAdapter(ARM_B)
    q = a.ik(np.array([0.45, -0.1, 0.04]), 0.0)
    b.reset(0, (0.9, 0.9, 0), PhysicalParams(), q0=q)
    a.reset(0, (0.9, 0.9, 0), PhysicalParams(), q0=q)
    assert np.linalg.norm(a.tcp() - b.tcp()) > 0.05  # same joints, different gripper position


def test_go1_leg_ik_matches_forward_kinematics():
    qa = QuadrupedAdapter()
    m, d = qa.model, qa.env.data
    rng = np.random.default_rng(1)
    for _ in range(50):
        d.qpos[:] = 0
        d.qpos[qa._free_q + 3] = 1
        feet = {l: home_feet()[l] + rng.uniform([-0.08, -0.04, -0.02], [0.1, 0.04, 0.1]) for l in LEGS}
        q = qa.ctrl_from_feet(feet)
        if q is None:
            continue
        d.qpos[qa._leg_q] = q
        mujoco.mj_kinematics(m, d)
        for l in LEGS:
            assert np.allclose(d.site_xpos[qa.foot_site[l]], feet[l], atol=1e-9)
