import math

import numpy as np

from morph.arm_skills import expert_action, run_episode
from morph.robots.arm import ArmAdapter
from morph.scene import ARM_A
from morph.sim import PhysicalParams
from morph.task import default_spec


def test_block_pose_readback_matches_reset():
    arm = ArmAdapter(ARM_A)
    for yaw in (-3.0, -1.2, 0.0, 0.7, 2.9):
        arm.reset(0, (0.45, -0.2, yaw), PhysicalParams())
        p = arm.env.block_pose()
        assert (p.x, p.y) == (0.45, -0.2)
        assert abs(math.remainder(p.yaw - yaw, 2 * math.pi)) < 1e-9
        assert p.tilt < 1e-6


def test_block_does_not_move_without_contact():
    """No robot contact -> the block stays where physics put it (no drift, no teleport)."""
    arm = ArmAdapter(ARM_A)
    arm.reset(0, (0.45, -0.2, 0.3), PhysicalParams())
    p0 = arm.env.block_pose()
    for _ in range(100):  # 2 s holding the home pose
        log = arm.env.advance(arm.ctrl(arm.home_q, close=False))
        assert not log.robot_contacts
    p1 = arm.env.block_pose()
    assert math.hypot(p1.x - p0.x, p1.y - p0.y) < 1e-4
    assert abs(p1.yaw - p0.yaw) < 1e-4


def _episode(seed):
    arm = ArmAdapter(ARM_A)
    rng = np.random.default_rng(seed)
    arm.reset(seed, (rng.uniform(0.35, 0.55), rng.uniform(-0.3, -0.1), rng.uniform(-3, 3)), PhysicalParams())
    spec = default_spec(0.45, 0.2, math.pi / 2)
    return run_episode(arm, spec, expert_action, record=True)


def test_deterministic_replay_and_physical_motion():
    r1, r2 = _episode(7), _episode(7)
    assert r1.outcome.final_pose == r2.outcome.final_pose
    assert [t["block"] for t in r1.trace] == [t["block"] for t in r2.trace]
    # block only moves while in contact with the gripper or shortly after, never jumps
    assert r1.max_step_disp < 0.01  # < 1 cm per 2 ms physics step (5 m/s bound)
    assert r1.outcome.success, r1.outcome.reasons
    assert r1.contacts, "success without any gripper-block contact would be impossible"
