"""Closed-loop arm control: task-space observations, a reactive scripted expert,
and the executor that turns a task-space action into joint commands via the
robot's own IK.

Task-space action (embodiment-agnostic, 5-D, all in [-1, 1]):
    [dx, dy, dz] * MAX_STEP metres, dyaw * MAX_DYAW radians, grip (>0 = close)
Joint-space action (arm-specific, 6-D):
    dq[5] * MAX_DQ radians, grip
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import wrap_angle
from .robots.arm import GRASP_Z, TRAVEL_Z, ArmAdapter
from .task import Outcome, TaskSpec, evaluate

# Actions are *leads*: the servo target is placed up to MAX_STEP ahead of the
# measured TCP, which keeps the controller stateless (Markov) for learning.
MAX_STEP = 0.01  # m (servos track within ~1 tick, so this is ~0.5 m/s)
MAX_DYAW = 0.03  # rad (~1.5 rad/s)
MAX_DQ = 0.04  # rad per tick
CONTROL_DT = 0.02

N_OBS_TASK = 11


def wrap_quarter(a: float) -> float:
    """Wrap to [-pi/4, pi/4): grasp-axis error for a square footprint."""
    q = math.pi / 2
    return (a + q / 2) % q - q / 2


@dataclass
class ArmState:
    tcp: np.ndarray
    hand_yaw: float
    aperture: float  # 0 closed .. 1 fully open (robot-normalised)
    gripping: bool  # both finger sides in contact with the block
    q: np.ndarray
    block: np.ndarray  # x, y, z, yaw
    goal: np.ndarray  # x, y, yaw
    grasp_yaw: float  # the arm skill layer's chosen closing-axis heading for this block/goal
    travel_z: float = TRAVEL_Z  # carry height; small arms get a lower one (see auto_arm)
    rest_z: float = 0.03  # object centre height once set down at the goal (floor, shelf top, bin floor)
    turn_dir: int = 0  # +1 / -1: which way to turn the object for big heading changes (0 = the short way)
    block_tilt: float = 0.0  # rad


def read_state(arm, spec: TaskSpec) -> ArmState:
    p = arm.env.block_pose()
    hy = arm.hand_yaw()
    gy = arm.choose_grasp_yaw((p.x, p.y), p.yaw, (spec.gx, spec.gy), wrap_angle(spec.gyaw - p.yaw), hy)
    tcp, ap = arm.tcp(), arm.aperture()
    # Holding = both finger sides touching the block, OR the block is off the floor, centred between
    # partly-closed fingers. The second clause removes one-tick contact flicker (seen on the xArm7
    # and Panda pads) that otherwise made the skill "let go" in mid-air. Memoryless by design.
    # (xy within the object's footprint: it can slide along the fingers, see expert_action's `aloft`)
    between = math.hypot(tcp[0] - p.x, tcp[1] - p.y) < 0.035 and abs(tcp[2] - p.z - 0.01) < 0.02
    holding = bool(arm.gripping() or (between and p.z > 0.036 and ap < 0.9))
    rest_z = getattr(getattr(arm, "block", None), "half_z", 0.03) + float(spec.goal.support_z.value)
    return ArmState(tcp=tcp, hand_yaw=hy, aperture=ap, gripping=holding, q=arm.q(),
                    travel_z=getattr(arm, "travel_z", TRAVEL_Z), rest_z=rest_z, turn_dir=getattr(arm, "turn_dir", 0),
                    block_tilt=float(p.tilt),
                    block=np.array([p.x, p.y, p.z, p.yaw]), goal=np.array([spec.gx, spec.gy, spec.gyaw]), grasp_yaw=gy)


def task_obs(s: ArmState) -> np.ndarray:
    """Embodiment-agnostic observation: object/goal/TCP relations and normalised gripper state; no joints."""
    rel = s.tcp - s.block[:3]
    bg = s.block[:2] - s.goal[:2]
    e_grasp = wrap_angle(s.grasp_yaw - s.hand_yaw)
    e_goal = wrap_angle(s.goal[2] - s.block[3])
    # Raw aperture is robot-specific (holding this block reads 0.38 on arm B, 0.6 on the Panda), so the
    # policy only sees whether the fingers are open enough to straddle the block, plus "holding".
    return np.array([rel[0], rel[1], rel[2], bg[0], bg[1], s.tcp[2], e_grasp, e_goal,
                     float(s.aperture > 0.8), float(s.gripping), s.block[2] - 0.03], dtype=np.float32)


def joint_obs(s: ArmState) -> np.ndarray:
    """Arm-specific observation used by the joint-space baseline policy."""
    return np.concatenate([task_obs(s), s.q.astype(np.float32)])


# ---------------------------------------------------------------- expert
@dataclass
class ExpertParams:
    carry_pos_tol: float = 0.01  # start lowering within 1 cm / 3 deg (goal tolerance is 5 cm / 15 deg);
    carry_yaw_tol: float = math.radians(3)  # tighter values made imitating policies hover forever
    placed_pos_tol: float = 0.015
    placed_yaw_tol: float = math.radians(5)


def expert_action(s: ArmState, p: ExpertParams = ExpertParams()) -> tuple[np.ndarray, str]:
    """Reactive (memoryless) pick-and-place expert. Returns (task action, phase label)."""
    bx, by, bz, byaw = s.block
    gx, gy, gyaw = s.goal
    tcp = s.tcp
    TZ = s.travel_z
    goal_err = math.hypot(bx - gx, by - gy)
    yaw_goal_err = wrap_angle(gyaw - byaw)
    if s.turn_dir and abs(yaw_goal_err) > math.radians(120) and np.sign(yaw_goal_err) != s.turn_dir:
        yaw_goal_err -= math.copysign(2 * math.pi, yaw_goal_err)  # the long way round is the one the wrist can do
    on_floor = bz < s.rest_z + 0.004  # set down on its target surface (the floor for most tasks)
    drop = s.rest_z - 0.03  # how much higher than the floor the object is set down
    in_hand = math.hypot(tcp[0] - bx, tcp[1] - by) < 0.02 and abs(tcp[2] - bz - 0.01) < 0.015
    # Up in the air, a heavy object can slide down or along the fingers while turning; that isn't a drop,
    # and deciding it is opens the hand mid-air. Only while it's still level: one that is pivoting in the
    # grip is better set down and picked up again than carried to the goal and placed on its side.
    aloft = bz > s.rest_z + 0.02 and s.block_tilt < math.radians(8) and math.hypot(tcp[0] - bx, tcp[1] - by) < 0.035 \
        and 0 < tcp[2] - bz < 0.04
    gripping = (in_hand or aloft) and s.gripping
    dyaw = 0.0
    close = False

    if on_floor and goal_err < p.placed_pos_tol and abs(yaw_goal_err) < p.placed_yaw_tol and not (
            gripping and tcp[2] > GRASP_Z + drop + 0.01):
        phase = "release_retreat"
        target = np.array([bx, by, TZ + 0.05]) if s.aperture > 0.8 else tcp.copy()
    elif gripping:
        close = True
        descending = tcp[2] < TZ - 0.015  # hysteresis: once lowering, accept looser alignment
        pos_tol = 0.012 if descending else p.carry_pos_tol
        yaw_tol = math.radians(4) if descending else p.carry_yaw_tol
        over_goal = goal_err < pos_tol and abs(yaw_goal_err) < yaw_tol
        if not over_goal:
            # Lift first; translate and rotate only once clear of the floor. The xy
            # target is always goal-relative (absolute), never "hold current TCP".
            clear = float(np.clip((tcp[2] - GRASP_Z - 0.02) / 0.06, 0.0, 1.0))
            phase = "lift" if clear < 1.0 else "transport"
            dyaw = yaw_goal_err * clear
            target = np.array([tcp[0] + (gx - bx) * clear, tcp[1] + (gy - by) * clear, TZ])
        else:
            phase = "lower"
            # Lower until the *block* reaches its resting height, whatever its offset in the grip.
            target = np.array([tcp[0] + gx - bx, tcp[1] + gy - by, tcp[2] - (bz - s.rest_z) - 0.002])
            dyaw = yaw_goal_err
    else:
        e = wrap_angle(s.grasp_yaw - s.hand_yaw)
        dyaw = e
        xy_err = math.hypot(tcp[0] - bx, tcp[1] - by)
        if in_hand and abs(e) < math.radians(4) and tcp[2] < GRASP_Z + 0.006:
            phase = "close"
            close = True
            target = tcp.copy()
        elif xy_err > 0.012 or abs(e) > math.radians(3):
            if tcp[2] < TZ - 0.02 and xy_err > 0.03:
                phase = "raise"
                target = np.array([tcp[0], tcp[1], TZ])
            else:
                phase = "approach"
                target = np.array([bx, by, TZ])
        elif s.aperture < 0.8:  # wait for the fingers to open before descending over the block
            phase = "open"
            target = tcp.copy()
        else:
            phase = "descend"
            target = np.array([bx, by, GRASP_Z])
    dp = np.clip((target - tcp) / MAX_STEP, -1, 1)
    a = np.array([dp[0], dp[1], dp[2], np.clip(dyaw / MAX_DYAW, -1, 1), 1.0 if close else -1.0])
    return a.astype(np.float32), phase


# -------------------------------------------------------------- executors
@dataclass
class ExecStats:
    ik_failures: int = 0


def apply_task_action(arm, s: ArmState, a: np.ndarray, stats: ExecStats) -> np.ndarray:
    """Task-space action -> this arm's joint targets through its own IK."""
    a = np.clip(a, -1, 1)
    ps, ys = getattr(arm, "lead_scale", (1.0, 1.0))  # per-robot servo calibration (adapter property)
    yaw = s.hand_yaw + a[3] * MAX_DYAW * ys
    q = None
    # The lead can overshoot the edge of a small arm's workspace; shorten it before giving up.
    for shrink in (1.0, 0.5, 0.25):
        tgt = s.tcp + a[:3] * MAX_STEP * max(ps * shrink, 1.0)
        tgt[2] = min(max(tgt[2], GRASP_Z - 0.01), s.travel_z + 0.05)  # floor guard / reachable height
        q = arm.ik(tgt, yaw)
        if q is not None:
            break
    if q is None and hasattr(arm, "ik_reconfigure") and getattr(arm, "_reconf_wait", 0) <= 0:
        # stuck on the wrong joint branch: head for another posture that reaches it, a bit per tick.
        # The search is expensive, so after it finds nothing wait a while before trying again.
        q = arm.ik_reconfigure(tgt, yaw)
        if q is not None:
            q = s.q + np.clip(q - s.q, -2 * MAX_DQ, 2 * MAX_DQ)
        else:
            arm._reconf_wait = 25
    elif q is None and hasattr(arm, "ik_reconfigure"):
        arm._reconf_wait -= 1
    if q is None:
        stats.ik_failures += 1
        q = s.q
    arm.last_q_cmd = q
    return arm.ctrl(q, close=bool(a[4] > 0))


def apply_joint_action(arm, s: ArmState, a: np.ndarray) -> np.ndarray:
    a = np.clip(a, -1, 1)
    n = len(s.q)
    q = np.clip(s.q + a[:n] * MAX_DQ, arm.q_lo, arm.q_hi)
    arm.last_q_cmd = q
    return arm.ctrl(q, close=bool(a[n] > 0))


def task_to_joint_label(arm, s: ArmState, a: np.ndarray) -> np.ndarray | None:
    """Convert an expert task-space action into the equivalent joint delta on *this* arm."""
    ps, ys = getattr(arm, "lead_scale", (1.0, 1.0))
    tgt = s.tcp + np.clip(a[:3], -1, 1) * MAX_STEP * ps
    tgt[2] = max(tgt[2], GRASP_Z - 0.01)
    q = arm.ik(tgt, s.hand_yaw + np.clip(a[3], -1, 1) * MAX_DYAW * ys)
    if q is None:
        return None
    return np.concatenate([np.clip((q - s.q) / MAX_DQ, -1, 1), [a[4]]]).astype(np.float32)


# ------------------------------------------------------------ episode loop
@dataclass
class EpisodeResult:
    outcome: Outcome
    sim_time: float
    frames: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    contacts: list = field(default_factory=list)
    max_step_disp: float = 0.0
    ik_failures: int = 0
    phases: list = field(default_factory=list)


def released(arm) -> bool:
    """Robot is not holding or touching the block with its fingers (checked by contacts)."""
    d = arm.env.data
    bg = arm.env.model.geom("block_geom").id
    for i in range(d.ncon):
        c = d.contact[i]
        if bg in (c.geom1, c.geom2) and ({c.geom1, c.geom2} & arm.finger_geoms):
            return False
    return True


def run_episode(arm: ArmAdapter, spec: TaskSpec, policy, mode: str = "task", max_time: float = 14.0,
                record: bool = False, render_every: int = 0, camera: str = "overview",
                stop_when_done: bool = True, noise: float = 0.0, rng: np.random.Generator | None = None,
                label_fn=None, on_tick=None) -> EpisodeResult:
    """Run one closed-loop episode.

    policy(state) -> (action, phase). mode: "task" or "joint".
    label_fn(state) is called before each action (used for DAgger labelling).
    """
    arm.env.set_goal_marker(spec.gx, spec.gy, spec.gyaw)
    stats = ExecStats()
    res = EpisodeResult(outcome=None, sim_time=0.0)  # type: ignore[arg-type]
    n = int(max_time / CONTROL_DT)
    done_ticks = 0
    for t in range(n):
        s = read_state(arm, spec)
        if label_fn is not None:
            label_fn(s)
        a, phase = policy(s)
        if noise > 0 and rng is not None:
            a = a.copy()
            a[:4] = a[:4] + rng.normal(0, noise, 4)
        ctrl = apply_task_action(arm, s, a, stats) if mode == "task" else apply_joint_action(arm, s, a)
        log = arm.env.advance(ctrl)
        if on_tick is not None:
            on_tick(arm, phase)
        res.max_step_disp = max(res.max_step_disp, log.max_step_disp)
        if log.robot_contacts:
            c = max(log.robot_contacts, key=lambda c: c.normal_force)
            res.contacts.append({"t": round(c.time, 3), "geom": c.geom, "force_n": round(c.normal_force, 3)})
        if not res.phases or res.phases[-1][1] != phase:
            res.phases.append((round(arm.env.data.time, 3), phase))
        if record:
            bp = arm.env.block_pose()
            res.trace.append({"t": round(float(arm.env.data.time), 3), "phase": phase,
                              "q": [round(float(v), 5) for v in s.q], "aperture": round(s.aperture, 4),
                              "gripping": s.gripping, "close": bool(a[4] > 0),
                              "tcp": [round(float(v), 5) for v in s.tcp], "hand_yaw": round(s.hand_yaw, 5),
                              "action": [round(float(v), 4) for v in a],
                              "ctrl": [round(float(v), 5) for v in ctrl],
                              "q_cmd": [round(float(v), 5) for v in getattr(arm, "last_q_cmd", s.q)],
                              "block": bp.as_dict()})
        if render_every and t % render_every == 0:
            res.frames.append(arm.env.render(camera))
        if stop_when_done and phase == "release_retreat" and s.tcp[2] > s.travel_z:
            done_ticks += 1
            if done_ticks > 5:
                break
    res.sim_time = float(arm.env.data.time)
    res.ik_failures = stats.ik_failures
    res.outcome = evaluate(spec, arm.env.block_pose(), arm.env.block_speed(), released(arm),
                           half_z=getattr(getattr(arm, "block", None), "half_z", 0.03))
    return res
