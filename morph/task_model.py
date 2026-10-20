"""The model, observation and reward for learning a task from scratch ("Learn this task").

A task file (task.json) holds only numbers and paths, so the GPU trainer (scripts/train_task.py, run in
the RL environment) and the CPU side (trying the result, exporting it) build exactly the same thing:

  robot      MJCF path (URDFs are converted first), mounted behind the work area if it's bolted down,
             or standing back from the object if it has a floating base; torque motors become
             position servos (the policy outputs joint targets)
  scene      the user's object and fixtures (same MJCF snippet Morph uses everywhere)
  target     where the goal puts the object: x, y, heading, resting height, tolerances, and which
             conditions matter (heading? upright?)

Observation (what the policy sees): joint angles and speeds, where the object is relative to the
target, which way is up for the object and its heading error, its velocity, where the robot's
end bodies (fingertips, tool, feet) are relative to the object, the last action, and for floating
robots the base's position, gravity direction and velocity. Reward: progress of the object towards
the target, closeness, reaching the object, keeping it upright, a bonus when every condition holds,
smooth actions; floating robots also get a small reward for staying up.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .scene import FLOOR_FRICTION, BlockSpec, _block_xml, _common_assets, _goal_xml

SIM_DT, CTRL_DT = 0.004, 0.02
MAX_TIPS = 6


MAX_STEP = 0.05  # rad per control step: joint targets move smoothly (2.5 rad/s), no slamming


def next_ctrl(xp, prev, action, q0, scale, lo, hi, step=MAX_STEP):
    """The joint targets for this step: head towards q0 + scale * action, at most `step` per step (scale and
    step can be per actuator: see Indices.act_scale / act_step).
    Same code for training (JAX) and trying / export (NumPy)."""
    want = xp.clip(q0 + action * scale, lo, hi)
    return prev + xp.clip(want - prev, -step, step)


# ------------------------------------------------------------------ acting in hand space
HAND_STEP = 0.01  # m per control step for a full action (about 0.35 m/s once the motors follow)
HAND_TURN = 0.05  # rad it may turn per step (2.5 rad/s)
HAND_DAMP = 0.05  # damped least squares: no wild joint jumps near a singular pose
HAND_LEAD = 2.0  # the joint targets lead the measured pose by this many steps' worth of the move


def hand_ctrl(xp, prev, action, xpos, xmat, xanchor, xaxis, qpos, hand: dict):
    """Hand-space actions (task["action_space"] == "hand"): the arm's action slots say which way the hand moves and turns
    (x, y, z in metres, then about x, y, z, in the world frame); Morph turns that into joint targets with one damped
    least-squares step of the arm's Jacobian (built from each joint's axis and anchor: exact for hinge and slide joints,
    the same in NumPy and JAX).
    Returns the new commands for the arm's actuators. Same code for training (JAX) and trying / export (NumPy)."""
    b = hand["body"]
    p = xpos[b] + xmat[b].reshape(3, 3) @ hand["offset"]
    ax = xaxis[hand["jids"]]  # (n, 3)
    r = p - xanchor[hand["jids"]]
    hinge = hand["hinge"][:, None]
    jp_ = hinge * xp.cross(ax, r) + (1 - hinge) * ax
    jr = hinge * ax
    J = xp.concatenate([jp_.T, jr.T], axis=0)  # (6, n)
    a = action[hand["slots"]]  # the arm's first six action slots
    v = xp.concatenate([a[:3] * HAND_STEP, a[3:6] * HAND_TURN])
    dq = J.T @ xp.linalg.solve(J @ J.T + HAND_DAMP ** 2 * xp.eye(6), v)
    # from where the arm IS (its Jacobian is for that pose; stepping from the commands, which the motors lag by up to
    # 0.25 rad, bent a straight move 35° off), leading by HAND_LEAD steps so the lagging motors still keep pace.
    # Bolted-down arms are gravity compensated, so aiming at the pose they're in doesn't let them sag
    q_new = xp.clip(qpos[hand["qadr"]] + HAND_LEAD * dq, hand["jlo"], hand["jhi"])
    return q_new * hand["gear"]


def apply_action(xp, prev, action, q0, ix, data=None):
    """The commands for this step, whichever way the task's policy acts: joint targets (next_ctrl) or, for a task that
    acts in hand space, the hand's moves for the arm and the joint way for everything else (gripper, base, head)."""
    c = next_ctrl(xp, prev, action, q0, ix.act_scale, ix.act_lo, ix.act_hi, ix.act_step)
    h = ix.hand
    if h is None or data is None:
        return c
    arm = xp.clip(hand_ctrl(xp, prev, action, data.xpos, data.xmat, data.xanchor, data.xaxis, data.qpos, h),
                  ix.act_lo[h["acts"]], ix.act_hi[h["acts"]])
    if xp is np:
        c = c.copy()
        c[h["acts"]] = arm
        return c
    return c.at[h["acts"]].set(arm)


def hand_space(task: dict) -> dict | None:
    """What acting in hand space needs, for a bolted-down arm whose gripping hand is moved by at least six position-
    controlled joints (fewer can't move a hand in all six directions): stored on the task by make_task."""
    from .body_programs import BodyRunner
    if task.get("floating"):
        return None
    try:
        rn = BodyRunner(task)
    except Exception:  # noqa: BLE001
        return None
    hand = next((p for p in rn.parts.values() if p.can_grip), None)
    if hand is None or len(hand.joints) < 6:
        return None
    m2 = rn.m
    return {"body": m2.body(hand.body).name, "offset": [float(v) for v in hand.offset],
            "joints": [m2.joint(j).name for j in hand.joints], "acts": [m2.actuator(a).name for a in hand.acts],
            "gears": [float(g) for g in hand.gears]}


def load_task(path: str | Path) -> dict:
    task = json.loads(Path(path).read_text())
    task["_dir"] = str(Path(path).resolve().parent)  # where its files are (demo.npz, starts.npz): always its own folder
    return task


def build_task_spec(task: dict, accurate: bool = False) -> mujoco.MjSpec:
    """accurate=False: the fast physics the GPU trains in (4 ms steps, 4 solver iterations).
    accurate=True: finer physics for trying the policy and for the exported model (2 ms steps, implicit
    integrator, 50 iterations). A policy that works in both has passed a sim-to-sim check; the UR5e's
    first wall-push policy blew up the coarse CPU solver but succeeded 6/6 in the accurate one."""
    blk = BlockSpec(**task["block"])
    scene = mujoco.MjSpec.from_string(f"""<mujoco model="morph_task">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{SIM_DT}" integrator="implicitfast" cone="elliptic" impratio="10"/>
  {_common_assets(blk)}
  <worldbody>
    <light pos="0.6 -0.5 2.6" dir="-0.15 0.2 -1" cutoff="45" exponent="2" castshadow="true" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" friction="{FLOOR_FRICTION} 0.005 0.0001"/>
    {_goal_xml()}
    {_block_xml(blk)}
    <frame name="mount"/>
  </worldbody>
</mujoco>""")
    robot = mujoco.MjSpec.from_file(task["robot_xml"])
    for k in list(robot.keys):
        robot.delete(k)
    rm = mujoco.MjSpec.from_file(task["robot_xml"]).compile()
    for a, act in enumerate(robot.actuators):  # torque motors -> position servos, torque limit kept
        if act.biastype == mujoco.mjtBias.mjBIAS_NONE and act.trntype == mujoco.mjtTrn.mjTRN_JOINT:
            tau = float(max(abs(rm.actuator_ctrlrange[a][0]), abs(rm.actuator_ctrlrange[a][1]))) or 30.0
            j = robot.joint(act.target)
            kp = tau / 0.3  # full torque at 0.3 rad of error
            act.gaintype, act.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
            act.gainprm[0] = kp
            act.biasprm[0], act.biasprm[1], act.biasprm[2] = 0.0, -kp, -0.05 * kp
            act.gear[0] = 1.0
            act.ctrllimited, act.ctrlrange = True, list(j.range) if j.range[1] > j.range[0] else [-math.pi, math.pi]
            act.forcelimited, act.forcerange = True, [-tau, tau]
    if task.get("gravcomp"):  # bolted-down arms: the controller holds the arm up against gravity, as real ones do
        for body in robot.bodies:
            body.gravcomp = 1.0
    g = task.get("gripper") or {}
    if g.get("gain_scale", 1.0) != 1.0:  # the same firmer gripper the arm setup found it needs
        for act in robot.actuators:
            if act.name == g["actuator"]:
                act.gainprm[0] *= g["gain_scale"]
                act.biasprm[1] *= g["gain_scale"]
                act.biasprm[2] *= g["gain_scale"]
    if accurate:
        scene.option.timestep = 0.002
        scene.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
        scene.option.iterations, scene.option.ls_iterations = 50, 25
    else:
        scene.option.iterations, scene.option.ls_iterations = 4, 8
        for k, v in (task.get("physics") or {}).items():  # what physics_check found this robot needs to stay stable
            if k in ("impratio", "iterations", "ls_iterations"):
                setattr(scene.option, k, type(getattr(scene.option, k))(v))
    for k in ("integrator", "timestep", "impratio", "cone", "noslip_iterations", "ccd_iterations", "iterations",
              "ls_iterations", "tolerance", "solver"):  # the scene's physics settings win; say so quietly
        setattr(robot.option, k, getattr(scene.option, k))
    frame = scene.worldbody.frames[0]
    mx, my, myaw = task["mount"]
    if task.get("mount_pose"):  # a hand turned palm up for an in-hand goal (in_hand.py)
        frame.pos, frame.quat = task["mount_pose"]["pos"], task["mount_pose"]["quat"]
    elif not task["floating"]:
        from .introspect import floor_offset
        frame.pos = [mx, my, floor_offset(task["robot_xml"])]
        frame.quat = [math.cos(myaw / 2), 0, 0, math.sin(myaw / 2)]
    root = [b for b in range(1, rm.nbody) if rm.body_parentid[b] == 0][0]
    frame.attach_body(robot.body(rm.body(root).name), "robot_", "")
    return scene


def build_task_model(task: dict, accurate: bool = False) -> mujoco.MjModel:
    return build_task_spec(task, accurate).compile()


@dataclass
class Indices:
    home_adr: np.ndarray  # the robot's joints' qpos addresses ...
    home_val: np.ndarray  # ... and their values in its default pose
    joint_vadr: np.ndarray  # the same joints' qvel addresses
    joints: list  # and their names (without the robot_ prefix)
    q0: np.ndarray  # default command per actuator (actuator order): a position target, or 0 for speed/torque
    pos_mask: np.ndarray  # 1 for actuators that hold a position (their length is observed), else 0
    tips: np.ndarray  # end bodies
    base: int  # floating base body, -1 if bolted down
    base_q: int
    base_v: int
    block: int
    block_q: int
    block_v: int
    act_lo: np.ndarray
    act_hi: np.ndarray
    tip_off: np.ndarray | None = None  # per end part, its actual tip (fingertip, toe) in the part's own frame
    robot_geoms: np.ndarray | None = None  # for telling what the object is touching
    block_geoms: np.ndarray | None = None
    grip_act: int = -1  # the gripper's actuator (-1: none known), and its command when open / closed
    grip_open: float = 0.0
    grip_close: float = 1.0
    act_scale: np.ndarray | None = None  # per actuator: how far one unit of action moves its command
    act_step: np.ndarray | None = None  # and how far the command may move per control step
    jnt_lo: np.ndarray | None = None  # the robot's joints' ranges (home_adr order), for goals about its own body
    jnt_hi: np.ndarray | None = None
    tip_names: list | None = None  # the end parts' names (tips order, without robot_)
    hand: dict | None = None  # acting in hand space (hand_ctrl): the arm's joints, actuators and the hand's point
    tip_labels: list | None = None  # per end part, the finger it ends ("thumb", ...) or None
    # every finger's surface, as points on its parts (fingers.finger_points): touch for the whole finger, not only its tip
    finger_body: np.ndarray | None = None  # per point: the body it's on, its place in that body's frame, how thick the
    finger_off: np.ndarray | None = None   # part is there, and which finger (index into finger_labels)
    finger_rad: np.ndarray | None = None
    finger_of: np.ndarray | None = None
    finger_labels: list | None = None
    block_half: tuple = (0.0, 0.0, 0.0)  # the object's half sizes
    senses: tuple = ()  # extra things the policy sees, from what the goal is about (task["senses"]: "spin", "touch")


def is_position(m: mujoco.MjModel, a: int) -> bool:
    return m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE and m.actuator_biasprm[a, 1] < 0


def robot_joints(m: mujoco.MjModel, prefix: str = "robot_") -> list[int]:
    return [j for j in range(m.njnt) if m.joint(j).name.startswith(prefix)
            and int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))]


def indices(m: mujoco.MjModel, task: dict) -> Indices:
    """Actuators are observed through what they act on (MuJoCo's actuator length and velocity): a joint's
    angle, or a tendon's length for robots like the Stretch whose drive and telescoping arm are tendons."""
    lo = [m.actuator_ctrlrange[a, 0] for a in range(m.nu)]
    hi = [m.actuator_ctrlrange[a, 1] for a in range(m.nu)]
    if task.get("home"):
        js = [m.joint("robot_" + n).id for n in task["home"]]
        vals = list(task["home"].values())
    else:  # task files from before tendons were handled: every actuator drives one joint
        js = [int(m.actuator_trnid[a, 0]) for a in range(m.nu)]
        vals = task["q0"]
    home_adr = np.array([m.jnt_qposadr[j] for j in js], dtype=int)
    joint_vadr = np.array([m.jnt_dofadr[j] for j in js], dtype=int)
    names = [m.joint(j).name.removeprefix("robot_") for j in js]
    pos_mask = np.array([1.0 if is_position(m, a) else 0.0 for a in range(m.nu)])
    robot_bodies = [b for b in range(1, m.nbody) if m.body(b).name.startswith("robot_")]
    parents = {int(m.body_parentid[b]) for b in robot_bodies}
    tips = [b for b in robot_bodies if b not in parents and any(m.geom_bodyid[g] == b for g in range(m.ngeom))]
    if len(tips) > MAX_TIPS:  # keep the ones farthest from the robot's root (fingertips, feet, the tool)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        root = robot_bodies[0]
        tips = sorted(tips, key=lambda b: -np.linalg.norm(d.xpos[b] - d.xpos[root]))[:MAX_TIPS]
    base, base_q, base_v = -1, -1, -1
    if task["floating"]:
        fj = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.jnt_bodyid[j] in robot_bodies][0]
        base, base_q, base_v = int(m.jnt_bodyid[fj]), int(m.jnt_qposadr[fj]), int(m.jnt_dofadr[fj])
    tip_off = _tip_points(m, task, np.array(tips), home_adr, np.array(vals, dtype=float))
    act_scale, act_step = action_units(m, task.get("action_scale", 0.8))
    for a, (sc, st) in (task.get("action_units") or {}).items():  # widened to cover what the recipe does (demo_to_task)
        act_scale[int(a)], act_step[int(a)] = max(act_scale[int(a)], sc), max(act_step[int(a)], st)
    g_ticks = task.get("gripper_ticks")  # how many control ticks the gripper takes to cross its range
    if g_ticks and task.get("gripper"):
        try:
            a = int(m.actuator("robot_" + task["gripper"]["actuator"]).id)
            act_step[a] = act_scale[a] / float(g_ticks)
        except KeyError:
            pass
    g = task.get("gripper") or {}
    try:
        grip = (int(m.actuator("robot_" + g["actuator"]).id), float(g["open"]), float(g["close"])) if g else (-1, 0.0, 1.0)
    except KeyError:
        grip = (-1, 0.0, 1.0)
    bj = m.joint("block_free")
    rng = [(m.jnt_range[j][0], m.jnt_range[j][1]) if m.jnt_limited[j] and m.jnt_range[j][1] > m.jnt_range[j][0]
           else (-math.pi, math.pi) for j in js]
    from .fingers import finger_of_body
    fingers = finger_of_body(m)
    ix_extra = dict(jnt_lo=np.array([r[0] for r in rng]), jnt_hi=np.array([r[1] for r in rng]),
                    tip_names=[m.body(int(b)).name.removeprefix("robot_") for b in tips], hand=_hand_indices(m, task),
                    tip_labels=[fingers.get(int(b)) for b in tips], senses=tuple(task.get("senses") or ()),
                    block_half=(task["block"]["half_x"], task["block"]["half_y"], task["block"]["half_z"]))
    fp = finger_points(m)
    if fp is not None:
        ix_extra.update(finger_body=fp[0], finger_off=fp[1], finger_rad=fp[2], finger_of=fp[3], finger_labels=fp[4])
    return Indices(home_adr, np.array(vals, dtype=float), joint_vadr, names, np.array(task["q0"]), pos_mask,
                   np.array(tips, dtype=int), base, base_q, base_v,
                   int(m.body("block").id), int(bj.qposadr[0]), int(bj.dofadr[0]), np.array(lo), np.array(hi), tip_off,
                   np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in set(robot_bodies)], dtype=int),
                   np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("block").id], dtype=int), *grip,
                   act_scale, act_step, **ix_extra)


def finger_points(m: mujoco.MjModel):
    """Points along every finger's collision shapes (centre and both ends of each, in its body's frame), with how thick
    the shape is there: (bodies, offsets, radii, finger index per point, finger labels), or None without a hand."""
    from .fingers import hands
    hs = hands(m)
    if not hs:
        return None
    labels, body, off, rad, of = [], [], [], [], []
    for h in hs:
        for lab, chain in h["fingers"].items():
            k = len(labels)
            labels.append(lab)
            for b in chain:
                for g in range(m.ngeom):
                    if m.geom_bodyid[g] != b or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                        continue
                    c, half = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
                    Rg = np.zeros(9)
                    mujoco.mju_quat2Mat(Rg, m.geom_quat[g])
                    Rg = Rg.reshape(3, 3)
                    long = int(np.argmax(half))
                    r = float(np.sort(half)[1])  # the shape's thickness across its length
                    for t in (-1.0, 0.0, 1.0):
                        e = np.zeros(3)
                        e[long] = t * max(half[long] - r, 0.0)
                        body.append(b)
                        off.append(m.geom_pos[g] + Rg @ (c + e))
                        rad.append(r)
                        of.append(k)
    if not body:
        return None
    return np.array(body, dtype=int), np.array(off), np.array(rad), np.array(of, dtype=int), labels


def finger_gaps(xp, xpos, xmat, bp, R, half, ix: "Indices"):
    """How far each finger is from the object's surface (m, per finger_labels): its nearest point, less its thickness."""
    pts = xpos[ix.finger_body] + xp.einsum("kij,kj->ki", xmat[ix.finger_body].reshape(-1, 3, 3), xp.asarray(ix.finger_off))
    out = xp.maximum(xp.abs((pts - bp) @ R) - xp.asarray(half), 0.0)
    gap = xp.sqrt(xp.sum(out * out, axis=1)) - xp.asarray(ix.finger_rad)
    return xp.stack([xp.min(xp.where(xp.asarray(ix.finger_of) == k, gap, 9.0)) for k in range(len(ix.finger_labels))])


def _hand_indices(m: mujoco.MjModel, task: dict) -> dict | None:
    hs = task.get("hand_space")
    if task.get("action_space") != "hand" or not hs:
        return None
    try:
        jids = np.array([m.joint(n).id for n in hs["joints"]], dtype=int)
        acts = np.array([m.actuator(n).id for n in hs["acts"]], dtype=int)
    except KeyError:
        return None
    lo = np.array([m.jnt_range[j][0] if m.jnt_limited[j] else -math.pi for j in jids])
    hi = np.array([m.jnt_range[j][1] if m.jnt_limited[j] else math.pi for j in jids])
    return {"body": int(m.body(hs["body"]).id), "offset": np.array(hs["offset"]), "jids": jids, "acts": acts,
            "qadr": np.array([m.jnt_qposadr[j] for j in jids], dtype=int),
            "hinge": np.array([1.0 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE else 0.0 for j in jids]),
            "gear": np.array(hs["gears"]), "jlo": lo, "jhi": hi, "slots": acts[:6]}


def action_units(m: mujoco.MjModel, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """What one unit of the policy's output means for each actuator. Joints that turn: `scale` radians either
    side of home, at most MAX_STEP per step (2.5 rad/s), as before. Everything else (a gripper commanded 0..255,
    fingers that slide 4 cm, a lift, a telescoping arm, a drive): its whole range, crossed in about a second.
    One radian-sized setting for everything left grippers either frozen (0.8 out of 255) or slamming."""
    sc, st = np.full(m.nu, float(scale)), np.full(m.nu, MAX_STEP)
    for a in range(m.nu):
        hinge = (m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                 and int(m.jnt_type[m.actuator_trnid[a, 0]]) == int(mujoco.mjtJoint.mjJNT_HINGE))
        if not hinge:
            span = float(m.actuator_ctrlrange[a, 1] - m.actuator_ctrlrange[a, 0])
            if span > 0:
                sc[a], st[a] = span, span / 50.0
    return sc, st


def _tip_points(m: mujoco.MjModel, task: dict, tips, home_adr, home_val) -> np.ndarray:
    """Where each end part actually ends: the point of its shapes farthest from the joint it hangs from
    (a finger's tip, not its base 5 cm up), in the part's frame, measured at the home pose."""
    d = mujoco.MjData(m)
    d.qpos[home_adr] = home_val
    mujoco.mj_forward(m, d)
    out = np.zeros((len(tips), 3))
    for k, b in enumerate(tips):
        R, p = d.xmat[b].reshape(3, 3), d.xpos[b]
        pivot = d.xpos[m.body_parentid[b]] if m.body_jntnum[b] == 0 else d.xanchor[m.body_jntadr[b]]
        best, far = p, -1.0
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b:
                continue
            c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            Rg, pg = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
            for corner in np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c:
                w = pg + Rg @ corner
                dist = float(np.linalg.norm(w - pivot))
                if dist > far:
                    best, far = w, dist
        out[k] = R.T @ (best - p)
    return out


def obs_size(ix: Indices) -> int:
    n = len(ix.q0)
    extra = (3 if "spin" in ix.senses else 0) + (len(ix.finger_labels or []) if "touch" in ix.senses else 0)
    return 3 * n + 3 + 3 + 2 + 3 + 3 * len(ix.tips) + (12 if ix.base >= 0 else 0) + extra


def observe(xp, qvel, act_len, act_vel, xpos, xmat, last_act, ix: Indices, target) -> "array":
    """Same code for NumPy (xp=np) and JAX (xp=jax.numpy). act_len / act_vel: MuJoCo's actuator_length and
    actuator_velocity (a joint's angle and speed, or a tendon's length and speed)."""
    tx, ty, tz, tyaw = target
    q = (act_len - ix.q0) * ix.pos_mask  # a wheel's "length" grows without bound: speed-driven ones only give speed
    qd = act_vel * 0.1
    bp = xpos[ix.block]
    R = xmat[ix.block].reshape(3, 3)
    rel = bp - xp.array([tx, ty, tz])
    up = R[:, 2]
    yaw = xp.arctan2(R[1, 0], R[0, 0])
    head = xp.array([xp.cos(yaw - tyaw), xp.sin(yaw - tyaw)])
    tips = (xpos[ix.tips] - bp).reshape(-1)
    parts = [q, qd, rel, up, head, qvel[ix.block_v:ix.block_v + 3] * 0.5, tips, last_act]
    if ix.base >= 0:
        Rb = xmat[ix.base].reshape(3, 3)
        parts += [xpos[ix.base] - bp, Rb.T @ xp.array([0.0, 0.0, -1.0]),
                  Rb.T @ qvel[ix.base_v:ix.base_v + 3] * 0.5, qvel[ix.base_v + 3:ix.base_v + 6] * 0.2]
    if "spin" in ix.senses:  # how it turns (world frame, rad/s x 0.1): the goal is about spin, and that isn't in the rest
        parts.append(R @ qvel[ix.block_v + 3:ix.block_v + 6] * 0.1)
    if "touch" in ix.senses and ix.finger_labels:  # per finger, 1 touching .. 0 at 2 cm or more (a sense of touch)
        gaps = finger_gaps(xp, xpos, xmat, bp, R, ix.block_half, ix)
        parts.append(xp.clip(1.0 - gaps / 0.02, 0.0, 1.0))
    return xp.concatenate(parts)


class Residual:
    """Learning corrections on top of a recipe (residual learning): the action sent is the recipe's action at
    this moment plus `scale` x the policy's output, so a policy that outputs nothing replays the recipe, and
    practice can only add fixes to it (follow the object when it's a little off, grip a bit lower). The policy
    also sees how far through the recipe it is (phase). Same code for JAX and NumPy."""

    def __init__(self, base, scale: float):
        self.base, self.scale, self.T = np.asarray(base, dtype=np.float32), float(scale), len(base)

    @classmethod
    def of(cls, task: dict, folder) -> "Residual | None":
        r = task.get("residual")
        if not r or not task.get("demo") or not (Path(folder) / task["demo"]).exists():
            return None
        z = np.load(Path(folder) / task["demo"])
        return cls(z["base"], r.get("scale", 0.3)) if "base" in z else None

    def action(self, xp, t, a):
        base = xp.asarray(self.base)[xp.minimum(t, self.T - 1)]
        return xp.clip(base + self.scale * a, -1.0, 1.0)

    def phase(self, xp, t):
        return xp.reshape(xp.minimum(xp.asarray(t, dtype=xp.float32) / self.T, 1.5), (1,))


def write_task(path: Path, *, robot_xml: str, block: BlockSpec, start, target: dict, mount, floating: bool,
               q0, conditions: list[dict], scene_path: str, robot_name: str, home: dict | None = None,
               fixtures: dict | None = None) -> dict:
    task = {"robot_xml": robot_xml, "robot": robot_name, "scene": scene_path,
            "block": {k: (list(v) if isinstance(v, tuple) else v) for k, v in
                      ((k, getattr(block, k)) for k in ("half_x", "half_y", "half_z", "mass", "friction", "fixtures_xml",
                                                        "shape", "mesh_file", "mesh_scale", "mesh_pos", "mesh_quat"))},
            "start": list(start), "target": target, "mount": list(mount), "floating": floating, "q0": list(map(float, q0)),
            "home": home or {}, "fixtures": fixtures or {}, "conditions": conditions, "action_scale": 0.5 if floating else 0.8, "episode_s": 8.0,
            "ctrl_dt": CTRL_DT, "sim_dt": SIM_DT}
    path.parent.mkdir(parents=True, exist_ok=True)
    if home:  # the default command per actuator, measured on the model as trained (tendon lengths included)
        m = build_task_model(task)
        d = mujoco.MjData(m)
        for n, v in home.items():
            d.qpos[m.jnt_qposadr[m.joint("robot_" + n).id]] = v
        mujoco.mj_forward(m, d)
        task["q0"] = [float(d.actuator_length[a]) if is_position(m, a) else 0.0 for a in range(m.nu)]
    path.write_text(json.dumps(task, indent=1))
    return task


def set_home(m: mujoco.MjModel, qpos, ix: Indices):
    qpos[ix.home_adr] = ix.home_val
    return qpos


def opposite_sides(d: mujoco.MjData, block_geoms, robot_geoms, centre) -> bool:
    """Do the robot's geoms touch the object from two opposite sides (a pinch), whatever else touches it? Two fingers
    resting on its top, or one pressing on one side, is not."""
    pts = []
    for i in range(d.ncon):
        con = d.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        if con.dist <= 0.002 and ((g1 in block_geoms and g2 in robot_geoms) or (g2 in block_geoms and g1 in robot_geoms)):
            pts.append(con.pos - centre)
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            na, nb = np.linalg.norm(pts[a]), np.linalg.norm(pts[b])
            if na > 1e-6 and nb > 1e-6 and float(pts[a] @ pts[b]) / (na * nb) < -0.3:
                return True
    return False


def contact_held(m: mujoco.MjModel, d: mujoco.MjData, ix: Indices) -> float:
    """Is the robot really holding the object? From the simulation's contacts: the robot touches it on two
    opposite sides (a pinch: fingers, a jaw, two hands) and nothing else touches it (not the floor, not a
    shelf or bin), so the robot alone is carrying it. Used whenever Morph shows or judges "held"."""
    robot, block = set(ix.robot_geoms.tolist()), set(ix.block_geoms.tolist())
    c0 = d.xpos[ix.block]
    pts = []
    for i in range(d.ncon):
        con = d.contact[i]
        if con.dist > 0.002:
            continue
        g1, g2 = int(con.geom1), int(con.geom2)
        if g1 in block or g2 in block:
            other = g2 if g1 in block else g1
            if other not in robot:
                return 0.0  # resting on (or pressed against) something else
            pts.append(con.pos - c0)
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            na, nb = np.linalg.norm(pts[a]), np.linalg.norm(pts[b])
            if na > 1e-6 and nb > 1e-6 and float(pts[a] @ pts[b]) / (na * nb) < -0.3:
                return 1.0  # touched from opposite sides
    return 0.0


def contact_touch(m: mujoco.MjModel, d: mujoco.MjData, ix: Indices) -> dict:
    """Which fingers and parts touch the object, from the contacts (fingers.touching): for measure(touch=...)."""
    from .fingers import touching
    return touching(m, d, ix.block)


def tip_touch(xp, pts, bp, R, half, ix: Indices, xpos=None, xmat=None) -> dict:
    """The learner's stand-in for contact_touch: an end part within 6 mm of the object's surface touches it; with a
    hand (and xpos, xmat), a finger touches if any part of it does (the judge's contacts count a thumb's base too:
    with tips alone, practice never saw the thumb resting on the pen that failed every try)."""
    out_ = xp.maximum(xp.abs((pts - bp) @ R) - xp.asarray(half), 0.0)
    near = (xp.sqrt(xp.sum(out_ * out_, axis=1)) < 0.006).astype(xp.float32)
    out = {n: near[k] for k, n in enumerate(ix.tip_names or [])}
    if ix.finger_labels and xpos is not None:
        gaps = finger_gaps(xp, xpos, xmat, bp, R, half, ix)
        labels = list(ix.finger_labels)
        for k, lab in enumerate(labels):
            out[lab] = (gaps[k] < 0.006).astype(xp.float32)
    else:
        labels = sorted({lab for lab in (ix.tip_labels or []) if lab})
        for lab in labels:
            out[lab] = xp.max(xp.stack([near[k] for k, x in enumerate(ix.tip_labels) if x == lab]))
    out["fingers_touching"] = sum(out[lab] for lab in labels) if labels else xp.sum(near)
    return out


def pinch_held(xp, pts, bp, R, half, task: dict):
    """The learner's stand-in for contact_held (contacts aren't available per copy on the GPU): two of the
    robot's tips at opposite faces of the object, and the object raised above whatever is under it."""
    L = (pts - bp) @ R  # tips in the object's frame
    h = xp.asarray(half)
    near = xp.all(xp.abs(L) <= h + 0.025, axis=1)  # each tip at (or just outside) the object's surface
    pinch = xp.zeros(())
    for ax in (0, 1):
        pos = near & (L[:, ax] > h[ax] - 0.012)
        neg = near & (L[:, ax] < -(h[ax] - 0.012))
        pinch = xp.maximum(pinch, (xp.any(pos) & xp.any(neg)).astype(xp.float32))
    support = xp.zeros(())  # the highest surface under the object: floor or a fixture's top
    for f in task.get("fixtures", {}).values():
        under = (f["lo"][0] <= bp[0]) & (bp[0] <= f["hi"][0]) & (f["lo"][1] <= bp[1]) & (bp[1] <= f["hi"][1])
        support = xp.maximum(support, xp.where(under, f["top_z"] if f["kind"] != "container" else f.get("rest_z", 0.0), 0.0))
    lowest = bp[2] - (xp.abs(R[2, 0]) * half[0] + xp.abs(R[2, 1]) * half[1] + xp.abs(R[2, 2]) * half[2])
    return pinch * (lowest > support + 0.005).astype(xp.float32)


def grip_terms(xp, pts, bp, R, half, task: dict, closure=None):
    """Graded versions of "held", for any body (fingertips, a jaw, two feet, two hands): how close the robot's
    tips are to opposite faces of the object (0..1), how far the nearest tip is from its surface (m), and how
    far it's lifted above whatever is under it (m). They give the learner a direction long before it ever
    manages a real grip."""
    L = (pts - bp) @ R  # tips in the object's frame
    h = xp.asarray(half)
    out = xp.maximum(xp.abs(L) - h, 0.0)  # how far outside the box, per axis
    touch = xp.min(xp.sqrt(xp.sum(out * out, axis=1)))
    pinch = xp.zeros(())
    for ax, others in ((0, (1, 2)), (1, (0, 2))):
        side = xp.sum(xp.stack([out[:, o] for o in others]), axis=0)  # beside the face rather than on it
        # within 2.5 cm of the face counts fully (a fingertip point sits a little outside the surface it presses on,
        # as in pinch_held); further away fades over a few cm
        c_pos = xp.exp(-xp.maximum(xp.abs(L[:, ax] - h[ax]) + side - 0.025, 0.0) / 0.03)
        c_neg = xp.exp(-xp.maximum(xp.abs(L[:, ax] + h[ax]) + side - 0.025, 0.0) / 0.03)
        pinch = xp.maximum(pinch, xp.max(c_pos) * xp.max(c_neg))
    support = xp.zeros(())  # the highest surface under the object: floor or a fixture's top
    for f in task.get("fixtures", {}).values():
        under = (f["lo"][0] <= bp[0]) & (bp[0] <= f["hi"][0]) & (f["lo"][1] <= bp[1]) & (bp[1] <= f["hi"][1])
        support = xp.maximum(support, xp.where(under, f["top_z"] if f["kind"] != "container" else f.get("rest_z", 0.0), 0.0))
    lowest = bp[2] - (xp.abs(R[2, 0]) * half[0] + xp.abs(R[2, 1]) * half[1] + xp.abs(R[2, 2]) * half[2])
    lifted = xp.maximum(lowest - support, 0.0)
    up = xp.clip(lifted / 0.02, 0.0, 1.0)
    if closure is None:  # no gripper motor known: around it, then off the ground
        grip = pinch * (0.4 + 0.6 * up)
    else:  # around it (0.25), closing on it (up to 0.6), lifting it (to 1): squeezing gets credit before it lifts
        grip = pinch * (0.25 + 0.35 * closure + 0.4 * closure * up)
    return grip, touch, lifted


def _goal_xyz(task: dict):
    """Where the goal puts the object (goal_x, goal_y, goal_z in formulas), when it has a place."""
    t = task.get("target") or {}
    return (t["x"], t["y"], t.get("z", 0.0)) if t.get("has_place", True) and "x" in t else None


def measure(xp, xpos, xmat, qvel, ix: Indices, task: dict, held=None, ctrl=None, qpos=None, touch=None) -> dict:
    """Every quantity a goal condition or a skill step can read (goal_terms), from the simulation state.
    Same code for training (JAX) and trying / practice runs (NumPy). held: from contact_held when the
    contacts are at hand (NumPy); otherwise the learner's geometric stand-in. ctrl: the commands (for how far the
    gripper is closed). touch: contact_touch when the contacts are at hand, else tip_touch."""
    from .goal_terms import quantities
    b = task["block"]
    half = (b["half_x"], b["half_y"], b["half_z"])
    r_obj = float(math.hypot(b["half_x"], b["half_y"]))
    bp = xpos[ix.block]
    R = xmat[ix.block].reshape(3, 3)
    pts = xpos[ix.tips]
    if ix.tip_off is not None and len(ix.tips):  # the actual fingertips / toes, not the parts' origins
        pts = pts + xp.einsum("kij,kj->ki", xmat[ix.tips].reshape(-1, 3, 3), xp.asarray(ix.tip_off))
    if not len(ix.tips):  # a robot with no shapes at all (meshes missing): a stand-in part, nowhere near the object
        pts = xp.reshape(bp + 9.0, (1, 3))
    d_tips = xp.linalg.norm(pts - bp, axis=1)
    hand = xp.min(d_tips)
    hand_pos = pts[xp.argmin(d_tips)]
    if held is None:
        held = pinch_held(xp, pts, bp, R, half, task)
    if ix.base >= 0:
        robot, base_z, base_up = xpos[ix.base][:2], xpos[ix.base][2], xmat[ix.base].reshape(3, 3)[2, 2]
    else:
        robot, base_z, base_up = xp.asarray(task["mount"][:2]), xp.asarray(0.0), xp.asarray(1.0)
    v = quantities(xp, pos=bp, R=R, vel=qvel[ix.block_v:ix.block_v + 3], ang=qvel[ix.block_v + 3:ix.block_v + 6],
                   start=task["start"][:2], half=half,
                   fixtures=task.get("fixtures", {}), held=held, robot_xy=robot, hand_dist=hand, hand_pos=hand_pos,
                   base_z=base_z, base_up=base_up, goal=_goal_xyz(task))
    closure = None
    if ctrl is not None and ix.grip_act >= 0 and ix.grip_close != ix.grip_open:
        closure = xp.clip((ctrl[ix.grip_act] - ix.grip_open) / (ix.grip_close - ix.grip_open), 0.0, 1.0)
    v["grip"], v["touch_dist"], v["lifted"] = grip_terms(xp, pts, bp, R, half, task, closure)
    v["closure"] = closure if closure is not None else xp.asarray(0.0)
    v["grip"] = xp.maximum(v["grip"], xp.asarray(held, dtype=float) if xp is np else held)  # a real hold is a full grip
    v["parts"] = {n: pts[k] for k, n in enumerate(ix.tip_names or [])}  # the robot's own body (goal_terms BODY_FUNCS)
    v["touch"] = touch if touch is not None else tip_touch(xp, pts, bp, R, half, ix, xpos, xmat)
    v["fingers_touching"] = v["touch"]["fingers_touching"]
    if qpos is not None and ix.jnt_lo is not None:
        frac = (qpos[ix.home_adr] - ix.jnt_lo) / (ix.jnt_hi - ix.jnt_lo)
        v["joints"] = {n: frac[k] for k, n in enumerate(ix.joints)}
    return v


def soft(v: dict) -> dict:
    """The same quantities with "held" graded (grip) instead of yes/no: for the learner's hints only; whether a
    step or the goal is done is always decided on the strict ones."""
    return dict(v, held=v["grip"]) if "grip" in v else v
