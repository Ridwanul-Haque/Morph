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


def next_ctrl(xp, prev, action, q0, scale, lo, hi):
    """The joint targets for this step: head towards q0 + scale * action, at most MAX_STEP per step.
    Same code for training (JAX) and trying / export (NumPy)."""
    want = xp.clip(q0 + action * scale, lo, hi)
    return prev + xp.clip(want - prev, -MAX_STEP, MAX_STEP)


def load_task(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def build_task_spec(task: dict, accurate: bool = False) -> mujoco.MjSpec:
    """accurate=False: the fast physics the GPU trains in (4 ms steps, 4 solver iterations).
    accurate=True: finer physics for trying the policy and for the exported model (2 ms steps, implicit
    integrator, 50 iterations). A policy that works in both has passed a sim-to-sim check; the UR5e's
    first wall-push policy blew up the coarse CPU solver but succeeded 6/6 in the accurate one."""
    blk = BlockSpec(**task["block"])
    scene = mujoco.MjSpec.from_string(f"""<mujoco model="morph_task">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{SIM_DT}" integrator="implicitfast" cone="elliptic" impratio="10"/>
  {_common_assets()}
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
    if accurate:
        scene.option.timestep = 0.002
        scene.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
        scene.option.iterations, scene.option.ls_iterations = 50, 25
    else:
        scene.option.iterations, scene.option.ls_iterations = 4, 8
    for k in ("integrator", "timestep", "impratio", "cone", "noslip_iterations", "ccd_iterations", "iterations",
              "ls_iterations", "tolerance", "solver"):  # the scene's physics settings win; say so quietly
        setattr(robot.option, k, getattr(scene.option, k))
    frame = scene.worldbody.frames[0]
    mx, my, myaw = task["mount"]
    if not task["floating"]:
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
    bj = m.joint("block_free")
    return Indices(home_adr, np.array(vals, dtype=float), joint_vadr, names, np.array(task["q0"]), pos_mask,
                   np.array(tips), base, base_q, base_v,
                   int(m.body("block").id), int(bj.qposadr[0]), int(bj.dofadr[0]), np.array(lo), np.array(hi), tip_off,
                   np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in set(robot_bodies)]),
                   np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("block").id]))


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
    return 3 * n + 3 + 3 + 2 + 3 + 3 * len(ix.tips) + (12 if ix.base >= 0 else 0)


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
    return xp.concatenate(parts)


def write_task(path: Path, *, robot_xml: str, block: BlockSpec, start, target: dict, mount, floating: bool,
               q0, conditions: list[dict], scene_path: str, robot_name: str, home: dict | None = None,
               fixtures: dict | None = None) -> dict:
    task = {"robot_xml": robot_xml, "robot": robot_name, "scene": scene_path,
            "block": {k: getattr(block, k) for k in ("half_x", "half_y", "half_z", "mass", "friction", "fixtures_xml")},
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


def measure(xp, xpos, xmat, qvel, ix: Indices, task: dict, held=None) -> dict:
    """Every quantity a goal condition or a skill step can read (goal_terms), from the simulation state.
    Same code for training (JAX) and trying / practice runs (NumPy). held: from contact_held when the
    contacts are at hand (NumPy); otherwise the learner's geometric stand-in."""
    from .goal_terms import quantities
    b = task["block"]
    half = (b["half_x"], b["half_y"], b["half_z"])
    r_obj = float(math.hypot(b["half_x"], b["half_y"]))
    bp = xpos[ix.block]
    R = xmat[ix.block].reshape(3, 3)
    pts = xpos[ix.tips]
    if ix.tip_off is not None:  # the actual fingertips / toes, not the parts' origins
        pts = pts + xp.einsum("kij,kj->ki", xmat[ix.tips].reshape(-1, 3, 3), xp.asarray(ix.tip_off))
    d_tips = xp.linalg.norm(pts - bp, axis=1)
    hand = xp.min(d_tips)
    hand_pos = pts[xp.argmin(d_tips)]
    if held is None:
        held = pinch_held(xp, pts, bp, R, half, task)
    if ix.base >= 0:
        robot, base_z, base_up = xpos[ix.base][:2], xpos[ix.base][2], xmat[ix.base].reshape(3, 3)[2, 2]
    else:
        robot, base_z, base_up = xp.asarray(task["mount"][:2]), xp.asarray(0.0), xp.asarray(1.0)
    return quantities(xp, pos=bp, R=R, vel=qvel[ix.block_v:ix.block_v + 3], start=task["start"][:2], half=half,
                      fixtures=task.get("fixtures", {}), held=held, robot_xy=robot, hand_dist=hand, hand_pos=hand_pos,
                      base_z=base_z, base_up=base_up)
