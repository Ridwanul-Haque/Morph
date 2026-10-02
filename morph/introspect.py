"""Robot introspection: work out what a robot is from its model file alone.

Given any MJCF (or URDF that MuJoCo can load), produce a RobotProfile:
base type (fixed / planar-mobile / floating), legs, arms, end effectors, gripper,
size, mass, reach and actuators, plus a Capabilities description derived from it.

Nothing here is specific to a particular robot; it only looks at the kinematic
tree, joint/actuator types and geometry.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .robots.base import Capabilities

FREE, SLIDE = int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_SLIDE)
TRN_JOINT = (int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT))
TRN_TENDON, TRN_SITE = int(mujoco.mjtTrn.mjTRN_TENDON), int(mujoco.mjtTrn.mjTRN_SITE)
WRAP_JOINT = int(mujoco.mjtWrap.mjWRAP_JOINT)
MESH, PLANE = int(mujoco.mjtGeom.mjGEOM_MESH), int(mujoco.mjtGeom.mjGEOM_PLANE)

GRIPPER_WORDS = re.compile(r"finger|fngr|gripper|jaw|knuckle|pad|claw|pinch|hand_?[lr]|left_?pad|right_?pad", re.I)
FOOT_WORDS = re.compile(r"foot|feet|toe|ankle|shin|calf|paw", re.I)


@dataclass
class Chain:
    """A kinematic chain from a branch point down to a leaf body."""

    bodies: list[str]
    joints: list[str]
    actuated_joints: list[str]
    leaf: str
    leaf_height: float  # m, at the home pose
    length: float  # summed body-origin distances along the chain (m)
    role: str = "unknown"  # leg | arm | gripper | rotor | other


@dataclass
class RobotProfile:
    name: str
    source: str
    base: str  # fixed | planar_mobile | wheeled | floating
    kind: str  # fixed_arm | mobile_manipulator | quadruped | biped | humanoid | legged | wheeled_base | aerial | unknown
    n_legs: int
    n_arms: int
    has_gripper: bool
    gripper: dict
    mass_kg: float
    size_m: dict  # height, length, width at the home pose
    reach_m: dict  # per arm: max horizontal reach from its base, z range of the end effector
    dof: int
    actuators: list[dict]
    chains: list[dict]
    sensors: list[str]
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        bits = [f"{self.name}: {self.kind.replace('_', ' ')}", f"{self.dof} DoF", f"{self.mass_kg:.1f} kg",
                f"{self.size_m['height']:.2f} m tall"]
        if self.n_legs:
            bits.append(f"{self.n_legs} legs")
        if self.n_arms:
            r = max((v["max_horizontal"] for v in self.reach_m.values()), default=0)
            bits.append(f"{self.n_arms} arm(s), reach {r:.2f} m")
        if self.has_gripper:
            kind = self.gripper.get("type", "gripper").replace("_", " ")
            bits.append(f"{kind} opens {self.gripper.get('max_opening_m', 0) * 100:.1f} cm")
        elif self.gripper.get("type") == "multi_finger_hand":
            bits.append(f"{self.gripper['n_fingers']}-finger hands")
        return ", ".join(bits)


# ----------------------------------------------------------------------------------------- loading
def load_model(path: str | Path) -> mujoco.MjModel:
    path = Path(path)
    return mujoco.MjModel.from_xml_path(str(path))


def home_qpos(m: mujoco.MjModel) -> np.ndarray:
    for k in range(m.nkey):
        if m.key(k).name.lower() in ("home", "default", "stand", "standing", "init"):
            return m.key_qpos[k].copy()
    if m.nkey:
        return m.key_qpos[0].copy()
    return m.qpos0.copy()


def _geom_lowest(m: mujoco.MjModel, d: mujoco.MjData, g: int) -> float:
    """Lowest world z of one geom (exact for meshes and primitives)."""
    R, p, t, s = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_type[g], m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        return float((v @ R.T)[:, 2].min() + p[2])
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(p[2] - s[0])
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return float(p[2] - abs(R[2, 2]) * s[1] - s[0])
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return float(p[2] - np.abs(R[2, :]) @ s[:3])
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return float(p[2] - abs(R[2, 2]) * s[1] - math.hypot(R[2, 0], R[2, 1]) * s[0])
    return float(p[2] - m.geom_rbound[g])


_FLOOR = {}


def floor_offset(path: str | Path) -> float:
    """How far to raise (or lower) a bolted-down robot's mount so its base sits on the floor: some model
    files put the base above their origin (the Menagerie xArm7's is 12 cm up) or below it. The base is
    the root body and whatever is rigidly attached to it, before the first joint. 0 for floating robots."""
    key = str(path)
    if key not in _FLOOR:
        m = mujoco.MjModel.from_xml_path(key)
        if any(m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in range(m.njnt)):
            _FLOOR[key] = 0.0
            return 0.0
        d = mujoco.MjData(m)
        d.qpos[:] = home_qpos(m)
        mujoco.mj_forward(m, d)
        rigid = {b for b in range(1, m.nbody) if m.body_jntnum[b] == 0}
        base = set()
        for b in range(1, m.nbody):  # bodies reachable from the world without crossing a joint
            chain, ok = b, True
            while chain:
                if chain not in rigid:
                    ok = False
                    break
                chain = int(m.body_parentid[chain])
            if ok:
                base.add(b)
        lows = [_geom_lowest(m, d, g) for g in range(m.ngeom) if m.geom_bodyid[g] in base]
        _FLOOR[key] = -min(lows) if lows else 0.0
    return _FLOOR[key]


def _name(m, objtype, i) -> str:
    return mujoco.mj_id2name(m, objtype, i) or f"{mujoco.mjtObj(objtype).name.lower()[6:]}{i}"


# ----------------------------------------------------------------------------------------- analysis
def _root_bodies(m) -> list[int]:
    return [b for b in range(1, m.nbody) if m.body_parentid[b] == 0]


def _base_type(m, root: int) -> str:
    js = [m.body_jntadr[root] + k for k in range(m.body_jntnum[root])] if m.body_jntnum[root] else []
    types = [int(m.jnt_type[j]) for j in js]
    if FREE in types:
        return "floating"
    # planar mobile base: slide joints on the root (possibly nested one level for x/y/yaw stacks)
    body, depth = root, 0
    slides = 0
    while depth < 3:
        for k in range(m.body_jntnum[body]):
            if int(m.jnt_type[m.body_jntadr[body] + k]) == SLIDE:
                slides += 1
        kids = [b for b in range(m.nbody) if m.body_parentid[b] == body and b != body]
        if len(kids) != 1:
            break
        body, depth = kids[0], depth + 1
    return "planar_mobile" if slides >= 2 else "fixed"


def _children(m) -> dict[int, list[int]]:
    ch: dict[int, list[int]] = {b: [] for b in range(m.nbody)}
    for b in range(1, m.nbody):
        ch[m.body_parentid[b]].append(b)
    return ch


def _actuated_joints(m) -> dict[int, list[int]]:
    """joint id -> actuator ids driving it (directly or through a tendon)."""
    out: dict[int, list[int]] = {}
    for a in range(m.nu):
        t, tid = int(m.actuator_trntype[a]), int(m.actuator_trnid[a, 0])
        if t in TRN_JOINT:
            out.setdefault(tid, []).append(a)
        elif t == TRN_TENDON:
            adr, num = m.tendon_adr[tid], m.tendon_num[tid]
            for w in range(adr, adr + num):
                if int(m.wrap_type[w]) == WRAP_JOINT:
                    out.setdefault(int(m.wrap_objid[w]), []).append(a)
    return out


def _chains(m, d, root: int, act_j) -> list[Chain]:
    """Split the tree into chains at every branching body."""
    ch = _children(m)
    chains = []

    def body_joints(b):
        return [m.body_jntadr[b] + k for k in range(m.body_jntnum[b])]

    def walk(start: int):
        seq = [start]
        b = start
        while len(ch[b]) == 1:
            b = ch[b][0]
            seq.append(b)
        # seq ends at a leaf or a branch point
        joints = [int(j) for bb in seq for j in body_joints(bb) if int(m.jnt_type[j]) != FREE]
        length = sum(float(np.linalg.norm(d.xpos[seq[i]] - d.xpos[seq[i - 1]])) for i in range(1, len(seq)))
        if start != root:
            length += float(np.linalg.norm(d.xpos[start] - d.xpos[m.body_parentid[start]]))
        leaf = seq[-1]
        chains.append(Chain(bodies=[_name(m, mujoco.mjtObj.mjOBJ_BODY, x) for x in seq],
                            joints=[_name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints],
                            actuated_joints=[_name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints if j in act_j],
                            leaf=_name(m, mujoco.mjtObj.mjOBJ_BODY, leaf),
                            leaf_height=float(_lowest_geom_z(m, d, leaf)), length=length))
        for c in ch[leaf]:
            walk(c)

    walk(root)
    return chains


def _lowest_geom_z(m, d, body: int) -> float:
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body and int(m.geom_type[g]) != PLANE]
    return float(_bbox(m, d, gs)[0][2]) if gs else float(d.xpos[body][2])


def _geom_points(m, d, g) -> np.ndarray:
    """Points bounding a geom in world coordinates (mesh vertices, or the 8 corners of its local box)."""
    if int(m.geom_type[g]) == MESH:
        mid = m.geom_dataid[g]
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        return m.mesh_vert[va:va + vn] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    s = m.geom_size[g]
    t = int(m.geom_type[g])
    if t == int(mujoco.mjtGeom.mjGEOM_BOX):
        h = s[:3]
    elif t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        h = np.array([s[0]] * 3)
    elif t in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
        h = np.array([s[0], s[0], s[1] + (s[0] if t == int(mujoco.mjtGeom.mjGEOM_CAPSULE) else 0)])
    elif t == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        h = s[:3]
    else:
        h = np.array([m.geom_rbound[g]] * 3)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h
    return corners @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]


def _robot_geoms(m, root: int) -> list[int]:
    def under(b):
        while b:
            if b == root:
                return True
            b = m.body_parentid[b]
        return False
    return [g for g in range(m.ngeom) if under(m.geom_bodyid[g]) and int(m.geom_type[g]) != PLANE]


def _bbox(m, d, geoms: list[int]) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in geoms:
        v = _geom_points(m, d, g)
        lo, hi = np.minimum(lo, v.min(0)), np.maximum(hi, v.max(0))
    return lo, hi


def _classify_chains(m, d, chains: list[Chain], base: str, floor_z: float) -> None:
    for c in chains:
        n_act = len(c.actuated_joints)
        near_floor = c.leaf_height - floor_z < 0.05
        if GRIPPER_WORDS.search(" ".join(c.bodies)) and n_act <= 2 and c.length < 0.2:
            c.role = "gripper"
        elif base == "floating" and near_floor and n_act >= 2:
            c.role = "leg"
        elif n_act >= 4 and not near_floor:
            c.role = "arm"
        elif FOOT_WORDS.search(c.leaf) and base == "floating":
            c.role = "leg"
        else:
            c.role = "other"


def finger_sides(m, d, finger_bodies: list[int], direct: list[int] | None, qpos0: np.ndarray) -> tuple[list[int], list[int]]:
    """Split finger bodies into the two sides of a parallel gripper; returns the fingertip body of each side."""
    # by name if possible, otherwise by which branch they hang off, otherwise by position
    names = [_name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in finger_bodies]
    left = [b for b, n in zip(finger_bodies, names) if re.search(r"left|_l\b|^l_|\bl\b", n, re.I)]
    right = [b for b, n in zip(finger_bodies, names) if re.search(r"right|_r\b|^r_|\br\b", n, re.I)]
    if (not left or not right) and direct and len(direct) == 2:
        def under(b, top):
            while b and b != top:
                b = m.body_parentid[b]
            return b == top
        left = [b for b in finger_bodies if under(b, direct[0])] or [direct[0]]
        right = [b for b in finger_bodies if under(b, direct[1])] or [direct[1]]
    if not left or not right:
        d.qpos[:] = qpos0
        mujoco.mj_kinematics(m, d)
        pts = np.array([d.xpos[b] for b in finger_bodies])
        axis = pts.max(0) - pts.min(0)
        k = int(np.argmax(axis))
        mid = pts[:, k].mean()
        left = [b for b, p in zip(finger_bodies, pts) if p[k] >= mid]
        right = [b for b, p in zip(finger_bodies, pts) if p[k] < mid]
    if not left or not right:
        half = max(1, len(finger_bodies) // 2)
        left, right = finger_bodies[:half], finger_bodies[half:] or finger_bodies[:1]

    # keep only the fingertip on each side: the finger body farthest from where the gripper attaches
    d.qpos[:] = qpos0
    mujoco.mj_kinematics(m, d)
    attach = d.xpos[_common_ancestor(m, finger_bodies)]
    left = [max(left, key=lambda x: np.linalg.norm(d.xpos[x] - attach))]
    right = [max(right, key=lambda x: np.linalg.norm(d.xpos[x] - attach))]

    return left, right


def _gripper_opening(m, finger_bodies: list[int], act_ids: list[int], qpos0: np.ndarray,
                     sides: tuple[list[int], list[int]] | None = None, direct: list[int] | None = None) -> dict:
    """Sweep the gripper joints across their range and measure the widest gap between the two
    finger sides' collision geoms (the usable opening)."""
    d = mujoco.MjData(m)
    joints = sorted({j for j, acts in _actuated_joints(m).items() if set(acts) & set(act_ids)})
    if not joints or (len(finger_bodies) < 2 and sides is None):
        return {}
    if sides is not None:
        return _jaw_travel(m, d, sides[0][0], act_ids, qpos0)
    left, right = finger_sides(m, d, finger_bodies, direct, qpos0)
    return _measure_gap(m, d, left, right, act_ids, qpos0, include_children=True)


def _settle_at_ends(m, d, act_ids: list[int], qpos0: np.ndarray, measure) -> list:
    """Drive the given actuators to each end of their range (gravity off, rest of the robot held)
    and call measure(d) once the physics has settled."""
    gravity = m.opt.gravity.copy()
    m.opt.gravity[:] = 0
    hold = _hold_ctrl(m, qpos0)
    out = []
    try:
        for end in (0, 1):
            mujoco.mj_resetData(m, d)
            d.qpos[:] = qpos0
            d.ctrl[:] = hold
            for a in act_ids:
                lo, hi = m.actuator_ctrlrange[a] if m.actuator_ctrllimited[a] else (-1.0, 1.0)
                d.ctrl[a] = lo if end == 0 else hi
            for _ in range(int(1.5 / m.opt.timestep)):
                mujoco.mj_step(m, d)
            out.append(measure(d))
    finally:
        m.opt.gravity[:] = gravity
    return out


def _jaw_travel(m, d, jaw: int, act_ids: list[int], qpos0: np.ndarray) -> dict:
    """Single moving jaw: how far its tip travels between closed and open (the usable opening)."""
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == jaw]
    if not gs:
        return {}
    mujoco.mj_resetData(m, d)
    d.qpos[:] = qpos0
    mujoco.mj_kinematics(m, d)
    pts = np.vstack([_geom_points(m, d, g) for g in gs])
    local = (pts - d.xpos[jaw]) @ d.xmat[jaw].reshape(3, 3)
    tip = local[np.argmax(np.linalg.norm(local, axis=1))]  # jaw point farthest from its hinge body origin
    ends = _settle_at_ends(m, d, act_ids, qpos0, lambda dd: dd.xpos[jaw] + dd.xmat[jaw].reshape(3, 3) @ tip)
    return {"max_opening_m": round(float(np.linalg.norm(ends[1] - ends[0])), 4),
            "method": "single jaw: tip travel between the two ends of the actuator range (upper-bound estimate)"}


def _measure_gap(m, d, left: list[int], right: list[int], act_ids: list[int], qpos0: np.ndarray,
                 include_children: bool) -> dict:
    def side_geoms(bodies):
        sub = set(bodies)
        if include_children:
            for b in range(m.nbody):  # include bodies below the finger (pads etc.)
                p = b
                while p and p not in sub:
                    p = m.body_parentid[p]
                if p:
                    sub.add(b)
        return [g for g in range(m.ngeom) if m.geom_bodyid[g] in sub and (m.geom_contype[g] or m.geom_conaffinity[g])]
    ga, gb = side_geoms(left), side_geoms(right)
    if not ga or not gb:
        return {}
    # Drive the gripper actuators to each end of their control range and let physics settle
    # (handles linkages and equality constraints); everything else holds its home pose.
    fromto = np.zeros(6)
    gravity = m.opt.gravity.copy()
    m.opt.gravity[:] = 0
    hold = _hold_ctrl(m, qpos0)
    best = 0.0
    try:
        for end in (0, 1):
            mujoco.mj_resetData(m, d)
            d.qpos[:] = qpos0
            d.ctrl[:] = hold
            for a in act_ids:
                lo, hi = m.actuator_ctrlrange[a] if m.actuator_ctrllimited[a] else (-1.0, 1.0)
                d.ctrl[a] = lo if end == 0 else hi
            for _ in range(int(1.5 / m.opt.timestep)):
                mujoco.mj_step(m, d)
            gap = min(mujoco.mj_geomDistance(m, d, x, y, 0.5, fromto) for x in ga for y in gb)
            best = max(best, gap)
    finally:
        m.opt.gravity[:] = gravity
    return {"max_opening_m": round(float(best), 4),
            "method": "gripper actuators driven to each end of their range; min collision gap between finger sides"}


def _common_ancestor(m, bodies: list[int]) -> int:
    def path(b):
        out = [b]
        while b:
            b = m.body_parentid[b]
            out.append(b)
        return out
    common = set(path(bodies[0]))
    for b in bodies[1:]:
        common &= set(path(b))
    return max(common, key=lambda b: len(path(b)))


def _hold_ctrl(m, qpos0: np.ndarray) -> np.ndarray:
    """Controls that keep position-servo joints at the home pose (zero for torque motors)."""
    u = np.zeros(m.nu)
    for a in range(m.nu):
        if int(m.actuator_trntype[a]) in TRN_JOINT and int(m.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_AFFINE):
            j = int(m.actuator_trnid[a, 0])
            u[a] = qpos0[m.jnt_qposadr[j]]
    if m.nkey:
        for k in range(m.nkey):
            if np.allclose(m.key_qpos[k], qpos0):
                u = m.key_ctrl[k].copy()
    return u


def _reach(m, arm: Chain, qpos0: np.ndarray, samples: int = 1500, seed: int = 0) -> dict:
    """Sample joint configurations within limits and record end-effector positions."""
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    j_ids = [m.joint(n).id for n in arm.joints]
    leaf = m.body(arm.leaf).id
    base_b = m.body(arm.bodies[0]).id
    pts = []
    for _ in range(samples):
        d.qpos[:] = qpos0
        for j in j_ids:
            lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-math.pi, math.pi)
            d.qpos[m.jnt_qposadr[j]] = rng.uniform(lo, hi)
        mujoco.mj_kinematics(m, d)
        pts.append(d.xpos[leaf].copy())
    pts = np.array(pts)
    mujoco.mj_kinematics(m, d)
    base = d.xpos[m.body_parentid[base_b]] if m.body_parentid[base_b] else d.xpos[base_b]
    horiz = np.linalg.norm(pts[:, :2] - base[:2], axis=1)
    return {"base_xy": [round(float(v), 3) for v in base[:2]], "max_horizontal": round(float(horiz.max()), 3),
            "min_horizontal": round(float(np.percentile(horiz, 1)), 3),
            "z_range": [round(float(pts[:, 2].min()), 3), round(float(pts[:, 2].max()), 3)]}


def _end_effectors(m, chains: list[Chain], act_j) -> list[dict]:
    """For every arm, look at what hangs off its tip: 2 short branches = parallel gripper,
    3+ = multi-finger hand, a jaw-like last link = single-jaw gripper, nothing = bare flange."""
    by_start_parent: dict[int, list[Chain]] = {}
    for c in chains:
        by_start_parent.setdefault(int(m.body_parentid[m.body(c.bodies[0]).id]), []).append(c)
    out = []
    for arm in [c for c in chains if c.role == "arm"]:
        tip = m.body(arm.leaf).id
        branches, frontier = [], [tip]
        while frontier:  # branches under the tip, and branches under those (linkage grippers, hands)
            nxt = []
            for b in frontier:
                for c in by_start_parent.get(b, []):
                    if c.role in ("arm", "leg") or c.length > 0.25:
                        continue
                    branches.append(c)
                    nxt.append(m.body(c.leaf).id)
            frontier = nxt
        direct = [c for c in by_start_parent.get(tip, []) if c in branches]
        acts = sorted({a for c in branches for jn in c.actuated_joints for a in act_j.get(m.joint(jn).id, [])})
        if len(direct) >= 3 and len(acts) >= 3:  # a motor per finger -> dexterous hand
            for c in branches:
                c.role = "finger"
            out.append({"arm": arm.leaf, "type": "multi_finger_hand", "fingers": [c.leaf for c in direct],
                        "n_fingers": len(direct), "actuators": [_name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in acts]})
        elif len(direct) >= 2 and acts:
            for c in branches:
                c.role = "gripper"
            out.append({"arm": arm.leaf, "type": "parallel_gripper", "fingers": [c.leaf for c in direct],
                        "_bodies": [m.body(c.leaf).id for c in branches], "_acts": acts,
                        "_direct": [m.body(c.bodies[0]).id for c in direct],
                        "actuators": [_name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in acts]})
        elif GRIPPER_WORDS.search(arm.bodies[-1]) and arm.actuated_joints:
            jaw = m.body(arm.bodies[-1]).id
            last_j = m.joint(arm.joints[-1]).id
            acts = act_j.get(last_j, [])
            out.append({"arm": arm.leaf, "type": "single_jaw_gripper", "fingers": [arm.bodies[-1], arm.bodies[-2]],
                        "_sides": ([jaw], [int(m.body_parentid[jaw])]), "_acts": acts,
                        "actuators": [_name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in acts]})
        else:
            out.append({"arm": arm.leaf, "type": "flange (no gripper)"})
    return out


def _wheels(m, d, root: int, floor_z: float) -> list[str]:
    """Continuous (unlimited) hinge joints on bodies whose geometry touches the floor."""
    out = []
    for j in range(m.njnt):
        if int(m.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE) or m.jnt_limited[j]:
            continue
        b = int(m.jnt_bodyid[j])
        p = b
        while p and p != root:
            p = int(m.body_parentid[p])
        if p != root:
            continue
        if _lowest_geom_z(m, d, b) - floor_z < 0.02 and not GRIPPER_WORDS.search(m.body(b).name):
            out.append(_name(m, mujoco.mjtObj.mjOBJ_JOINT, j))
    return out


def profile_robot(path: str | Path, name: str | None = None) -> RobotProfile:
    path = Path(path)
    m = load_model(path)
    d = mujoco.MjData(m)
    q0 = home_qpos(m)
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)

    roots = _root_bodies(m)
    geoms_all = {r: _robot_geoms(m, r) for r in roots}
    root = max(roots, key=lambda r: (len(geoms_all[r]), m.body_subtreemass[r])) if roots else 0
    base = _base_type(m, root)
    act_j = _actuated_joints(m)
    geoms = geoms_all.get(root, [])
    lo, hi = _bbox(m, d, geoms) if geoms else (np.zeros(3), np.zeros(3))
    floor_z = float(lo[2])
    chains = _chains(m, d, root, act_j)
    _classify_chains(m, d, chains, base, floor_z)

    effectors = _end_effectors(m, chains, act_j)
    wheels = _wheels(m, d, root, floor_z) if base == "floating" else []
    if len(wheels) >= 2:
        base = "wheeled"
        for c in chains:
            if c.role == "leg" and any(j in wheels for j in c.joints):
                c.role = "wheel"
    legs = [c for c in chains if c.role == "leg"]
    arms = [c for c in chains if c.role == "arm"]

    # rotors: floating base, no legs, actuators acting on sites (thrust)
    site_acts = [a for a in range(m.nu) if int(m.actuator_trntype[a]) == TRN_SITE]
    aerial = base == "floating" and not legs and len(site_acts) >= 3

    notes = []
    gripper_info = {}
    graspers = [e for e in effectors if e["type"] in ("parallel_gripper", "single_jaw_gripper")]
    hands = [e for e in effectors if e["type"] == "multi_finger_hand"]
    if graspers:
        e = graspers[0]
        gripper_info = {k: v for k, v in e.items() if not k.startswith("_")}
        if e["type"] == "single_jaw_gripper":
            gripper_info |= _gripper_opening(m, [], e["_acts"], q0, sides=e["_sides"])
        else:
            gripper_info |= _gripper_opening(m, e["_bodies"], e["_acts"], q0, direct=e.get("_direct"))
    elif hands:
        gripper_info = {k: v for k, v in hands[0].items() if not k.startswith("_")}
        notes.append(f"multi-finger hand ({hands[0]['n_fingers']} fingers): no generic grasp skill for dexterous hands yet")
    if wheels:
        notes.append(f"wheeled base ({len(wheels)} wheel joints: {', '.join(wheels)})")

    if aerial:
        kind = "aerial"
    elif base == "floating" and len(legs) == 4:
        kind = "quadruped" if not arms else "legged_manipulator"
    elif base == "floating" and len(legs) == 2:
        kind = "humanoid" if arms else "biped"
    elif base == "floating" and legs:
        kind = "legged"
    elif base in ("planar_mobile", "wheeled"):
        kind = "mobile_manipulator" if arms else "wheeled_base"
    elif arms:
        kind = "fixed_arm"
    else:
        kind = "unknown"

    reach = {}
    for i, a in enumerate(arms):
        reach[f"arm{i}:{a.leaf}"] = _reach(m, a, q0)

    actuators = []
    for a in range(m.nu):
        t, tid = int(m.actuator_trntype[a]), int(m.actuator_trnid[a, 0])
        target = (_name(m, mujoco.mjtObj.mjOBJ_JOINT, tid) if t in TRN_JOINT else
                  _name(m, mujoco.mjtObj.mjOBJ_TENDON, tid) if t == TRN_TENDON else
                  _name(m, mujoco.mjtObj.mjOBJ_SITE, tid) if t == TRN_SITE else "?")
        servo = int(m.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_AFFINE) and m.actuator_biasprm[a, 1] < 0
        actuators.append({"name": _name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a), "target": target,
                          "mode": "position" if servo else "torque/force",
                          "ctrlrange": [round(float(v), 3) for v in m.actuator_ctrlrange[a]] if m.actuator_ctrllimited[a] else None,
                          "forcerange": [round(float(v), 3) for v in m.actuator_forcerange[a]] if m.actuator_forcelimited[a] else None})
    if any(x["mode"] == "torque/force" for x in actuators) and kind != "aerial":
        notes.append("some actuators are torque/force controlled: position targets need a PD layer")

    size = hi - lo
    return RobotProfile(
        name=name or path.stem, source=str(path), base=base, kind=kind, n_legs=len(legs), n_arms=len(arms),
        has_gripper=bool(graspers), gripper=gripper_info, mass_kg=round(float(m.body_subtreemass[root]), 3),
        size_m={"height": round(float(size[2]), 3), "length": round(float(size[0]), 3), "width": round(float(size[1]), 3)},
        reach_m=reach, dof=int(m.nv - (6 if base == "floating" else 0)), actuators=actuators,
        chains=[{"role": c.role, "leaf": c.leaf, "n_joints": len(c.joints), "n_actuated": len(c.actuated_joints),
                 "length_m": round(c.length, 3), "leaf_height_m": round(c.leaf_height - floor_z, 3)} for c in chains
                if c.joints or c.role != "other"],
        sensors=[_name(m, mujoco.mjtObj.mjOBJ_SENSOR, s) for s in range(m.nsensor)], notes=notes)


# ----------------------------------------------------------------------------------------- capabilities
def capabilities_from_profile(p: RobotProfile, object_size_m=(0.05, 0.05, 0.06)) -> Capabilities:
    """What this body could plausibly do with generic skills, before anything is validated."""
    prehensile = p.has_gripper and p.n_arms > 0
    opening = p.gripper.get("max_opening_m", 0.0)
    push = p.kind in ("quadruped", "legged", "legged_manipulator", "biped", "humanoid", "wheeled_base",
                      "mobile_manipulator", "fixed_arm")
    if p.base == "fixed" and p.reach_m:
        r = max(p.reach_m.values(), key=lambda v: v["max_horizontal"])
        ws = {"type": "annulus", "center": r["base_xy"], "r_min": r["min_horizontal"], "r_max": r["max_horizontal"],
              "method": "sampled joint space (introspection)"}
    else:
        ws = {"type": "floor_reachable_by_moving", "object_height_range_m": [0.02, max(0.12, p.size_m["height"] * 0.5)]}
    modes = (["grasp"] if prehensile else []) + (["push"] if push else [])
    notes = list(p.notes)
    if prehensile and opening and opening < min(object_size_m[:2]):
        notes.append(f"gripper opening {opening * 100:.1f} cm is smaller than the object ({min(object_size_m[:2]) * 100:.0f} cm)")
        prehensile = False
        modes = [x for x in modes if x != "grasp"]
    if p.kind == "aerial":
        notes.append("no contact skill for aerial robots yet")
    return Capabilities(
        robot=p.name, embodiment={"fixed_arm": "fixed_arm", "mobile_manipulator": "mobile_pusher",
                                  "wheeled_base": "mobile_pusher"}.get(p.kind, "legged" if p.n_legs else "fixed_arm"),
        model_source=f"introspected from {Path(p.source).name}",
        base_mobility={"fixed": "fixed", "planar_mobile": "holonomic_planar", "wheeled": "holonomic_planar"}.get(p.base, "legged_planar"),
        end_effectors=([f"gripper ({', '.join(p.gripper.get('fingers', []))})"] if p.has_gripper else []) +
                      [f"arm tip ({k.split(':')[1]})" for k in p.reach_m] + (["feet"] if p.n_legs else []),
        contact_modes=modes, prehensile=prehensile, object_workspace=ws, max_payload_kg=0.0,
        max_contact_force_n=float(max((abs(a["forcerange"][1]) for a in p.actuators if a["forcerange"]), default=0.0)),
        can_set_planar_yaw=bool(modes), can_flip_object=False,
        action_primitives=[], modeled_limits={"dof": p.dof, "mass_kg": p.mass_kg, "size_m": p.size_m},
        validation={"status": "not yet validated: capabilities inferred from the model only"}, notes=notes)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Describe a robot from its MJCF/URDF model")
    ap.add_argument("model")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    p = profile_robot(a.model)
    if a.json:
        print(json.dumps(p.as_dict(), indent=2))
    else:
        print(p.summary())
        for c in p.chains:
            print(f"  {c['role']:8s} -> {c['leaf']:28s} joints {c['n_joints']:2d} (actuated {c['n_actuated']}), "
                  f"length {c['length_m']:.2f} m, tip {c['leaf_height_m']:.2f} m above ground")
        for n in p.notes:
            print("  note:", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
