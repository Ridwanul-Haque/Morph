"""Build a working arm adapter for a robot Morph has never seen, from its model file.

Everything the hand-written arm configs specify is derived here instead:
which joints form the arm and which motors drive them, which actuator is the
gripper and which end of its range means "open", where the grasp point is and
how the fingers close (the tool frame), a hover pose to start from, where to
mount the arm so it can reach the workspace, and how fast its servos track.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from .. import introspect as I
from ..scene import BlockSpec
from ..sim import PhysicalParams
from .menagerie_arm import MenagerieArmAdapter, MenagerieArmConfig

WORKSPACE_CENTER = (0.46, 0.0)  # centre of the block/goal area used by Morph's scenes


class NotAGraspingArm(RuntimeError):
    pass


def _tool_frame(m, d, tip: int, left: list[int], right: list[int], act_ids: list[int], q0: np.ndarray):
    """Drive the gripper to both ends of its range, find which end is open, and measure the grasp
    point and closing axis between the fingertips at the open end (expressed in the tip body)."""
    def side_geoms(bodies):
        return [g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies and (m.geom_contype[g] or m.geom_conaffinity[g])]
    ga, gb = side_geoms(left), side_geoms(right)
    if not ga or not gb:
        raise NotAGraspingArm("fingertips have no collision geometry")
    fromto = np.zeros(6)

    def measure(dd):
        best = None
        for x in ga:
            for y in gb:
                dist = mujoco.mj_geomDistance(m, dd, x, y, 0.5, fromto)
                if best is None or dist < best[0]:
                    best = (dist, fromto.copy())
        R, p = dd.xmat[tip].reshape(3, 3).copy(), dd.xpos[tip].copy()
        return best[0], best[1], R, p, dd.qpos.copy(), dd.ctrl.copy()
    ends = I._settle_at_ends(m, d, act_ids, q0, measure)
    open_i = int(np.argmax([e[0] for e in ends]))
    gap, ft, R, p, q_open, _ = ends[open_i]
    dd_open = mujoco.MjData(m)
    dd_open.qpos[:] = q_open
    mujoco.mj_kinematics(m, dd_open)
    q_closed = ends[1 - open_i][4]
    # grasp point = midpoint between the two finger pads' geometric centres (keeps it centred even
    # when the pads' nearest points are off-centre); closing axis = from one pad centre to the other
    pa = np.vstack([I._geom_points(m, dd_open, g) for g in ga])
    pb = np.vstack([I._geom_points(m, dd_open, g) for g in gb])
    close_w = (pb.mean(0) - pa.mean(0))
    close_w /= np.linalg.norm(close_w) + 1e-9

    def inner_surface_centre(pts, towards):
        """Centre of the points facing the other finger (within 5 mm of the innermost plane)."""
        proj = pts @ towards
        return pts[proj > proj.max() - 0.005].mean(0)
    centre_w = (inner_surface_centre(pa, close_w) + inner_surface_centre(pb, -close_w)) / 2
    centre = R.T @ (centre_w - p)  # grasp point in the tip frame
    close = R.T @ close_w
    approach = centre / (np.linalg.norm(centre) + 1e-9)  # from the wrist towards the fingers
    approach = approach - close * float(approach @ close)
    approach /= np.linalg.norm(approach) + 1e-9
    x = np.cross(close, approach)
    Rt = np.column_stack([x, close, approach])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, Rt.flatten())
    lo_hi = [m.actuator_ctrlrange[act_ids[0]][k] if m.actuator_ctrllimited[act_ids[0]] else (-1.0, 1.0)[k] for k in (0, 1)]
    return {"tcp_pos": tuple(float(v) for v in centre), "tcp_quat": tuple(float(v) for v in quat),
            "ctrl_open": float(lo_hi[open_i]), "ctrl_close": float(lo_hi[1 - open_i]),
            "gap_open_m": float(gap), "q_open": q_open, "q_closed": q_closed}


def auto_arm_config(model_path: str | Path, name: str | None = None) -> tuple[MenagerieArmConfig, dict]:
    """Derive a MenagerieArmConfig for an arm with a gripper. Returns (config, derivation report)."""
    path = Path(model_path).resolve()
    m = I.load_model(path)
    d = mujoco.MjData(m)
    q0 = I.home_qpos(m)
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)
    roots = I._root_bodies(m)
    geoms_all = {r: I._robot_geoms(m, r) for r in roots}
    root = max(roots, key=lambda r: (len(geoms_all[r]), m.body_subtreemass[r]))
    base = I._base_type(m, root)
    act_j = I._actuated_joints(m)
    lo, _ = I._bbox(m, d, geoms_all[root])
    chains = I._chains(m, d, root, act_j)
    I._classify_chains(m, d, chains, base, float(lo[2]))
    effectors = I._end_effectors(m, chains, act_j)
    graspers = [e for e in effectors if e["type"] in ("parallel_gripper", "single_jaw_gripper")]
    if base != "fixed" or not graspers:
        raise NotAGraspingArm(f"not a fixed arm with a gripper (base={base}, effectors={[e['type'] for e in effectors]})")
    eff = graspers[0]
    arm = next(c for c in chains if c.role == "arm" and c.leaf == eff["arm"])
    grip_acts = list(eff["_acts"])
    grip_joints = {j for j, acts in act_j.items() if set(acts) & set(grip_acts)}

    # arm joints + the position servo driving each one
    arm_joints, arm_acts = [], []
    for jn in arm.joints:
        j = m.joint(jn).id
        if j in grip_joints:
            continue
        acts = [a for a in act_j.get(j, []) if int(m.actuator_trntype[a]) in I.TRN_JOINT]
        if not acts:
            continue
        a = acts[0]
        if not (int(m.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_AFFINE) and m.actuator_biasprm[a, 1] < 0):
            raise NotAGraspingArm(f"joint {jn} is torque-driven; needs a PD layer (not implemented)")
        arm_joints.append(jn)
        arm_acts.append(I._name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a))
    if len(arm_joints) < 5:
        raise NotAGraspingArm(f"only {len(arm_joints)} position-controlled arm joints; top-down grasping needs 5+")

    tip = m.body(arm.leaf).id
    if eff["type"] == "single_jaw_gripper":
        left, right = eff["_sides"]
        tip = int(m.body_parentid[left[0]])
    else:
        left, right = I.finger_sides(m, d, eff["_bodies"], eff.get("_direct"), q0)
        tip = I._common_ancestor(m, left + right)
    tf = _tool_frame(m, d, tip, left, right, grip_acts, q0)

    ap_joint = sorted(grip_joints)[0]
    ap_adr = m.jnt_qposadr[ap_joint]
    prof = I.profile_robot(path, name=name)
    reach = max(v["max_horizontal"] for v in prof.reach_m.values())
    base_x = WORKSPACE_CENTER[0] - float(np.clip(0.55 * reach, 0.2, 0.55))
    home_q = tuple(float(q0[m.joint(j).qposadr[0]]) for j in arm_joints)

    cfg = MenagerieArmConfig(
        name=name or path.stem, xml=str(path), root_body=I._name(m, mujoco.mjtObj.mjOBJ_BODY, root),
        license="see the model folder", arm_joints=tuple(arm_joints), arm_actuators=tuple(arm_acts),
        gripper_actuator=I._name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, grip_acts[0]),
        ctrl_open=tf["ctrl_open"], ctrl_close=tf["ctrl_close"],
        tcp_body=I._name(m, mujoco.mjtObj.mjOBJ_BODY, tip), tcp_pos=tf["tcp_pos"], tcp_quat=tf["tcp_quat"],
        left_bodies=tuple(I._name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in left),
        right_bodies=tuple(I._name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in right),
        aperture_joint=I._name(m, mujoco.mjtObj.mjOBJ_JOINT, ap_joint),
        aperture_open=float(tf["q_open"][ap_adr]), aperture_closed=float(tf["q_closed"][ap_adr]),
        home_q=home_q, base_pos=(base_x, 0.0, 0.0),
        notes=["config derived automatically from the model (morph.robots.auto_arm)"])
    report = {"arm_joints": arm_joints, "gripper": eff["type"], "gripper_actuator": cfg.gripper_actuator,
              "open_end": "high" if tf["ctrl_open"] > tf["ctrl_close"] else "low",
              "opening_m": round(tf["gap_open_m"], 4), "tcp_body": cfg.tcp_body,
              "tcp_pos": [round(v, 4) for v in tf["tcp_pos"]], "base_x": round(base_x, 3), "reach_m": reach}
    return cfg, report


def _calibrate(ad: MenagerieArmAdapter, center=WORKSPACE_CENTER) -> tuple[float, float]:
    """Measure how much of a 1 cm / 0.03 rad lead the servos track per control tick."""
    tgt = np.array([center[0], center[1], getattr(ad, "travel_z", 0.15) - 0.02])
    q = ad.ik(tgt, 0.0, q0=ad.home_q)
    if q is None:
        q = np.asarray(ad.home_q)
    ad.env.data.qpos[ad._qadr] = q
    mujoco.mj_forward(ad.model, ad.env.data)
    for _ in range(40):
        ad.env.advance(ad.ctrl(q, False))
    moved, turned = [], []
    for _ in range(10):
        tcp = ad.tcp()
        qq = ad.ik(tcp + np.array([0.0, 0.01, 0.0]), ad.hand_yaw())
        if qq is None:
            break
        ad.env.advance(ad.ctrl(qq, False))
        moved.append(ad.tcp()[1] - tcp[1])
    for _ in range(10):
        hy = ad.hand_yaw()
        qq = ad.ik(ad.tcp(), hy + 0.03)
        if qq is None:
            break
        ad.env.advance(ad.ctrl(qq, False))
        turned.append(ad.hand_yaw() - hy)
    rp = float(np.mean(moved[4:])) / 0.01 if len(moved) > 4 else 0.2
    ry = float(np.mean(turned[4:])) / 0.03 if len(turned) > 4 else 0.15
    return (float(np.clip(0.95 / max(rp, 1e-3), 1.0, 8.0)), float(np.clip(0.95 / max(ry, 1e-3), 1.0, 8.0)))


CARRY_HEIGHTS = (0.15, 0.13, 0.11, 0.095, 0.08)  # 0.08 still carries the 6 cm block clear of the floor


def _find_hover(ad: MenagerieArmAdapter, tries: int = 60, seed: int = 0, center=WORKSPACE_CENTER,
                heights=CARRY_HEIGHTS):
    """Highest carry height (<= 0.15 m) at which the gripper can point straight down over the
    workspace, and a hover pose there. Restarts IK from random joint configurations because
    solving from the model's zero pose often stalls on joint limits."""
    rng = np.random.default_rng(seed)
    seeds = [np.asarray(ad.home_q)] + [rng.uniform(ad.q_lo, ad.q_hi) for _ in range(tries)]
    for z in heights:
        for q0 in seeds:
            for yaw in (0.0, math.pi / 2):
                q = ad.ik(np.array([center[0], center[1], z]), yaw, q0=q0, iters=200)
                if q is not None:
                    return q, z
    return None, None


def _calibrate_grip(ad: MenagerieArmAdapter, cfg: MenagerieArmConfig, block, scales=(1, 2, 4, 8, 16),
                    center=WORKSPACE_CENTER, pick=None, place=None):
    """Try a real pick-and-place; if the block slips out, stiffen the gripper servo (force limit
    unchanged) and try again. Returns the adapter with the smallest gain that worked."""
    from ..arm_skills import expert_action, run_episode
    from ..task import default_spec
    trials = []
    for k in scales:
        if k != cfg.gripper_gain_scale:
            cfg.gripper_gain_scale = float(k)
            home, lead, tz = ad.home_q, ad.lead_scale, getattr(ad, "travel_z", 0.15)
            ad = MenagerieArmAdapter(cfg, block)
            ad.home_q, ad.lead_scale, ad.travel_z = home, lead, tz
        if pick is not None:  # in the user's scene: from where the object starts to a clear spot nearby
            ad.reset(0, pick, PhysicalParams(0.3, 0.6))
            r = run_episode(ad, default_spec(place[0], place[1], pick[2] + 0.5), expert_action, max_time=12.0)
        else:
            ad.reset(0, (center[0], center[1] - 0.12, 0.3), PhysicalParams(0.3, 0.6))
            r = run_episode(ad, default_spec(center[0], center[1] + 0.12, 1.2), expert_action, max_time=12.0)
        trials.append({"gain_scale": k, "success": r.outcome.success, "reasons": r.outcome.reasons[:2]})
        if r.outcome.success:
            break
    # one step of margin: the trial grasp is a single clean pick, real ones land off-centre
    nxt = [k for k in scales if k > cfg.gripper_gain_scale]
    if trials[-1]["success"] and nxt:
        cfg.gripper_gain_scale = float(nxt[0])
        home, lead, tz = ad.home_q, ad.lead_scale, getattr(ad, "travel_z", 0.15)
        ad = MenagerieArmAdapter(cfg, block)
        ad.home_q, ad.lead_scale, ad.travel_z = home, lead, tz
    return ad, trials


TASK_POINTS = [(x, y) for x in (0.36, 0.46, 0.56) for y in (-0.30, -0.15, 0.0, 0.15, 0.30)]


def _coverage(ad: MenagerieArmAdapter, travel_z: float, points=None) -> float:
    """Fraction of Morph's task area (block and goal regions) the gripper can reach pointing down,
    at both grasp height and carry height. The block is square, so any hand heading is within 45
    degrees of a usable one; checking straight-on and the two 45-degree headings covers the worst case."""
    from ..arm_skills import GRASP_Z
    bx, by = ad.cfg.base_pos[0], ad.cfg.base_pos[1]
    rng = np.random.default_rng(1)
    seeds = [ad.home_q] + [rng.uniform(ad.q_lo, ad.q_hi) for _ in range(2)]

    def reachable(p, yaw):
        return any(ad.ik(p, yaw, q0=q0, iters=120) is not None for q0 in seeds)

    ok = 0.0
    points = points or TASK_POINTS
    for x, y in points:
        radial = math.atan2(y - by, x - bx)
        good = 0
        for dyaw in (0.0, math.pi / 4, -math.pi / 4):
            good += all(reachable(np.array([x, y, z]), radial + dyaw) for z in (GRASP_Z, travel_z))
        ok += good / 3
    return ok / len(points)


def _clear_spot(ad, start, keepout, clearance):
    """A floor spot 12-24 cm from `start` the arm can reach, away from every fixture (for the grip test)."""
    for r in (0.15, 0.2, 0.12, 0.24):
        for k in range(12):
            a = k * math.pi / 6
            x, y = start[0] + r * math.cos(a), start[1] + r * math.sin(a)
            if all(not (lo[0] - clearance < x < hi[0] + clearance and lo[1] - clearance < y < hi[1] + clearance)
                   for lo, hi in keepout) and ad.reach_ok(x, y):
                return x, y
    return None


def build_auto_arm(model_path: str | Path, name: str | None = None, block: BlockSpec | None = None, log=None,
                   task_xy: list | None = None, min_carry_z: float | None = None, start=None, keepout=None,
                   mount: tuple | None = None):
    """Config -> adapter. Then, by experiment: choose where to mount the arm (most reachable task
    area), a carry height and hover pose, the servo speed calibration and the gripper strength.

    task_xy: where the work happens (object start, goal) in a user's scene; default Morph's task area.
    min_carry_z: lowest usable carry height (TCP), e.g. to lift the object over a bin's walls.
    mount: (x, y, heading) where the user put the arm: used as is instead of searching for a mount."""
    from dataclasses import replace
    block = block or BlockSpec()
    cfg, report = auto_arm_config(model_path, name)
    reach = report["reach_m"]
    if task_xy:
        pts = [tuple(map(float, p[:2])) for p in task_xy]
        center = tuple(np.mean(pts, axis=0))
        points = pts + [tuple((np.array(a) + np.array(b)) / 2) for a, b in zip(pts, pts[1:])]
    else:
        center, points = WORKSPACE_CENTER, None
    heights = CARRY_HEIGHTS
    if min_carry_z:
        # lowest height that clears what the object passes over: less lift, less swing, shorter drop
        heights = tuple(sorted(h for h in (0.21, 0.19, 0.17) + CARRY_HEIGHTS + (round(min_carry_z + 0.005, 3),)
                               if h >= min_carry_z))
        if not heights:
            raise NotAGraspingArm(f"the object has to be carried above {min_carry_z * 100:.0f} cm")
    best = None
    # mounting distance behind the task area, as a fraction of reach; or exactly where the user put it
    tries = [(0.55, (mount[0], mount[1], 0.0), mount[2])] if mount else \
        [(k, (center[0] - k * reach, center[1], 0.0), 0.0) for k in (0.35, 0.45, 0.55, 0.65)]
    for k, base, yaw in tries:
        cfg.base_pos, cfg.base_yaw = base, yaw
        ad = MenagerieArmAdapter(cfg, block)
        ad.reset(0, (0.9, 0.9, 0.0), PhysicalParams())
        hover, travel_z = _find_hover(ad, center=center, heights=heights)
        if hover is None:
            continue
        ad.home_q, ad.travel_z = hover, travel_z
        cov = _coverage(ad, travel_z, points)
        # The highest carry height can sit at the edge of the wrist's range; a lower one sometimes
        # frees up enough of it to rotate the block on the way. Keep going down while that helps.
        for z in ([] if min_carry_z else heights[heights.index(travel_z) + 1:]):
            q = ad.ik(np.array([center[0], center[1], z]), 0.0, q0=hover, iters=200)
            if q is None:
                break
            ad.home_q, ad.travel_z = q, z
            c = _coverage(ad, z, points)
            if c <= cov + 0.02:
                ad.home_q, ad.travel_z = hover, travel_z
                break
            hover, travel_z, cov = q, z, c
        if log:
            log(f"  mount {cfg.base_pos[0]:+.2f} m: carry height {travel_z:.2f} m, reachable task area {cov:.0%}")
        # most reachable area, then highest carry height, then a mid-range mount (awkward wrist angles
        # close to the base made the carry rotation hit joint limits)
        key = (round(cov, 2), travel_z, -abs(k - 0.55))
        if best is None or key > best[0]:
            best = (key, cov, travel_z, cfg.base_pos, hover, cfg.base_yaw)
    if best is None:
        raise NotAGraspingArm(("from where it's placed, the gripper can't point down over the task area" if mount else
                               "no mounting position lets the gripper point down over the task area")
                              + (f" at {min_carry_z * 100:.0f} cm or higher" if min_carry_z else ""))
    _, cov, travel_z, cfg.base_pos, hover, cfg.base_yaw = best
    cfg.home_q = tuple(float(v) for v in hover)
    ad = MenagerieArmAdapter(cfg, block)
    ad.home_q, ad.travel_z = np.array(cfg.home_q), travel_z
    ad.reset(0, (0.9, 0.9, 0.0), PhysicalParams())
    report |= {"base_x": round(cfg.base_pos[0], 3), "travel_z": travel_z, "task_area_reachable": round(cov, 3)}
    cfg.lead_scale = _calibrate(ad, center)
    ad.lead_scale = cfg.lead_scale
    report["lead_scale"] = [round(v, 2) for v in cfg.lead_scale]
    # grip test: a practice pick. In a user's scene it happens right there (object where it starts, set down at a
    # clear spot); only if there's no clear spot in reach does it fall back to an empty floor.
    spot = _clear_spot(ad, start, keepout or [], math.hypot(block.half_x, block.half_y) + 0.03) if start else None
    if log:
        log("grip test: a practice pick to find how firmly to hold" + ("" if spot or not block.fixtures_xml
                                                                         else " (on an empty floor: no clear spot nearby)"))
    if spot or not block.fixtures_xml:
        ad, report["grip_trials"] = _calibrate_grip(ad, cfg, block, center=center, pick=start if spot else None, place=spot)
    else:
        bare = replace(block, fixtures_xml="")
        cal = MenagerieArmAdapter(cfg, bare)
        cal.home_q, cal.travel_z, cal.lead_scale = ad.home_q, ad.travel_z, ad.lead_scale
        cal, report["grip_trials"] = _calibrate_grip(cal, cfg, bare, center=center)
        ad = MenagerieArmAdapter(cfg, block)
        ad.home_q, ad.travel_z, ad.lead_scale = cal.home_q, cal.travel_z, cal.lead_scale
    ad.reset(0, (0.9, 0.9, 0.0), PhysicalParams())
    report["config"] = {k: v for k, v in asdict(cfg).items() if k not in ("notes",)}
    return ad, report
