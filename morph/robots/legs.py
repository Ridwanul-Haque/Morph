"""Leg geometry for the stance / foot-push controller (go1_control.Go1Controller).

The controller needs, per leg: where the foot rests under the trunk, which actuators move it, and an
IK from a foot position in the trunk frame to joint targets. For the Go1 these were written by hand
(closed-form IK). `measure_legs` gets the same things from any legged model: leg names from the
foot bodies, nominal feet and standing height from the keyframe, and a damped least-squares IK on
the leg's own joints using MuJoCo's Jacobians (trunk held fixed in a scratch copy of the model).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import mujoco
import numpy as np


@dataclass
class LegGeometry:
    legs: tuple[str, ...]  # leg labels; the controller's crawl / push logic uses the FR/FL/RR/RL convention
    nominal: dict  # leg -> foot xy under the trunk when standing (trunk frame)
    side: dict  # leg -> +1 left, -1 right
    ctrl_idx: dict  # leg -> actuator indices of that leg
    foot_geom: dict  # leg -> geom name
    foot_site: dict  # leg -> site name
    stand_height: float
    foot_z: float  # foot site height above the floor when resting
    ik: Callable[[str, np.ndarray], np.ndarray | None]
    notes: list = field(default_factory=list)


def go1_geometry() -> LegGeometry:
    """The hand-written Go1 geometry (unchanged; kept so the Go1 results reproduce exactly)."""
    from .quadruped import HIP, L_ABD, LEGS, SIDE, STAND_HEIGHT, leg_ik
    return LegGeometry(legs=LEGS, nominal={l: np.array([HIP[l][0], HIP[l][1] + L_ABD * SIDE[l]]) for l in LEGS},
                       side=dict(SIDE), ctrl_idx={l: np.arange(3 * i, 3 * i + 3) for i, l in enumerate(LEGS)},
                       foot_geom={l: l for l in LEGS}, foot_site={l: l for l in LEGS}, stand_height=STAND_HEIGHT,
                       foot_z=0.006, ik=leg_ik, notes=["Go1: hand-written closed-form leg IK"])


def _label(x: float, y: float) -> str:
    return ("F" if x > 0 else "R") + ("L" if y > 0 else "R")


def measure_legs(m: mujoco.MjModel, torso: str, feet: list[dict], joints: list[str], qpos_home: np.ndarray,
                 foot_z: float | None = None) -> LegGeometry:
    """Leg geometry of any 4-legged model from its standing keyframe. `feet` / `joints` as in LeggedSetup;
    foot sites must already exist in `m` as morph_foot{i}."""
    d = mujoco.MjData(m)
    d.qpos[:] = qpos_home
    mujoco.mj_forward(m, d)
    tb = m.body(torso).id
    R, p0 = d.xmat[tb].reshape(3, 3), d.xpos[tb].copy()

    def in_trunk(x):
        return R.T @ (x - p0)

    legs, nominal, side, ctrl_idx, fgeom, fsite, leg_dofs, leg_q = [], {}, {}, {}, {}, {}, {}, {}
    act_joint = [m.actuator_trnid[a, 0] for a in range(m.nu)]
    for i, f in enumerate(feet):
        site = m.site(f"morph_foot{i}").id
        pf = in_trunk(d.site_xpos[site])
        lab = _label(pf[0], pf[1])
        if lab in legs:
            raise ValueError(f"two feet map to {lab}; can't label legs by position")
        # this leg's joints: actuated joints on the path from the foot body up to the trunk
        chain, b = [], m.body(f["body"]).id
        while b != tb and b != 0:
            chain += [j for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b])]
            b = m.body_parentid[b]
        js = [j for j in reversed(chain) if j in act_joint]
        legs.append(lab)
        nominal[lab] = pf[:2]
        side[lab] = 1 if pf[1] > 0 else -1
        ctrl_idx[lab] = np.array([act_joint.index(j) for j in js])
        fgeom[lab] = m.geom(f["geom_id"]).name if "geom_id" in f else f["geom"]
        fsite[lab] = f"morph_foot{i}"
        leg_dofs[lab] = np.array([m.jnt_dofadr[j] for j in js])
        leg_q[lab] = np.array([m.jnt_qposadr[j] for j in js])
    order = tuple(l for l in ("FR", "FL", "RR", "RL") if l in legs)
    if len(order) != 4:
        raise ValueError(f"expected four legs FR/FL/RR/RL, found {legs}")

    # scratch model for IK: trunk frozen at the origin with identity orientation
    sd = mujoco.MjData(m)
    sd.qpos[:] = qpos_home
    fq = m.jnt_qposadr[m.body_jntadr[tb]]
    sd.qpos[fq:fq + 7] = [0, 0, 0, 1, 0, 0, 0]
    jp = np.zeros((3, m.nv))
    lo_hi = {l: m.jnt_range[[m.dof_jntid[v] for v in leg_dofs[l]]] for l in order}
    site_id = {l: m.site(fsite[l]).id for l in order}
    home = {l: qpos_home[leg_q[l]].copy() for l in order}
    last = {l: home[l].copy() for l in order}

    def solve(leg: str, p: np.ndarray, q: np.ndarray) -> np.ndarray | None:
        qa, dofs, s = leg_q[leg], leg_dofs[leg], site_id[leg]
        lo, hi = lo_hi[leg][:, 0], lo_hi[leg][:, 1]
        for _ in range(30):
            sd.qpos[qa] = q
            mujoco.mj_kinematics(m, sd)
            e = p - sd.site_xpos[s]
            if np.linalg.norm(e) < 5e-4:
                return q
            mujoco.mj_comPos(m, sd)
            mujoco.mj_jacSite(m, sd, jp, None, s)
            J = jp[:, dofs]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), e)
            q = np.clip(q + np.clip(dq, -0.4, 0.4), lo, hi)
        sd.qpos[qa] = q
        mujoco.mj_kinematics(m, sd)
        return q if np.linalg.norm(p - sd.site_xpos[s]) < 2e-3 else None

    def ik(leg: str, p: np.ndarray) -> np.ndarray | None:
        # start from the last solution (smooth, same knee branch); if that stalls on a joint limit,
        # try again from the standing posture before calling the point unreachable
        q = solve(leg, p, last[leg].copy())
        if q is None:
            q = solve(leg, p, home[leg].copy())
        if q is not None:
            last[leg] = q
        return q

    stand = float(qpos_home[fq + 2])
    if foot_z is None:
        foot_z = float(np.mean([d.site_xpos[m.site(fsite[l]).id][2] for l in order]))
    return LegGeometry(legs=order, nominal=nominal, side=side, ctrl_idx=ctrl_idx, foot_geom=fgeom, foot_site=fsite,
                       stand_height=stand, foot_z=foot_z, ik=ik,
                       notes=[f"legs labelled by foot position: {', '.join(order)}",
                              "numeric leg IK (damped least squares on each leg's own joints)"])
