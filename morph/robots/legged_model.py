"""Turn any legged MJCF into the model a walking policy is trained and run on.

Nothing here is specific to one robot: the torso, legs and feet come from introspection, the
standing pose from the model's keyframe, and the servo gains are scaled from the ones that work
for the Go1 by weight x leg length (the torque needed to hold the body up). The same builder
makes the GPU training model (feet-only contacts, 1-iteration solver) and the CPU model Morph
runs the policy in (all collisions, full solver), so observations match between the two.

Changes made to the robot, all recorded in `LeggedSetup.notes`:
  * torque motors -> position servos (the policy outputs joint targets), torque limits kept
  * gyro / velocimeter / accelerometer / up-vector at the robot's IMU site (added at the torso if it has none)
  * a site and a velocity sensor at each foot (for the foot rewards)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .. import introspect as I

# Reference: Playground's Go1 (12.7 kg, 0.426 m legs) walks with Kp 35, Kd 0.5, action scale 0.5.
REF_KP, REF_KD, REF_MASS_X_LEG = 35.0, 0.5, 12.7 * 0.426
SIM_DT, CTRL_DT = 0.004, 0.02


@dataclass
class LeggedSetup:
    model_path: str
    name: str
    torso: str
    joints: list[str]  # actuated leg joints, in actuator order
    feet: list[dict]  # per leg: body, geom, pos (in body frame), radius
    default_pose: list[float]
    stand_height: float
    kp: float
    kd: float
    action_scale: float = 0.5
    torque_limits: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    driven: list[int] = field(default_factory=list)  # which of `joints` the walking policy moves (the legs); [] = all

    @property
    def n(self) -> int:
        return len(self.joints)

    @property
    def acts(self) -> list[int]:
        return self.driven or list(range(self.n))


def _foot_geom(m: mujoco.MjModel, d: mujoco.MjData, body: int) -> tuple[int, float]:
    """The lowest collision geom of a leg's last body, and its radius (for spheres/capsules)."""
    best, best_z = -1, np.inf
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != body or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
            continue
        z = d.geom_xpos[g][2] - (m.geom_size[g][0] if m.geom_type[g] in (2, 3) else 0.0)
        if z < best_z:
            best, best_z = g, z
    if best < 0:
        raise ValueError(f"leg ending in {m.body(body).name} has no collision geom to stand on")
    r = float(m.geom_size[best][0]) if m.geom_type[best] in (2, 3) else 0.0
    return best, r


def legged_setup(model_path: str | Path, name: str | None = None) -> LeggedSetup:
    prof = I.profile_robot(model_path, name=name)
    if prof.base != "floating" or prof.n_legs < 2:
        raise ValueError(f"{prof.name} is not a legged robot ({prof.kind}, base {prof.base})")
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    q0 = I.home_qpos(m)
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)
    free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
    torso = m.jnt_bodyid[free]

    joints, taus = [], []
    for a in range(m.nu):
        if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError("only joint-driven actuators are supported for learned walking so far")
        j = m.actuator_trnid[a, 0]
        joints.append(m.joint(j).name)
        lim = m.actuator_forcerange[a] if m.actuator_forcelimited[a] else m.actuator_ctrlrange[a]
        taus.append(float(max(abs(lim[0]), abs(lim[1]))) if m.actuator_biastype[a] == 0 else
                    float(max(abs(m.actuator_forcerange[a][0]), abs(m.actuator_forcerange[a][1]))))

    feet = []
    for c in prof.chains:
        if c["role"] != "leg":
            continue
        b = m.body(c["leaf"]).id
        g, r = _foot_geom(m, d, b)
        feet.append({"body": c["leaf"], "geom": m.geom(g).name or f"geom{g}", "geom_id": int(g),
                     "pos": [float(v) for v in m.geom_pos[g]], "radius": r})

    # the walking policy moves the legs only; an arm or a head holds its resting pose (Spot's policy swung its arm from
    # the first step and flipped over within 0.6 s, every try)
    leg_joints = set()
    for f in feet:
        b = m.body(f["body"]).id
        while b != torso and b > 0:
            leg_joints |= {int(j) for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b])}
            b = int(m.body_parentid[b])
    driven = [i for i, jn in enumerate(joints) if m.joint(jn).id in leg_joints]
    legs = [c for c in prof.chains if c["role"] == "leg"]
    leg_len = float(np.mean([c["length_m"] for c in legs]))
    k = prof.mass_kg * leg_len / REF_MASS_X_LEG
    default = [float(q0[m.jnt_qposadr[m.joint(jn).id]]) for jn in joints]
    notes = [f"{len(feet)} feet from introspection: {', '.join(f['body'] for f in feet)}",
             f"servo gains scaled x{k:.2f} from the Go1's by weight x leg length "
             f"({prof.mass_kg:.1f} kg, {leg_len:.2f} m legs)",
             "standing pose from the model's keyframe" if m.nkey else "standing pose = model zero pose (no keyframe)"]
    if len(driven) < len(joints):
        notes.append(f"walking moves the {len(driven)} leg joints; the other {len(joints) - len(driven)} "
                     f"({', '.join(jn for i, jn in enumerate(joints) if i not in driven)}) hold their resting pose")
    return LeggedSetup(model_path=str(model_path), name=prof.name, torso=m.body(torso).name, joints=joints,
                       feet=feet, default_pose=default, stand_height=float(q0[m.jnt_qposadr[free] + 2]),
                       kp=REF_KP * k, kd=REF_KD * k, torque_limits=taus, notes=notes, driven=driven)


def build_spec(s: LeggedSetup, training: bool = True, standalone: bool = True) -> mujoco.MjSpec:
    """Robot + sensors (+ floor and a home keyframe if standalone). training=True: feet-only contacts
    and the fast solver used on the GPU; otherwise all collisions, full solver, and feet that keep the
    training floor friction whatever floor they're put on (Morph's scene floor is slippery for the block)."""
    spec = mujoco.MjSpec.from_file(s.model_path)
    spec.option.timestep = SIM_DT
    if training:
        spec.option.iterations, spec.option.ls_iterations = 1, 5
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
        spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    else:
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
        spec.option.iterations, spec.option.ls_iterations = 100, 50

    # position servos in place of torque motors; torque limits unchanged
    for a, (act, tau) in enumerate(zip(spec.actuators, s.torque_limits)):
        j = spec.joint(s.joints[a])
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = s.kp
        act.biasprm[0], act.biasprm[1], act.biasprm[2] = 0.0, -s.kp, 0.0
        act.gear[0] = 1.0
        act.ctrllimited, act.ctrlrange = True, list(j.range)
        if tau > 0:  # (Spot's servos give no limit: 0 here, and a [0, 0] range wouldn't compile)
            act.forcelimited, act.forcerange = True, [-tau, tau]
    # use the robot's own IMU mount if the model has one, otherwise put one at the torso origin
    own = [st for st in spec.sites if "imu" in st.name.lower()]
    imu = own[0].name if own else "morph_imu"
    if not own:
        spec.body(s.torso).add_site(name=imu, pos=[0, 0, 0])
    for i, f in enumerate(s.feet):
        spec.body(f["body"]).add_site(name=f"morph_foot{i}", pos=f["pos"], size=[max(f["radius"], 0.01)] * 3)

    S, SITE = mujoco.mjtSensor, mujoco.mjtObj.mjOBJ_SITE
    for kind, nm in ((S.mjSENS_GYRO, "gyro"), (S.mjSENS_VELOCIMETER, "local_linvel"),
                     (S.mjSENS_ACCELEROMETER, "accelerometer"), (S.mjSENS_FRAMEZAXIS, "upvector"),
                     (S.mjSENS_FRAMELINVEL, "global_linvel"), (S.mjSENS_FRAMEANGVEL, "global_angvel")):
        spec.add_sensor(name=f"morph_{nm}", type=kind, objtype=SITE, objname=imu)
    for i in range(len(s.feet)):
        spec.add_sensor(name=f"morph_foot{i}_global_linvel", type=S.mjSENS_FRAMELINVEL, objtype=SITE,
                        objname=f"morph_foot{i}")

    feet = {f["geom_id"] for f in s.feet}
    if training:  # only the feet touch anything; the floor only touches feet
        for gi, g in enumerate(spec.geoms):
            g.contype, g.conaffinity = (0, 1) if gi in feet else (0, 0)
    else:
        for gi, g in enumerate(spec.geoms):
            if gi in feet:
                g.priority = 2
                g.friction = [max(g.friction[0], 0.6), g.friction[1], g.friction[2]]
                g.condim = max(g.condim, 3)
    if not standalone:
        return spec
    floor = spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                                    friction=[0.6, 0.005, 0.0001], condim=3)
    floor.contype, floor.conaffinity = (1, 0) if training else (1, 1)
    floor.priority = 1
    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])

    # one "home" keyframe with the servo targets at the standing pose
    for k in list(spec.keys):
        spec.delete(k)
    m0 = mujoco.MjModel.from_xml_path(s.model_path)
    q0 = I.home_qpos(m0)
    spec.add_key(name="home", qpos=list(q0), ctrl=list(s.default_pose))
    return spec


def apply_joint_gains(m: mujoco.MjModel, s: LeggedSetup) -> None:
    for jn in s.joints:  # joint damping acts as the servo's D term, as in Playground's Go1
        dof = m.joint(jn).dofadr[0]
        m.dof_damping[dof] = s.kd
        # enough inertia for the damping to settle within a step: Spot's light wrist, given the legs' servo, ran away
        # in 0.14 s and every reward was NaN (the Go2's legs are unchanged: 0.005 is already more)
        m.dof_armature[dof] = max(m.dof_armature[dof], 0.005, 1.5 * s.kd * SIM_DT)


def build_model(s: LeggedSetup, training: bool = True) -> mujoco.MjModel:
    m = build_spec(s, training).compile()
    apply_joint_gains(m, s)
    return m


def main() -> None:
    import argparse
    import json
    from dataclasses import asdict
    ap = argparse.ArgumentParser(description="derive the walking-policy setup for a legged robot")
    ap.add_argument("model")
    a = ap.parse_args()
    s = legged_setup(a.model)
    m = build_model(s)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(int(2.0 / SIM_DT)):  # does it stand on the servos alone?
        mujoco.mj_step(m, d)
    print(json.dumps(asdict(s), indent=1))
    print(f"after 2 s standing on the servos: torso z {d.qpos[2]:.3f} (start {s.stand_height:.3f}), "
          f"up-vector z {d.sensor('morph_upvector').data[2]:.3f}")


if __name__ == "__main__":
    main()
