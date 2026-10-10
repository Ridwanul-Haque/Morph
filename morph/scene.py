"""MJCF scene builders.

Every scene shares the same floor, arrow-marked block and goal marker so that
source and target robots are evaluated against an identical object-space task.
The arms are generated from explicit link-length parameters (see ArmGeometry);
the quadruped is the Unitree Go1 from MuJoCo Menagerie (BSD-3-Clause).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]
MENAGERIE = REPO_ROOT / "third_party" / "mujoco_menagerie"
GO1_XML = MENAGERIE / "unitree_go1" / "go1.xml"

TIMESTEP = 0.002
# Block-floor friction is set by the block geom: the floor is deliberately low
# friction and MuJoCo combines pair friction with max(), so the block's value wins.
FLOOR_FRICTION = 0.1


@dataclass(frozen=True)
class BlockSpec:
    """The manipulated object. Arrow on top points along the block's local +x."""

    half_x: float = 0.025  # 5 x 5 x 6 cm: fits the Panda's 8 cm gripper
    half_y: float = 0.025
    half_z: float = 0.03
    mass: float = 0.2
    friction: float = 0.6
    # Static things from a user's scene (bins, platforms, walls), as world-frame MJCF bodies. They go
    # wherever the block goes, so every robot's scene and the push planner's model contain them.
    fixtures_xml: str = ""
    # its shape: "box" (half_x, half_y, half_z), "sphere" (radius half_z), "cylinder" (radius half_x, half-height
    # half_z, standing up), "capsule" (radius half_x, half_z to the tip of a cap). Everything that plans or checks
    # positions treats it as its bounding box (half_x, half_y, half_z).
    shape: str = "box"
    # shape "mesh": the object's own 3D model (an STL / OBJ, in metres after mesh_scale), placed in the object's frame
    # by mesh_pos / mesh_quat (centred, resting flat). Collisions use its convex hull; checks use half_x/y/z.
    mesh_file: str = ""
    mesh_scale: tuple = (1.0, 1.0, 1.0)
    mesh_pos: tuple = (0.0, 0.0, 0.0)
    mesh_quat: tuple = (1.0, 0.0, 0.0, 0.0)

    def as_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "fixtures_xml"}


def mesh_name(block) -> str:
    import hashlib
    key = f"{block.mesh_file}|{tuple(round(float(v), 9) for v in block.mesh_scale)}"
    return "objmesh_" + hashlib.sha1(key.encode()).hexdigest()[:10]


@dataclass(frozen=True)
class ArmGeometry:
    """Yaw-pitch-pitch-pitch-roll arm with a parallel-jaw gripper.

    Zero pose points straight up. Pitch joints rotate about the local y axis,
    so the cumulative pitch is the link's angle from vertical.
    """

    name: str
    base_xy: tuple[float, float] = (0.0, 0.0)
    pedestal: float = 0.25  # shoulder height above floor
    l1: float = 0.35  # upper arm
    l2: float = 0.30  # forearm
    l3: float = 0.14  # wrist pitch axis to tool centre point (TCP)
    max_opening: float = 0.075  # per finger, from hand centreline
    grip_force: float = 25.0
    color: tuple[float, float, float] = (0.25, 0.45, 0.85)
    joint_ranges: dict = field(
        default_factory=lambda: {
            "j1": (-math.pi, math.pi),
            "j2": (-0.6, 2.4),
            "j3": (0.0, 2.9),
            "j4": (-1.6, 3.1),
            "j5": (-6.2, 6.2),  # continuous-style wrist roll
        }
    )

    def as_dict(self) -> dict:
        return asdict(self)


# Two materially different arms. B has a 49% longer upper arm, a shorter
# forearm, a 10 cm lower shoulder and a longer wrist. Over the task workspace,
# A's joint solutions put B's gripper a median 12 cm (10th pct. 9 cm) away from
# the intended point, so joint-space knowledge does not carry over directly.
ARM_A = ArmGeometry(name="arm_A")
ARM_B = ArmGeometry(
    name="arm_B",
    pedestal=0.15,
    l1=0.52,
    l2=0.26,
    l3=0.18,
    color=(0.85, 0.45, 0.2),
)


def _block_xml(block: BlockSpec) -> str:
    hx, hy, hz = block.half_x, block.half_y, block.half_z
    top = hz + 0.0006
    shape = getattr(block, "shape", "box")
    if shape == "mesh":  # the object's own 3D model (its convex hull collides)
        pos, q = " ".join(f"{v:.6g}" for v in block.mesh_pos), " ".join(f"{v:.6g}" for v in block.mesh_quat)
        return f"""
    <body name="block" pos="0.45 -0.15 {hz}">
      <freejoint name="block_free"/>
      <geom name="block_geom" type="mesh" mesh="{mesh_name(block)}" pos="{pos}" quat="{q}" mass="{block.mass}"
            friction="{block.friction} 0.005 0.0001" rgba="0.95 0.75 0.2 1" condim="4"/>
      <site name="block_center" pos="0 0 0" size="0.005"/>
    </body>{block.fixtures_xml}"""
    if shape != "box":
        size = {"sphere": f"{hz}", "cylinder": f"{hx} {hz}", "capsule": f"{hx} {max(hz - hx, 0.001)}"}[shape]
        return f"""
    <body name="block" pos="0.45 -0.15 {hz}">
      <freejoint name="block_free"/>
      <geom name="block_geom" type="{shape}" size="{size}" mass="{block.mass}"
            friction="{block.friction} 0.005 0.0001" rgba="0.95 0.75 0.2 1" condim="{6 if shape == 'sphere' else 4}"/>
      <!-- visual-only stripe, so turning and rolling show -->
      <geom type="box" size="{hx * 0.9} {hx * 0.12} {hz * 0.9}" contype="0" conaffinity="0" mass="0" rgba="0.8 0.1 0.1 1"/>
      <site name="block_center" pos="0 0 0" size="0.005"/>
    </body>{block.fixtures_xml}"""
    return f"""
    <body name="block" pos="0.45 -0.15 {hz}">
      <freejoint name="block_free"/>
      <geom name="block_geom" type="box" size="{hx} {hy} {hz}" mass="{block.mass}"
            friction="{block.friction} 0.005 0.0001" rgba="0.95 0.75 0.2 1" condim="4"/>
      <!-- visual-only asymmetric arrow along local +x; no collision -->
      <geom type="box" size="{hx * 0.55} {hy * 0.12} 0.0006" pos="{-hx * 0.2} 0 {top}"
            contype="0" conaffinity="0" mass="0" rgba="0.1 0.1 0.1 1"/>
      <geom type="box" size="{hx * 0.28} {hx * 0.28} 0.0006" pos="{hx * 0.45} 0 {top}"
            euler="0 0 0.7854" contype="0" conaffinity="0" mass="0" rgba="0.1 0.1 0.1 1"/>
      <geom type="box" size="0.0006 {hy * 0.9} {hz * 0.9}" pos="{hx + 0.0006} 0 0"
            contype="0" conaffinity="0" mass="0" rgba="0.8 0.1 0.1 1"/>
      <site name="block_center" pos="0 0 0" size="0.005"/>
    </body>{block.fixtures_xml}"""


def _goal_xml() -> str:
    # Mocap body: moved at runtime to show the confirmed goal. Visual only.
    return """
    <body name="goal" mocap="true" pos="0.45 0.2 0.0005">
      <geom type="cylinder" size="0.05 0.0005" contype="0" conaffinity="0" rgba="0.1 0.8 0.2 0.35"/>
      <geom type="box" size="0.05 0.004 0.0007" pos="0.0 0 0" contype="0" conaffinity="0" rgba="0.05 0.5 0.1 1"/>
      <geom type="box" size="0.012 0.012 0.0007" pos="0.045 0 0" euler="0 0 0.7854"
            contype="0" conaffinity="0" rgba="0.05 0.5 0.1 1"/>
    </body>"""


def _common_assets(block=None) -> str:
    """The shared look of every scene, plus the object's own mesh if it is one (only this model's object)."""
    return """
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global azimuth="135" elevation="-30" offwidth="1280" offheight="720"/>
    <quality shadowsize="8192" offsamples="8"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.5 0.7" rgb2="0.05 0.05 0.1" width="256" height="1536"/>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.60 0.64 0.68"
             rgb2="0.52 0.56 0.60" markrgb="0.35 0.35 0.38" width="300" height="300"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="10 10" reflectance="0"/>""" + (
        f'''
    <mesh name="{mesh_name(block)}" file="{block.mesh_file}" scale="{block.mesh_scale[0]} {block.mesh_scale[1]} {block.mesh_scale[2]}" maxhullvert="64"/>'''
        if block is not None and getattr(block, "mesh_file", "") else "") + """
  </asset>"""


def _arm_xml(g: ArmGeometry) -> str:
    r = g.joint_ranges
    c = " ".join(f"{v:.3f}" for v in g.color) + " 1"
    bx, by = g.base_xy
    fz = g.l3  # finger centre sits at the TCP
    return f"""
    <body name="arm_base" pos="{bx} {by} 0" childclass="arm">
      <geom type="cylinder" size="0.06 {g.pedestal / 2}" pos="0 0 {g.pedestal / 2}" rgba="0.3 0.3 0.3 1"/>
      <body name="link1" pos="0 0 {g.pedestal}" gravcomp="1">
        <joint name="j1" axis="0 0 1" range="{r['j1'][0]} {r['j1'][1]}"/>
        <geom type="sphere" size="0.045" rgba="{c}" contype="0" conaffinity="0" mass="0.8"/>
        <body name="upper" gravcomp="1">
          <joint name="j2" axis="0 1 0" range="{r['j2'][0]} {r['j2'][1]}"/>
          <geom type="capsule" fromto="0 0 0 0 0 {g.l1}" size="0.03" rgba="{c}" mass="1.0"/>
          <body name="fore" pos="0 0 {g.l1}" gravcomp="1">
            <joint name="j3" axis="0 1 0" range="{r['j3'][0]} {r['j3'][1]}"/>
            <geom type="capsule" fromto="0 0 0 0 0 {g.l2}" size="0.025" rgba="{c}" mass="0.6"/>
            <body name="wrist" pos="0 0 {g.l2}" gravcomp="1">
              <joint name="j4" axis="0 1 0" range="{r['j4'][0]} {r['j4'][1]}"/>
              <geom type="sphere" size="0.03" rgba="0.2 0.2 0.2 1" mass="0.2"/>
              <body name="hand" gravcomp="1">
                <joint name="j5" axis="0 0 1" range="{r['j5'][0]} {r['j5'][1]}"/>
                <geom type="capsule" fromto="0 0 0 0 0 {fz - 0.05}" size="0.015" rgba="0.2 0.2 0.2 1" mass="0.1"/>
                <geom name="palm" type="box" size="0.02 {g.max_opening + 0.02} 0.01" pos="0 0 {fz - 0.045}"
                      rgba="0.2 0.2 0.2 1" mass="0.1"/>
                <site name="tcp" pos="0 0 {fz}" size="0.008" rgba="1 0 0 1"/>
                <body name="finger_l" pos="0 0 {fz}" gravcomp="1">
                  <joint name="fl" type="slide" axis="0 1 0" range="0 {g.max_opening}" damping="5"/>
                  <geom name="finger_l" type="box" size="0.018 0.008 0.034" pos="0 0.008 0"
                        friction="1.5 0.05 0.001" condim="4" solref="0.006 1" rgba="0.15 0.15 0.15 1" mass="0.05"/>
                </body>
                <body name="finger_r" pos="0 0 {fz}" gravcomp="1">
                  <joint name="fr" type="slide" axis="0 -1 0" range="0 {g.max_opening}" damping="5"/>
                  <geom name="finger_r" type="box" size="0.018 0.008 0.034" pos="0 -0.008 0"
                        friction="1.5 0.05 0.001" condim="4" solref="0.006 1" rgba="0.15 0.15 0.15 1" mass="0.05"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>"""


def _arm_actuators(g: ArmGeometry) -> str:
    fmax = g.grip_force
    kp_f = 2500.0  # squeeze = kp_f * GRIP_LEAD (see arm_skills), capped by grip_force
    return f"""
  <actuator>
    <position name="a1" joint="j1" kp="2000" dampratio="1" ctrlrange="{g.joint_ranges['j1'][0]} {g.joint_ranges['j1'][1]}" forcerange="-60 60"/>
    <position name="a2" joint="j2" kp="4000" dampratio="1" ctrlrange="{g.joint_ranges['j2'][0]} {g.joint_ranges['j2'][1]}" forcerange="-150 150"/>
    <position name="a3" joint="j3" kp="2500" dampratio="1" ctrlrange="{g.joint_ranges['j3'][0]} {g.joint_ranges['j3'][1]}" forcerange="-90 90"/>
    <position name="a4" joint="j4" kp="600" dampratio="1" ctrlrange="{g.joint_ranges['j4'][0]} {g.joint_ranges['j4'][1]}" forcerange="-30 30"/>
    <position name="a5" joint="j5" kp="200" dampratio="1" ctrlrange="{g.joint_ranges['j5'][0]} {g.joint_ranges['j5'][1]}" forcerange="-8 8"/>
    <position name="afl" joint="fl" kp="{kp_f}" ctrlrange="0 {g.max_opening}" forcerange="{-fmax} {fmax}"/>
    <position name="afr" joint="fr" kp="{kp_f}" ctrlrange="0 {g.max_opening}" forcerange="{-fmax} {fmax}"/>
  </actuator>"""


def arm_scene_xml(g: ArmGeometry, block: BlockSpec) -> str:
    return f"""<mujoco model="morph_{g.name}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{TIMESTEP}" integrator="implicit" cone="elliptic" impratio="10" noslip_iterations="4"/>
  <default>
    <joint armature="0.02" damping="1"/>
    <default class="arm"><geom contype="2" conaffinity="1"/></default>
  </default>
  {_common_assets(block)}
  <worldbody>
    <light pos="0.6 -0.5 2.6" dir="-0.15 0.2 -1" cutoff="45" exponent="2" castshadow="true" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" friction="{FLOOR_FRICTION} 0.005 0.0001"/>
    <camera name="overview" pos="1.35 -0.9 0.95" xyaxes="0.55 0.83 0 -0.42 0.28 0.86"/>
    <camera name="top" pos="0.45 0.0 1.5" xyaxes="1 0 0 0 1 0"/>
    {_goal_xml()}
    {_block_xml(block)}
    {_arm_xml(g)}
  </worldbody>
  {_arm_actuators(g)}
</mujoco>"""


def build_arm_model(g: ArmGeometry, block: BlockSpec) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(arm_scene_xml(g, block))


def build_quadruped_model(block: BlockSpec) -> mujoco.MjModel:
    """Go1 from Menagerie plus the shared floor, block and goal marker (via MjSpec)."""
    if not GO1_XML.exists():
        raise FileNotFoundError(
            f"{GO1_XML} missing. Run scripts/fetch_assets.sh to download MuJoCo Menagerie's unitree_go1."
        )
    robot = mujoco.MjSpec.from_file(str(GO1_XML))
    robot.option.timestep = TIMESTEP
    # Wrap our shared scene elements in an XML spec and attach the robot to it.
    scene_xml = f"""<mujoco model="morph_go1">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{TIMESTEP}" cone="elliptic" impratio="100"/>
  {_common_assets(block)}
  <worldbody>
    <light pos="0.6 -0.5 2.6" dir="-0.15 0.2 -1" cutoff="45" exponent="2" castshadow="true" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" friction="{FLOOR_FRICTION} 0.005 0.0001"/>
    <camera name="overview" pos="1.6 -1.3 1.1" xyaxes="0.63 0.77 0 -0.4 0.33 0.85"/>
    <camera name="top" pos="0.45 0.0 2.0" xyaxes="1 0 0 0 1 0"/>
    {_goal_xml()}
    {_block_xml(block)}
    <frame name="robot_mount"/>
  </worldbody>
</mujoco>"""
    scene = mujoco.MjSpec.from_string(scene_xml)
    frame = scene.worldbody.frames[0]
    frame.attach_body(robot.body("trunk"), "", "")
    # Actuators, defaults and assets travel with attach; keyframes do not, so the
    # quadruped adapter sets its home posture explicitly.
    return scene.compile()


PUSHER_TIP_X = 0.16  # tip axis ahead of the base origin (m)


def pusher_scene_xml(block: BlockSpec) -> str:
    """Arm-free holonomic mobile pusher. The body rides above the block (z >= 0.07 m) so
    only the vertical tip (radius 0.023 m, like Go1's foot) can touch it."""
    return f"""<mujoco model="morph_pusher">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{TIMESTEP}" integrator="implicitfast" cone="elliptic" impratio="10"/>
  {_common_assets(block)}
  <worldbody>
    <light pos="0.6 -0.5 2.6" dir="-0.15 0.2 -1" cutoff="45" exponent="2" castshadow="true" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" friction="{FLOOR_FRICTION} 0.005 0.0001"/>
    <camera name="overview" pos="1.35 -0.9 0.95" xyaxes="0.55 0.83 0 -0.42 0.28 0.86"/>
    <camera name="top" pos="0.45 0.0 1.5" xyaxes="1 0 0 0 1 0"/>
    {_goal_xml()}
    {_block_xml(block)}
    <body name="pusher_base" pos="0 0 0.10">
      <joint name="px" type="slide" axis="1 0 0" damping="20"/>
      <joint name="py" type="slide" axis="0 1 0" damping="20"/>
      <joint name="pyaw" type="hinge" axis="0 0 1" damping="2"/>
      <geom name="pusher_body" type="box" size="0.10 0.08 0.03" rgba="0.3 0.6 0.35 1" mass="3.0"/>
      <geom type="box" size="0.03 0.01 0.005" pos="0.07 0 0.032" rgba="1 1 1 1" contype="0" conaffinity="0" mass="0"/>
      <geom name="pusher_arm" type="box" size="0.04 0.012 0.01" pos="0.13 0 0" rgba="0.3 0.6 0.35 1" mass="0.1"/>
      <geom name="pusher_tip" type="capsule" fromto="{PUSHER_TIP_X} 0 -0.065 {PUSHER_TIP_X} 0 0" size="0.023"
            friction="0.8 0.02 0.01" rgba="0.15 0.15 0.15 1" mass="0.1"/>
      <site name="tip" pos="{PUSHER_TIP_X} 0 -0.065" size="0.006"/>
    </body>
  </worldbody>
  <actuator>
    <position name="ax" joint="px" kp="800" forcerange="-30 30"/>
    <position name="ay" joint="py" kp="800" forcerange="-30 30"/>
    <position name="ayaw" joint="pyaw" kp="60" forcerange="-6 6"/>
  </actuator>
</mujoco>"""


def build_pusher_model(block: BlockSpec) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(pusher_scene_xml(block))
