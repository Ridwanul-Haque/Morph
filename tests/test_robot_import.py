from pathlib import Path

import mujoco

from morph.introspect import profile_robot
from morph.robot_import import import_robot

DATA = Path(__file__).parent / "data"


def test_urdf_arm_gets_motors_and_stays_fixed():
    xml, notes = import_robot(DATA / "tiny_arm.urdf")
    m = mujoco.MjModel.from_xml_path(str(xml))
    assert m.nu == 8  # six arm joints + two fingers, all position servos now
    assert all(m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE for a in range(m.nu))
    p = profile_robot(xml)
    assert p.base == "fixed" and p.n_arms == 1 and p.gripper.get("type") == "parallel_gripper"


def test_urdf_legged_robot_gets_a_floating_base():
    xml, notes = import_robot(DATA / "tiny_dog.urdf")
    p = profile_robot(xml)
    assert p.base == "floating" and p.n_legs == 4 and p.kind == "quadruped"
    assert any("floating base" in n for n in notes)


def test_xacro_expands_with_ros_packages_and_meshes():
    # macros, an include through $(find pkg), an arg, and a package:// mesh
    xml, notes = import_robot(DATA / "ros_ws/src/tiny_arm_description/urdf/tiny_arm.urdf.xacro")
    m = mujoco.MjModel.from_xml_path(str(xml))
    assert m.nu == 4 and m.nmesh == 1  # the base's STL was found through its package
    assert abs(m.actuator_forcerange[0][1] - 40.0) < 1e-6  # the effort arg's default
    assert any("xacro" in n for n in notes)
