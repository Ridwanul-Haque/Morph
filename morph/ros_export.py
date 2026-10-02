"""Turn a learned policy into a ROS 2 package (ament_python) that runs it on a robot.

    <export>/ros2/morph_<robot>_policy/
      package.xml, setup.py, setup.cfg, resource/
      morph_<robot>_policy/policy_core.py   the policy + observation, no ROS (numpy + mujoco)
      morph_<robot>_policy/policy_node.py   ROS 2 node: /joint_states + object pose in, joint commands out
      morph_<robot>_policy/sim_bridge.py    a stand-in robot: MuJoCo sim that speaks the same topics
      launch/policy.launch.py               ros2 launch <pkg> policy.launch.py [sim:=true]
      config/params.yaml                    topics, joint names, command type, speed limit, frames
      policy/                               policy.npz, policy.json, model.xml, assets/

The node reads sensor_msgs/JointState and a geometry_msgs/PoseStamped for the object (and nav_msgs/Odometry
for robots with a floating base), rebuilds exactly the observation the policy was trained on (the end
bodies' positions come from forward kinematics on the training model, so they match), and publishes joint
position targets to a ros2_control controller: trajectory_msgs/JointTrajectory for a
joint_trajectory_controller, or std_msgs/Float64MultiArray for a forward_position_controller.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

CORE = r'''"""The learned policy and its observation, without ROS (numpy + mujoco only).

Poses come in the "input frame" (the robot's base frame for a bolted-down robot, your world/odom frame
for a mobile one) and are turned into the frame the policy was trained in (Morph's scene frame).
"""
import json
import math
from pathlib import Path

import mujoco
import numpy as np


def _yaw_of(q):  # quaternion w, x, y, z -> heading
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


class Frame2D:
    """x, y, yaw of one frame in another; maps points, quaternions and vectors."""

    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.x, self.y, self.yaw = x, y, yaw
        c, s = math.cos(yaw), math.sin(yaw)
        self.R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        self.q = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])

    def point(self, p):
        return self.R @ np.asarray(p, float) + [self.x, self.y, 0.0]

    def vector(self, v):
        return self.R @ np.asarray(v, float)

    def quat(self, q):
        return _qmul(self.q, np.asarray(q, float))

    def inverse(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return Frame2D(-(c * self.x + s * self.y), -(-s * self.x + c * self.y), -self.yaw)


class MorphPolicy:
    def __init__(self, policy_dir, speed_scale=1.0, scene_origin=(0.0, 0.0, 0.0)):
        d = Path(policy_dir)
        self.spec = json.loads((d / "policy.json").read_text())
        self.m = mujoco.MjModel.from_xml_path(str(d / "model.xml"))
        self.d = mujoco.MjData(self.m)
        z = np.load(d / "policy.npz")
        self.W = [z[f"W{i}"] for i in range(len([k for k in z.files if k.startswith("W")]))]
        self.B = [z[f"b{i}"] for i in range(len(self.W))]
        self.mean, self.std = z["obs_mean"], z["obs_std"]
        ix = self.ix = self.spec["indices"]
        self.qadr, self.vadr = np.array(ix["joint_qadr"], dtype=int), np.array(ix["joint_vadr"], dtype=int)
        self.tips, self.mask = np.array(ix["tips"]), np.array(ix["pos_mask"])
        self.q0, self.lo, self.hi = np.array(self.spec["default_pose"]), np.array(ix["act_lo"]), np.array(ix["act_hi"])
        self.target = np.array(self.spec["target"][:3])
        self.tyaw = self.spec["target"][3]
        self.floating = ix["base"] >= 0
        self.max_step = self.spec["max_step"] * min(max(speed_scale, 0.05), 1.0)
        # input frame -> scene frame: a bolted-down robot's base sits at the training mount
        mx, my, myaw = self.spec["mount"]
        self.to_scene = Frame2D(*scene_origin).inverse() if self.floating else Frame2D(mx, my, myaw)
        self.reset()

    @property
    def joints(self):
        """The joints the policy observes (their positions and speeds come from joint_states)."""
        return list(self.spec["joints"])

    @property
    def commands(self):
        """One command per actuator: a joint's name, or the actuator's (e.g. a tendon-driven telescoping arm)."""
        return list(self.spec["commands"])

    def reset(self):
        """Start again from wherever the robot is now (the first step's commands hold the current pose)."""
        self.d.qpos[:] = self.spec["initial_qpos"]
        self.d.qvel[:] = 0
        self.last = np.zeros(self.m.nu)
        self.ctrl = None

    def set_state(self, joint_pos, joint_vel, obj_pos, obj_quat, obj_vel=(0, 0, 0),
                  base_pos=None, base_quat=None, base_lin_vel=None, base_ang_vel=None):
        """joint_* in `joints` order; poses in the input frame (quaternions w, x, y, z);
        base velocities as nav_msgs/Odometry gives them (in the base's own frame)."""
        d, ix, f = self.d, self.ix, self.to_scene
        d.qpos[self.qadr] = joint_pos
        d.qvel[self.vadr] = joint_vel
        b = self.m.jnt_qposadr[self.m.body_jntadr[ix["block"]]]
        d.qpos[b:b + 3] = f.point(obj_pos)
        d.qpos[b + 3:b + 7] = f.quat(obj_quat)
        d.qvel[ix["block_v"]:ix["block_v"] + 3] = f.vector(obj_vel)
        if self.floating and base_pos is not None:
            a = self.m.jnt_qposadr[self.m.body_jntadr[ix["base"]]]
            d.qpos[a:a + 3] = f.point(base_pos)
            d.qpos[a + 3:a + 7] = f.quat(base_quat)
            mujoco.mj_kinematics(self.m, d)
            Rb = d.xmat[ix["base"]].reshape(3, 3)
            v = ix["base_v"]
            d.qvel[v:v + 3] = Rb @ np.asarray(base_lin_vel if base_lin_vel is not None else (0, 0, 0), float)
            d.qvel[v + 3:v + 6] = base_ang_vel if base_ang_vel is not None else (0, 0, 0)
        mujoco.mj_forward(self.m, d)  # positions of the robot's end parts, actuator lengths and speeds

    def observation(self):
        d, ix = self.d, self.ix
        bp = d.xpos[ix["block"]]
        R = d.xmat[ix["block"]].reshape(3, 3)
        yaw = math.atan2(R[1, 0], R[0, 0])
        parts = [(d.actuator_length - self.q0) * self.mask, d.actuator_velocity * 0.1, bp - self.target, R[:, 2],
                 [math.cos(yaw - self.tyaw), math.sin(yaw - self.tyaw)],
                 d.qvel[ix["block_v"]:ix["block_v"] + 3] * 0.5, (d.xpos[self.tips] - bp).ravel(), self.last]
        if self.floating:
            Rb = d.xmat[ix["base"]].reshape(3, 3)
            v = ix["base_v"]
            parts += [d.xpos[ix["base"]] - bp, Rb.T @ [0, 0, -1.0], Rb.T @ d.qvel[v:v + 3] * 0.5, d.qvel[v + 3:v + 6] * 0.2]
        return np.concatenate(parts)

    def act(self, obs):
        x = (obs - self.mean) / self.std
        for i, (w, b) in enumerate(zip(self.W, self.B)):
            x = x @ w + b
            if i < len(self.W) - 1:
                x = x / (1 + np.exp(-x))
        return np.tanh(x[: self.W[-1].shape[1] // 2])

    def step(self):
        """One control step: the next command per actuator (`commands` order)."""
        if self.ctrl is None:
            self.ctrl = np.clip(self.d.actuator_length * self.mask, self.lo, self.hi)
        self.last = self.act(self.observation())
        want = np.clip(self.q0 + self.last * self.spec["action_scale"], self.lo, self.hi)
        self.ctrl = self.ctrl + np.clip(want - self.ctrl, -self.max_step, self.max_step)
        return self.ctrl.copy()

    def distance(self):
        bp = self.d.xpos[self.ix["block"]]
        return float(np.hypot(*(bp[:2] - self.target[:2])) + abs(bp[2] - self.target[2]))

    def scene_to_input(self):
        return self.to_scene.inverse()
'''

NODE = r'''"""ROS 2 node that runs the policy Morph learned for __ROBOT__.

In:  joint_states (sensor_msgs/JointState), object_pose (geometry_msgs/PoseStamped)__ODOM_DOC__
Out: joint position targets for a ros2_control controller (see config/params.yaml), and
     ~/status (std_msgs/String, JSON: running, distance to the goal in metres)
"""
import json
import math
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .policy_core import MorphPolicy


def policy_dir(node):
    p = node.get_parameter("policy_dir").value
    if p:
        return p
    from ament_index_python.packages import get_package_share_directory
    return str(Path(get_package_share_directory("__PKG__")) / "policy")


class PolicyNode(Node):
    def __init__(self):
        super().__init__("morph_policy")
        for name, default in (("policy_dir", ""), ("joint_names", [""]), ("command_names", [""]),
                              ("joint_states_topic", "/joint_states"),
                              ("object_pose_topic", "/morph/object_pose"), ("odom_topic", "/odom"),
                              ("command_mode", "trajectory"),
                              ("command_topic", "/joint_trajectory_controller/joint_trajectory"),
                              ("speed_scale", 1.0), ("max_duration", 0.0), ("autostart", True),
                              ("scene_origin", [0.0, 0.0, 0.0]), ("stale_after", 0.5)):
            self.declare_parameter(name, default)
        g = lambda n: self.get_parameter(n).value
        self.core = MorphPolicy(policy_dir(self), g("speed_scale"), tuple(g("scene_origin")))
        names = [n for n in g("joint_names") if n]
        self.joint_names = names or self.core.joints  # your robot's names for the observed joints, same order
        if len(self.joint_names) != len(self.core.joints):
            raise ValueError(f"joint_names needs {len(self.core.joints)} names ({self.core.joints})")
        cmds = [n for n in g("command_names") if n]
        self.command_names = cmds or self.core.commands
        if len(self.command_names) != len(self.core.commands):
            raise ValueError(f"command_names needs {len(self.core.commands)} names ({self.core.commands})")
        self.mode, self.dt = g("command_mode"), self.core.spec["control_dt"]
        self.max_duration = g("max_duration") or 1.5 * self.core.spec["episode_s"]
        self.stale = g("stale_after")
        self.js = self.obj = self.odom = None
        self.obj_prev, self.obj_vel = None, np.zeros(3)
        self.running, self.t_start = bool(g("autostart")), None
        self.create_subscription(JointState, g("joint_states_topic"), self.on_joints, 10)
        self.create_subscription(PoseStamped, g("object_pose_topic"), self.on_object, 10)
        if self.core.floating:
            self.create_subscription(Odometry, g("odom_topic"), self.on_odom, 10)
        self.create_subscription(Bool, "~/enable", self.on_enable, 10)
        msg_type = JointTrajectory if self.mode == "trajectory" else Float64MultiArray
        self.pub = self.create_publisher(msg_type, g("command_topic"), 10)
        self.status = self.create_publisher(String, "~/status", 10)
        self.create_timer(self.dt, self.tick)
        self.get_logger().info(f"policy for {self.core.spec['robot']}: {len(self.command_names)} commands, "
                               f"{1 / self.dt:.0f} Hz, {self.mode} commands on {g('command_topic')}")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_enable(self, msg):
        self.running, self.t_start = msg.data, None
        if msg.data:
            self.core.reset()

    def on_joints(self, msg):
        self.js = msg

    def on_object(self, msg):
        t = self.now()
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        if self.obj_prev is not None and t - self.obj_prev[0] > 1e-3:
            v = (p - self.obj_prev[1]) / (t - self.obj_prev[0])
            self.obj_vel = 0.5 * self.obj_vel + 0.5 * v  # finite difference, lightly smoothed
        self.obj_prev = (t, p)
        o = msg.pose.orientation
        self.obj = (t, p, np.array([o.w, o.x, o.y, o.z]))

    def on_odom(self, msg):
        self.odom = msg

    def joint_positions(self):
        idx = {n: i for i, n in enumerate(self.js.name)}
        missing = [n for n in self.joint_names if n not in idx]
        if missing:
            raise KeyError(f"joint_states has no {missing[0]} (set joint_names in params.yaml)")
        pos = np.array([self.js.position[idx[n]] for n in self.joint_names])
        vel = np.array([self.js.velocity[idx[n]] if len(self.js.velocity) > idx[n] else 0.0 for n in self.joint_names])
        return pos, vel

    def tick(self):
        if not self.running or self.js is None or self.obj is None or (self.core.floating and self.odom is None):
            return
        if self.now() - self.obj[0] > self.stale:
            self.get_logger().warn("object pose is stale: holding", throttle_duration_sec=2.0)
            return
        if self.t_start is None:
            self.t_start = self.now()
            self.core.reset()
        if self.now() - self.t_start > self.max_duration:
            self.running = False
            self.get_logger().info(f"done after {self.max_duration:.0f} s; publish true on ~/enable to run again")
            self.publish_status()
            return
        try:
            pos, vel = self.joint_positions()
        except KeyError as e:
            self.get_logger().error(str(e), throttle_duration_sec=5.0)
            return
        kw = {}
        if self.core.floating:
            p, q = self.odom.pose.pose.position, self.odom.pose.pose.orientation
            tw = self.odom.twist.twist
            kw = dict(base_pos=(p.x, p.y, p.z), base_quat=(q.w, q.x, q.y, q.z),
                      base_lin_vel=(tw.linear.x, tw.linear.y, tw.linear.z),
                      base_ang_vel=(tw.angular.x, tw.angular.y, tw.angular.z))
        self.core.set_state(pos, vel, self.obj[1], self.obj[2], self.obj_vel, **kw)
        self.publish(self.core.step())
        self.publish_status()

    def publish(self, targets):
        if self.mode == "trajectory":
            msg = JointTrajectory()
            msg.joint_names = list(self.command_names)
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in targets]
            ns = int(self.dt * 2 * 1e9)  # reach it by the next command, a little slack
            pt.time_from_start = Duration(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
            msg.points = [pt]
        else:
            msg = Float64MultiArray(data=[float(x) for x in targets])
        self.pub.publish(msg)

    def publish_status(self):
        self.status.publish(String(data=json.dumps({"running": self.running, "distance_m": round(self.core.distance(), 4)})))


def main():
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
'''

BRIDGE = r'''"""A stand-in robot for trying the policy node without hardware: the training scene in MuJoCo,
publishing the same topics a real setup would (joint_states, the object's pose from a "camera", odometry)
and following the joint commands the node sends. ros2 launch __PKG__ policy.launch.py sim:=true
"""
from pathlib import Path

import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory

from .policy_core import MorphPolicy
from .policy_node import policy_dir


class SimBridge(Node):
    def __init__(self):
        super().__init__("morph_sim_bridge")
        for name, default in (("policy_dir", ""), ("joint_names", [""]), ("command_names", [""]),
                              ("joint_states_topic", "/joint_states"),
                              ("object_pose_topic", "/morph/object_pose"), ("odom_topic", "/odom"),
                              ("command_mode", "trajectory"),
                              ("command_topic", "/joint_trajectory_controller/joint_trajectory"),
                              ("scene_origin", [0.0, 0.0, 0.0]), ("speed_scale", 1.0), ("max_duration", 0.0),
                              ("autostart", True), ("stale_after", 0.5)):
            self.declare_parameter(name, default)
        g = lambda n: self.get_parameter(n).value
        d = Path(policy_dir(self))
        core = MorphPolicy(d, 1.0, tuple(g("scene_origin")))  # only for its frames and indices
        self.spec, self.ix, self.out = core.spec, core.ix, core.scene_to_input()
        self.m = mujoco.MjModel.from_xml_path(str(d / "model.xml"))
        self.d = mujoco.MjData(self.m)
        self.d.qpos[:] = self.spec["initial_qpos"]
        self.d.ctrl[:] = np.clip(core.q0, core.lo, core.hi)
        mujoco.mj_forward(self.m, self.d)
        self.qadr, self.vadr = core.qadr, core.vadr
        self.names = [n for n in g("joint_names") if n] or core.joints
        self.cmd_names = [n for n in g("command_names") if n] or core.commands
        self.floating = core.floating
        self.pub_js = self.create_publisher(JointState, g("joint_states_topic"), 10)
        self.pub_obj = self.create_publisher(PoseStamped, g("object_pose_topic"), 10)
        self.pub_odom = self.create_publisher(Odometry, g("odom_topic"), 10) if self.floating else None
        if g("command_mode") == "trajectory":
            self.create_subscription(JointTrajectory, g("command_topic"), self.on_traj, 10)
        else:
            self.create_subscription(Float64MultiArray, g("command_topic"), self.on_array, 10)
        self.dt = self.spec["control_dt"]
        self.create_timer(self.dt, self.tick)

    def on_traj(self, msg):
        idx = {n: i for i, n in enumerate(msg.joint_names)}
        self.d.ctrl[:] = [msg.points[-1].positions[idx[n]] for n in self.cmd_names]

    def on_array(self, msg):
        self.d.ctrl[:] = msg.data

    def tick(self):
        for _ in range(max(1, round(self.dt / self.m.opt.timestep))):
            mujoco.mj_step(self.m, self.d)
        stamp = self.get_clock().now().to_msg()
        js = JointState()
        js.header.stamp = stamp
        js.name = list(self.names)
        js.position = [float(x) for x in self.d.qpos[self.qadr]]
        js.velocity = [float(x) for x in self.d.qvel[self.vadr]]
        self.pub_js.publish(js)
        b = self.ix["block"]
        p, q = self.out.point(self.d.xpos[b]), self.out.quat(self.d.xquat[b])
        ps = PoseStamped()
        ps.header.stamp, ps.header.frame_id = stamp, "base_link" if not self.floating else "odom"
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, p)
        ps.pose.orientation.w, ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z = map(float, q)
        self.pub_obj.publish(ps)
        if self.floating:
            base = self.ix["base"]
            od = Odometry()
            od.header.stamp, od.header.frame_id, od.child_frame_id = stamp, "odom", "base_link"
            bp, bq = self.out.point(self.d.xpos[base]), self.out.quat(self.d.xquat[base])
            od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = map(float, bp)
            o = od.pose.pose.orientation
            o.w, o.x, o.y, o.z = map(float, bq)
            v = self.ix["base_v"]
            Rb = self.d.xmat[base].reshape(3, 3)
            lin = Rb.T @ self.d.qvel[v:v + 3]  # Odometry twists are in the child (base) frame
            od.twist.twist.linear.x, od.twist.twist.linear.y, od.twist.twist.linear.z = map(float, lin)
            ang = self.d.qvel[v + 3:v + 6]
            od.twist.twist.angular.x, od.twist.twist.angular.y, od.twist.twist.angular.z = map(float, ang)
            self.pub_odom.publish(od)


def main():
    rclpy.init()
    node = SimBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
'''

LAUNCH = r'''from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params = LaunchConfiguration("params_file")
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=PathJoinSubstitution(
            [FindPackageShare("__PKG__"), "config", "params.yaml"])),
        DeclareLaunchArgument("sim", default_value="false",
                              description="also start a MuJoCo stand-in robot publishing the same topics"),
        Node(package="__PKG__", executable="sim_bridge", name="morph_sim_bridge", parameters=[params],
             condition=IfCondition(LaunchConfiguration("sim")), output="screen"),
        Node(package="__PKG__", executable="policy_node", name="morph_policy", parameters=[params], output="screen"),
    ])
'''

PACKAGE_XML = '''<?xml version="1.0"?>
<package format="3">
  <name>__PKG__</name>
  <version>0.1.0</version>
  <description>Policy learned by Morph for __ROBOT__: __GOAL__</description>
  <maintainer email="you@example.com">you</maintainer>
  <license>Apache-2.0</license>
  <exec_depend>rclpy</exec_depend>
  <exec_depend>builtin_interfaces</exec_depend>
  <exec_depend>geometry_msgs</exec_depend>
  <exec_depend>nav_msgs</exec_depend>
  <exec_depend>sensor_msgs</exec_depend>
  <exec_depend>std_msgs</exec_depend>
  <exec_depend>trajectory_msgs</exec_depend>
  <exec_depend>launch_ros</exec_depend>
  <exec_depend>ament_index_python</exec_depend>
  <exec_depend>python3-numpy</exec_depend>
  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
'''

SETUP = '''import os
from glob import glob

from setuptools import setup

package_name = "__PKG__"


def tree(src, dst):
    out = {}
    for f in glob(os.path.join(src, "**", "*"), recursive=True):
        if os.path.isfile(f):
            out.setdefault(os.path.join(dst, os.path.relpath(os.path.dirname(f), src)), []).append(f)
    return [(k.rstrip("/."), v) for k, v in out.items()]


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ] + tree("policy", "share/" + package_name + "/policy"),
    install_requires=["setuptools"],
    zip_safe=True,
    description="Policy learned by Morph",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "policy_node = " + package_name + ".policy_node:main",
        "sim_bridge = " + package_name + ".sim_bridge:main",
    ]},
)
'''


def package_name(robot: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", robot.lower()).strip("_")
    return f"morph_{s or 'robot'}_policy"


def _yaml_list(xs) -> str:
    return "[" + ", ".join(json.dumps(x) for x in xs) + "]"


def write_ros2_package(export_dir: Path, policy: dict, goal_text: str) -> Path:
    """export_dir holds policy.npz / policy.json / model.xml / assets (a Morph export)."""
    pkg = package_name(policy["robot"])
    root = export_dir / "ros2" / pkg
    if root.exists():
        shutil.rmtree(root)
    (root / pkg).mkdir(parents=True)
    (root / "resource").mkdir()
    (root / "resource" / pkg).write_text("")
    (root / "launch").mkdir()
    (root / "config").mkdir()
    pol = root / "policy"
    pol.mkdir()
    for f in ("policy.npz", "policy.json", "model.xml"):
        shutil.copy(export_dir / f, pol / f)
    if (export_dir / "assets").exists():
        shutil.copytree(export_dir / "assets", pol / "assets")
    floating = policy["indices"]["base"] >= 0
    sub = lambda s: (s.replace("__PKG__", pkg).replace("__ROBOT__", policy["robot"]).replace("__GOAL__", goal_text)
                     .replace("__ODOM_DOC__", ", odom (nav_msgs/Odometry)" if floating else ""))
    (root / pkg / "__init__.py").write_text("")
    (root / pkg / "policy_core.py").write_text(CORE)
    (root / pkg / "policy_node.py").write_text(sub(NODE))
    (root / pkg / "sim_bridge.py").write_text(sub(BRIDGE))
    (root / "launch" / "policy.launch.py").write_text(sub(LAUNCH))
    (root / "package.xml").write_text(sub(PACKAGE_XML))
    (root / "setup.py").write_text(sub(SETUP))
    (root / "setup.cfg").write_text(f"[develop]\nscript_dir=$base/lib/{pkg}\n[install]\ninstall_scripts=$base/lib/{pkg}\n")
    frame = ("odom (your world frame; set scene_origin to where Morph's scene origin is in it)" if floating
             else "the robot's base frame (base_link)")
    (root / "config" / "params.yaml").write_text(f"""# Settings for the policy node (and the sim stand-in). Edit to match your robot.
/**:
  ros__parameters:
    # Your robot's names for the joints the policy watches (read from joint_states), in this order
    # (leave as is if they're the same):
    joint_names: {_yaml_list(policy['joints'])}
    # Your controller's names for the commands the policy sends, in this order:
    command_names: {_yaml_list(policy['commands'])}
    joint_states_topic: /joint_states
    # The object's pose (geometry_msgs/PoseStamped) in {frame}: from your perception or a mocap system
    object_pose_topic: /morph/object_pose
    odom_topic: /odom
    # trajectory: trajectory_msgs/JointTrajectory to a joint_trajectory_controller
    # position:   std_msgs/Float64MultiArray to a forward_position_controller (e.g. /forward_position_controller/commands)
    command_mode: trajectory
    command_topic: /joint_trajectory_controller/joint_trajectory
    # 1.0 = joint targets move at most {policy['max_step']} rad per {policy['control_dt']} s step, as trained. Start lower on hardware.
    speed_scale: 1.0
    # stop after this many seconds (0 = 1.5 x the {policy['episode_s']:.0f} s training episode); publish true on /morph_policy/enable to rerun
    max_duration: 0.0
    autostart: true
    scene_origin: [0.0, 0.0, 0.0]
    # hold still if no object pose arrived for this long (s)
    stale_after: 0.5
""")
    (root / "README.md").write_text(f"""# {pkg}

The policy Morph learned for **{policy['robot']}** ({goal_text}), as a ROS 2 package.

```bash
cp -r {pkg} ~/ros2_ws/src/ && cd ~/ros2_ws
# the node needs `mujoco` (for forward kinematics on the training model) in the Python ROS runs on:
python3 -m venv --system-site-packages .venv && .venv/bin/pip install mujoco
source /opt/ros/$ROS_DISTRO/setup.bash && source .venv/bin/activate
python3 -m colcon build --packages-select {pkg} && source install/setup.bash   # python3 -m: nodes use the venv
ros2 launch {pkg} policy.launch.py sim:=true    # try it against a simulated stand-in first
ros2 launch {pkg} policy.launch.py              # then your robot (edit config/params.yaml)
ros2 topic echo /morph_policy/status            # running, distance to the goal
```

(Or `pip install --user mujoco` and a plain `colcon build`.)

**In:** `/joint_states` (sensor_msgs/JointState), the object's pose on `/morph/object_pose`
(geometry_msgs/PoseStamped, in {frame}){', and `/odom` (nav_msgs/Odometry)' if floating else ''}.
**Out:** joint position targets to a ros2_control controller: a `JointTrajectory` for a
`joint_trajectory_controller` (default) or a `Float64MultiArray` for a `forward_position_controller`.

The node rebuilds exactly what the policy saw in training (joint angles and speeds, the object relative to
the goal, the robot's end parts relative to the object via forward kinematics on the training model) and
runs it at {1 / policy['control_dt']:.0f} Hz. Joint targets move smoothly (at most {policy['max_step']} rad per step times
`speed_scale`).

It was trained in simulation{' with physics variation (' + policy['randomize'] + ')' if policy.get('randomize') not in (None, 'off') else ''}. On real hardware, start with a low
`speed_scale`, keep an emergency stop in reach, and expect to fine-tune.
""")
    return root
