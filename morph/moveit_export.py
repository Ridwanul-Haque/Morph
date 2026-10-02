"""Hand a working pick-and-place to MoveIt 2.

When one of Morph's scripted arm runs succeeds, it has shown in simulation a sequence of arm
configurations that works for this robot, object and scene: above the object, at the grasp, lifted,
above the goal, placed. Those are recorded here, at the moments the skill changes phase, as joint angles
and as the hand's pose, both in the arm's base frame. The fixtures become collision boxes.

The export is a folder: plan.json plus morph_pick_place.py, a moveit_py script that adds the fixtures to
the planning scene and plans and executes each step with your robot's MoveIt config (joint-space goals
by default: they don't depend on how your MoveIt config names or orients its end-effector frame).
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOVEIT = ROOT / "runs" / "moveit"

# the skill's phase -> the waypoint reached when that phase ends (retries: "open" and "raise" start over)
WAYPOINTS = {"approach": ("above_object", "open"), "descend": ("grasp", "open"), "close": ("grasp", "closed"),
             "lift": ("lifted", "closed"), "transport": ("above_goal", "closed"), "lower": ("placed", "closed"),
             "release_retreat": ("released", "open")}


class Recorder:
    """on_tick callback for run_episode: keeps the arm's state at each phase change."""

    def __init__(self, ad):
        self.ad = ad
        m = ad.model
        self.joints = list(ad.cfg.arm_joints)
        self.qadr = [m.joint(j).qposadr[0] for j in self.joints]
        self.root = m.body(ad.cfg.root_body).id
        self.hand = m.body(ad.cfg.tcp_body).id
        self.tcp = m.site("tcp").id
        self.phase, self.steps = None, []

    def _in_base(self, p, R):
        d = self.ad.env.data
        Rb, pb = d.xmat[self.root].reshape(3, 3), d.xpos[self.root]
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, (Rb.T @ R).flatten())
        return [round(float(v), 5) for v in Rb.T @ (p - pb)], [round(float(v), 6) for v in q]

    def __call__(self, ad, phase):
        if phase == self.phase:
            return
        prev, self.phase = self.phase, phase
        if phase in ("open", "raise"):  # a retry: the sequence starts over
            self.steps = []
            return
        if prev not in WAYPOINTS:
            return
        d = ad.env.data
        name, grip = WAYPOINTS[prev]
        hp, hq = self._in_base(d.xpos[self.hand], d.xmat[self.hand].reshape(3, 3))
        tp, tq = self._in_base(d.site_xpos[self.tcp], d.site_xmat[self.tcp].reshape(3, 3))
        self.steps.append({"name": name, "after": prev, "gripper": grip,
                           "joints": {j: round(float(d.qpos[a]), 5) for j, a in zip(self.joints, self.qadr)},
                           "hand_pose": {"position": hp, "orientation_wxyz": hq}, "tcp_pose": {"position": tp, "orientation_wxyz": tq}})

    def finish(self):
        """One waypoint per step. The skill flickers between phases while it corrects (lift <-> transport
        near the carry height, lower <-> release while letting go), so: the last pose above the object,
        the grasp, the first pose after lifting, the last carried pose before lowering, the first placed
        pose, the final retreat."""
        self.__call__(self.ad, "done")
        s = self.steps
        first = lambda pred, xs=None: next((x for x in (xs if xs is not None else s) if pred(x)), None)
        idx = lambda x: s.index(x) if x is not None else len(s)
        placed = first(lambda x: x["name"] == "placed")
        closed = [x for x in s if x["name"] == "grasp" and x["gripper"] == "closed"]
        grasp_open = [x for x in s if x["name"] == "grasp" and x["gripper"] == "open"]
        after_grip = s[idx(closed[-1]) + 1:] if closed else s
        lifted = first(lambda x: x["name"] in ("lifted", "above_goal"), after_grip)
        carried = [x for x in s[:idx(placed)] if x["name"] in ("lifted", "above_goal") and idx(x) > idx(lifted)]
        above = [x for x in s[:idx(grasp_open[0] if grasp_open else None)] if x["name"] == "above_object"]
        released = [x for x in s if x["name"] == "released"]
        out = []
        for x, name in ((above[-1] if above else None, "above_object"), (grasp_open[-1] if grasp_open else None, "grasp"),
                        (closed[-1] if closed else None, "grasp"), (lifted, "lifted"),
                        (carried[-1] if carried else None, "above_goal"), (placed, "placed"),
                        (released[-1] if released else None, "released")):
            if x is not None:
                out.append({**x, "name": name})
        return out


def scene_in_base(ad, fixtures, block) -> list[dict]:
    """Fixtures as boxes in the arm's base frame (MoveIt collision objects); the parcel's size for reference."""
    d = ad.env.data
    root = ad.model.body(ad.cfg.root_body).id
    Rb, pb = d.xmat[root].reshape(3, 3), d.xpos[root]
    yaw = math.atan2(Rb[1, 0], Rb[0, 0])
    out = []
    for name, f in fixtures.items():
        c = (np.array(f.lo) + np.array(f.hi)) / 2
        size = np.array(f.hi) - np.array(f.lo)
        pc = Rb.T @ (c - pb)
        out.append({"id": name, "kind": f.kind, "center": [round(float(v), 4) for v in pc],
                    "size": [round(float(v), 4) for v in size], "yaw": round(-yaw, 5)})
    return out


def write_moveit_export(robot: str, goal_text: str, steps: list[dict], obstacles: list[dict], block, dest: Path,
                        hand_link: str) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    plan = {"robot": robot, "goal": goal_text, "frame": "the arm's base link (its root body in the robot file)",
            "hand_link": hand_link, "steps": steps, "obstacles": obstacles,
            "object": {"size": [2 * block.half_x, 2 * block.half_y, 2 * block.half_z], "mass": block.mass},
            "note": "recorded from a successful run in Morph's simulation"}
    (dest / "plan.json").write_text(json.dumps(plan, indent=1))
    (dest / "morph_pick_place.py").write_text(SCRIPT.replace("__ROBOT__", robot).replace("__HAND__", hand_link))
    (dest / "moveit_py.yaml").write_text(MOVEIT_PY_YAML)
    (dest / "morph_pick_place.launch.py").write_text(LAUNCH.replace("__ROBOT__", robot))
    joints = list(steps[0]["joints"]) if steps else []
    (dest / "README.md").write_text(f"""# Pick and place for {robot}, planned by Morph, run with MoveIt 2

Goal: {goal_text}. Morph found this sequence in simulation (plan.json): {', '.join(s['name'] for s in steps)}.

1. Edit the top of `morph_pick_place.py`: your MoveIt planning group names (arm, gripper), the gripper's
   named states, and `JOINT_NAMES` if your URDF names the joints differently from Morph's model
   ({', '.join(joints)}).
2. Put the object where Morph's scene had it, relative to the robot (plan.json has every pose in the base frame).
3. Run it with your robot's MoveIt config (it needs `robot_description`, the SRDF and kinematics):

```bash
ros2 launch ./morph_pick_place.launch.py moveit_config_pkg:=<your_robot>_moveit_config robot_name:=<your_robot>
```

The fixtures (walls, shelves, bins) are added to the planning scene as boxes, so MoveIt plans around them.
`USE_POSES = True` plans to the hand's poses instead of joint angles (needs your EE link = `{hand_link}`).

Morph checked the sequence in MuJoCo only; it has not been run through MoveIt here. Try it in MoveIt's
fake-hardware mode first (`use_fake_hardware:=true` in most MoveIt configs), with RViz open.
""")
    zp = shutil.make_archive(str(dest), "zip", dest)
    return Path(zp)


SCRIPT = r'''#!/usr/bin/env python3
"""Pick and place for __ROBOT__, found by Morph in simulation, executed with MoveIt 2 (moveit_py).

Reads plan.json next to this file: the arm's joint angles (and hand poses) at each step, all in the
arm's base frame, and the scene's fixtures as boxes.
"""
import json
import math
import sys
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive

# ---- edit these for your robot's MoveIt config -------------------------------------------------------
ARM_GROUP = "arm"             # planning group of the arm (e.g. "panda_arm", "ur_manipulator")
GRIPPER_GROUP = "gripper"     # planning group of the gripper (e.g. "hand"); "" if you drive it yourself
GRIPPER_STATES = {"open": "open", "closed": "close"}  # the gripper's named states in your SRDF
BASE_FRAME = "base_link"      # your arm's base link (plan.json poses are in it)
HAND_LINK = "__HAND__"        # the link whose pose plan.json records (only used with USE_POSES)
JOINT_NAMES = {}              # Morph's joint name -> yours, e.g. {"joint1": "panda_joint1"}
USE_POSES = False             # plan to the hand's poses instead of joint angles
SPEED = 0.3                   # velocity and acceleration scaling
# -------------------------------------------------------------------------------------------------------

PLAN = json.loads((Path(__file__).resolve().parent / "plan.json").read_text())


def execute(robot, component, name):
    component.set_start_state_to_current_state()
    result = component.plan()
    if not result:
        raise RuntimeError(f"MoveIt found no plan for '{name}'")
    try:
        robot.execute(result.trajectory, controllers=[])
    except TypeError:  # newer moveit_py: execute(group, trajectory, ...)
        robot.execute(component.planning_group_name if hasattr(component, "planning_group_name") else ARM_GROUP,
                      result.trajectory, blocking=True)


def add_fixtures(robot):
    with robot.get_planning_scene_monitor().read_write() as scene:
        for ob in PLAN["obstacles"]:
            co = CollisionObject()
            co.header.frame_id, co.id = BASE_FRAME, f"morph_{ob['id']}"
            box = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(v) for v in ob["size"]])
            p = Pose()
            p.position.x, p.position.y, p.position.z = (float(v) for v in ob["center"])
            p.orientation.z, p.orientation.w = math.sin(ob["yaw"] / 2), math.cos(ob["yaw"] / 2)
            co.primitives, co.primitive_poses, co.operation = [box], [p], CollisionObject.ADD
            scene.apply_collision_object(co)
        scene.current_state.update()


def main():
    rclpy.init()
    robot = MoveItPy(node_name="morph_pick_place")
    arm = robot.get_planning_component(ARM_GROUP)
    grip = robot.get_planning_component(GRIPPER_GROUP) if GRIPPER_GROUP else None
    add_fixtures(robot)
    model = robot.get_robot_model()
    gripper_now = None

    def set_gripper(want):
        nonlocal gripper_now
        if grip is not None and want != gripper_now:
            grip.set_goal_state(configuration_name=GRIPPER_STATES[want])
            execute(robot, grip, want)
            gripper_now = want
            time.sleep(0.3)

    set_gripper("open")
    for step in PLAN["steps"]:
        print(f"-> {step['name']}", flush=True)
        if step["gripper"] == "open":  # let go before moving away; grasp only once there
            set_gripper("open")
        if USE_POSES:
            ps = PoseStamped()
            ps.header.frame_id = BASE_FRAME
            (ps.pose.position.x, ps.pose.position.y, ps.pose.position.z) = step["hand_pose"]["position"]
            w, x, y, z = step["hand_pose"]["orientation_wxyz"]
            ps.pose.orientation.w, ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z = w, x, y, z
            arm.set_goal_state(pose_stamped_msg=ps, pose_link=HAND_LINK)
        else:
            state = RobotState(model)
            state.set_to_default_values()
            state.joint_positions = {JOINT_NAMES.get(k, k): v for k, v in step["joints"].items()}
            state.update()
            arm.set_goal_state(robot_state=state)
        execute(robot, arm, step["name"])
        set_gripper(step["gripper"])
    print("done", flush=True)
    rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
'''

MOVEIT_PY_YAML = """planning_scene_monitor_options:
  name: "planning_scene_monitor"
  robot_description: "robot_description"
  joint_state_topic: "/joint_states"
  attached_collision_object_topic: "/moveit_cpp/planning_scene_monitor"
  publish_planning_scene_topic: "/moveit_cpp/publish_planning_scene"
  monitored_planning_scene_topic: "/moveit_cpp/monitored_planning_scene"
  wait_for_initial_state_timeout: 10.0

planning_pipelines:
  pipeline_names: ["ompl"]

plan_request_params:
  planning_attempts: 3
  planning_pipeline: ompl
  max_velocity_scaling_factor: 0.3
  max_acceleration_scaling_factor: 0.3
"""

LAUNCH = r'''"""ros2 launch ./morph_pick_place.launch.py moveit_config_pkg:=<pkg> robot_name:=<name>"""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = Path(__file__).resolve().parent


def nodes(context):
    from moveit_configs_utils import MoveItConfigsBuilder
    cfg = (MoveItConfigsBuilder(LaunchConfiguration("robot_name").perform(context),
                                package_name=LaunchConfiguration("moveit_config_pkg").perform(context))
           .moveit_cpp(file_path=str(HERE / "moveit_py.yaml")).to_moveit_configs())
    return [Node(executable="python3", arguments=[str(HERE / "morph_pick_place.py")], name="morph_pick_place",
                 output="screen", parameters=[cfg.to_dict()])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("moveit_config_pkg", description="your robot's MoveIt config package"),
        DeclareLaunchArgument("robot_name", default_value="__ROBOT__"),
        OpaqueFunction(function=nodes),
    ])
'''
