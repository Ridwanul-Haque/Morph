"""Learn this task: for a robot with no ready-made skill for the goal, train one from scratch.

  make_task      project + goal + robot -> runs/learned/<robot>__<goal>/task.json (see task_model.py)
  train          scripts/train_task.py on the local GPU (or a prepared cloud job, like gait training)
  try_policy     run the learned policy on the CPU and check the real goal conditions
  export         a folder to take away: the policy (NumPy weights), what it expects and outputs
                 (policy.json), the exact model it was trained on, a standalone runner, a README
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import mujoco
import numpy as np

from .goal_terms import History, fixtures_geometry
from .goals import GoalSpec, check, lets_go, placement, release_spot
from .introspect import home_qpos, profile_robot
from .project import Project, _free_spot, _params, _start_poses
from .robot_import import import_robot
from .sim import SimulationEnvironment, SimulationUnstable
from .skill_steps import StepTracker, plan_steps
from .task_model import (CTRL_DT, MAX_STEP, build_task_model, contact_held, indices, load_task, measure, next_ctrl,
                         observe, write_task)

ROOT = Path(__file__).resolve().parents[1]
LEARNED = ROOT / "runs" / "learned"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40]


def make_task(pj: Project, goal: GoalSpec, robot_path: str, llm=None, log=print, plan: bool = True,
              stand_at: tuple | None = None) -> Path:
    """The task file for learning this goal with this robot, including the steps to go about it
    (skill_steps: Nemotron's when llm is given, else Morph's own; plan=False: none). stand_at: (x, y,
    heading) where the user put the robot, used instead of Morph's choice."""
    xml, _ = import_robot(robot_path)
    prof = profile_robot(xml)
    p = placement(goal, pj.scene)
    floating = prof.base != "fixed"
    sx, sy, syaw = pj.scene.start
    if floating:
        mx, my, myaw = _free_spot(pj, (sx, sy), (p["x"], p["y"]), dist=0.3 + 0.6 * max(prof.size_m.get("length", 0.5), 0.3))
    else:
        reach = max((v["max_horizontal"] for v in prof.reach_m.values()), default=0.6)
        cx, cy = (sx + p["x"]) / 2, (sy + p["y"]) / 2
        mx, my = _behind(pj, (sx, sy), (p["x"], p["y"]), (cx, cy), 0.55 * reach)
        spot = release_spot(goal, pj.scene)
        if spot is not None and math.hypot(spot[0] - sx, spot[1] - sy) > 0.6 * reach:
            # throwing from somewhere: stand between where it's picked up and where it's let go, a little aside
            cx, cy = (sx + spot[0]) / 2, (sy + spot[1]) / 2
            dx, dy = spot[0] - sx, spot[1] - sy
            n = math.hypot(dx, dy)
            mx, my = cx - 0.25 * dy / n, cy + 0.25 * dx / n
    m = mujoco.MjModel.from_xml_path(str(xml))
    q = home_qpos(m)
    if not floating:
        # turn the base so the arm's home pose points at the work area (the Menagerie UR5e's points sideways)
        d = mujoco.MjData(m)
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        root = [b_ for b_ in range(1, m.nbody) if m.body_parentid[b_] == 0][0]
        far = max(range(1, m.nbody), key=lambda b_: np.linalg.norm(d.xpos[b_][:2] - d.xpos[root][:2]))
        v = d.xpos[far][:2] - d.xpos[root][:2]
        myaw = math.atan2(cy - my, cx - mx) - (math.atan2(v[1], v[0]) if np.linalg.norm(v) > 0.05 else 0.0)
    if stand_at is not None:
        mx, my, myaw = (float(v) for v in stand_at)
    q0 = [0.0] * m.nu  # measured on the task model by write_task
    home = {m.joint(j).name: float(q[m.jnt_qposadr[j]]) for j in range(m.njnt)
            if int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))}
    b = pj.scene.block
    target = {"x": p["x"], "y": p["y"], "yaw": p["yaw"], "z": p["support_z"] + b.half_z, "pos_tol": p["pos_tol"],
              "yaw_tol": p["yaw_tol"], "heading": any(c.kind == "heading" for c in goal.conditions),
              "upright": any(c.kind == "upright" for c in goal.conditions), "lift": bool(p["needs_lift"]),
              "has_place": bool(p.get("has_place", True)), "lets_go": lets_go(goal)}
    out = task_dir(prof.name, goal)
    task = write_task(out / "task.json", robot_xml=str(xml), block=b, start=pj.scene.start, target=target,
                      mount=(mx, my, myaw), floating=floating, q0=q0, conditions=[c.model_dump() for c in goal.conditions],
                      scene_path=pj.scene.path, robot_name=prof.name, home=home,
                      fixtures=fixtures_geometry(pj.scene))
    if floating:  # standing height from the model's keyframe
        fj = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
        task["base_z"] = float(q[m.jnt_qposadr[fj] + 2]) or 0.4
    if plan:
        steps, info = plan_steps(prof, goal, pj.scene, p, llm, log)
        task["steps"] = [st.model_dump() for st in steps[:-1]]  # the last step is the goal itself
        task["steps_final"] = steps[-1].model_dump()
        task["steps_from"] = info["from"]
        if info.get("rationale"):
            task["steps_rationale"] = info["rationale"]
    (out / "task.json").write_text(json.dumps(task, indent=1))
    return out / "task.json"


def _behind(pj: Project, start, target, centre, dist: float) -> tuple[float, float]:
    """Where to bolt an arm down: behind the object, opposite the way it has to go (so it pushes rather
    than pulls), turned aside if that spot is inside a fixture. West of the work area if it needn't go anywhere."""
    d = np.array(target) - np.array(start)
    base = math.atan2(-d[1], -d[0]) if np.linalg.norm(d) > 0.03 else math.pi
    for k in range(9):
        a = base + (k + 1) // 2 * (0.5 if k % 2 else -0.5)
        x, y = centre[0] + dist * math.cos(a), centre[1] + dist * math.sin(a)
        if all(not (f.lo[0] - 0.12 < x < f.hi[0] + 0.12 and f.lo[1] - 0.12 < y < f.hi[1] + 0.12) for f in pj.scene.fixtures.values()):
            return float(x), float(y)
    return float(centre[0] + dist * math.cos(base)), float(centre[1] + dist * math.sin(base))


def task_dir(robot_name: str, goal: GoalSpec) -> Path:
    return LEARNED / f"{robot_name}__{slug(' '.join(c.describe() for c in goal.conditions if c.kind != 'upright')) or 'goal'}"


def learned_for(robot_name: str, goal: GoalSpec) -> Path | None:
    """task.json of a finished learned skill for this robot and goal, if there is one."""
    d = task_dir(robot_name, goal)
    return d / "task.json" if (d / "policy.npz").exists() and (d / "task.json").exists() else None


class NumpyPolicy:
    def __init__(self, path: Path):
        z = np.load(path)
        self.mean, self.std = z["obs_mean"], z["obs_std"]
        n = len([k for k in z.files if k.startswith("W")])
        self.W = [z[f"W{i}"] for i in range(n)]
        self.b = [z[f"b{i}"] for i in range(n)]
        self.act_dim = self.W[-1].shape[1] // 2

    def __call__(self, obs):
        x = (np.asarray(obs, float) - self.mean) / self.std
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = x / (1.0 + np.exp(-x))  # swish
        return np.tanh(x[: self.act_dim])


def rollout(env, m, ix, task, pol, start, params, seed, stop=None, watch=()) -> bool:
    """One episode of a policy from a start pose. Returns True if the simulation blew up.
    watch: things that follow the run every control tick (skill_steps.StepTracker, goal_terms.History),
    each given the measured quantities with "held" from the real contacts."""
    t = task["target"]
    tgt = (t["x"], t["y"], t["z"], t["yaw"])
    ctrl0 = np.clip(ix.q0, ix.act_lo, ix.act_hi)
    env.reset(seed, start, params)
    d = env.data
    d.qpos[ix.home_adr] = ix.home_val
    if ix.base >= 0:
        mx, my, myaw = task["mount"]
        d.qpos[ix.base_q:ix.base_q + 7] = [mx, my, task.get("base_z", 0.4), math.cos(myaw / 2), 0, 0, math.sin(myaw / 2)]
    d.ctrl[:] = ctrl0
    env.set_goal_marker(t["x"], t["y"], t["yaw"])
    env.track_extra = (lambda: d.xpos[ix.base][:2]) if ix.base >= 0 else None
    mujoco.mj_forward(m, d)
    last, ctrl = np.zeros(m.nu), ctrl0.copy()
    try:
        for _ in range(int(task["episode_s"] / CTRL_DT)):
            if stop is not None and stop():
                break
            a = pol(observe(np, d.qvel, d.actuator_length, d.actuator_velocity, d.xpos, d.xmat, last, ix, tgt))
            last = a
            ctrl = next_ctrl(np, ctrl, a, ctrl0, task["action_scale"], ix.act_lo, ix.act_hi)
            env.advance(ctrl)
            if watch:
                v = measure(np, d.xpos, d.xmat, d.qvel, ix, task, held=contact_held(m, d, ix))
                for w in watch:
                    w.update(v)
    except SimulationUnstable:
        return True
    return False


def practice_run(task_path: Path, policy_path: Path, seed: int = 0, stop=None, on_step=None) -> dict:
    """A training checkpoint, played once (for watching it learn). Returns how close it got and how far
    through the steps; on_step(k, name) is called as each step is reached."""
    task = load_task(task_path)
    m = build_task_model(task, accurate=True)
    ix = indices(m, task)
    env = SimulationEnvironment(m, control_dt=CTRL_DT)
    from .sim import PhysicalParams
    tr = StepTracker(task.get("steps", []), task.get("fixtures", {}), on_step)
    blew = rollout(env, m, ix, task, NumpyPolicy(policy_path), task["start"], PhysicalParams(), seed, stop, (tr,))
    p = env.block_pose()
    t = task["target"]
    return {"dist_cm": round(math.hypot(p.x - t["x"], p.y - t["y"]) * 100, 1), "blew_up": blew,
            "tilt_deg": round(math.degrees(p.tilt)), "steps_reached": tr.reached, "steps_total": len(tr.steps),
            "stuck_at": tr.steps[tr.reached][0] if tr.reached < len(tr.steps) else ""}


def try_policy(pj: Project, goal: GoalSpec, task_path: Path, n: int = 5, log=print, emit=None) -> dict:
    task = load_task(task_path)
    folder = task_path.parent
    m = build_task_model(task, accurate=True)
    ix = indices(m, task)
    pol = NumpyPolicy(folder / "policy.npz")
    env = SimulationEnvironment(m, control_dt=CTRL_DT)
    trials = []
    for i, start in enumerate(_start_poses(pj.scene, n)):
        if emit:
            emit(type="trial_start", robot=task["robot"], i=i, n=n)
        tr = StepTracker(task.get("steps", []), task.get("fixtures", {}))
        hist = History([c.model_dump() for c in goal.conditions], task.get("fixtures", {}),
                       (task["block"]["half_x"], task["block"]["half_y"], task["block"]["half_z"]))
        blew_up = rollout(env, m, ix, task, pol, start, _params(i), i, watch=(tr, hist))
        d = env.data
        held = contact_held(m, d, ix) > 0.5
        res = check(goal, pj.scene, m, d, env.block_pose(), env.block_speed(), not held, history=hist)
        if blew_up:
            res["success"] = False
            res["reasons"].insert(0, "the simulation became unstable (the robot hit the object too hard)")
        trials.append({"success": res["success"], "conditions": res["results"], "reasons": res["reasons"],
                       "steps": tr.summary()})
        if emit:
            emit(type="trial", robot=task["robot"], i=i, n=n, success=res["success"], conditions=res["results"],
                 reasons=res["reasons"])
        log(f"  learned policy, try {i + 1}: {'success' if res['success'] else 'fail: ' + '; '.join(res['reasons'][:2])}"
            + (f" ({tr.summary()})" if tr.steps and not res["success"] else ""))
    ok = sum(t["success"] for t in trials)
    return {"success": ok, "n": len(trials), "trials": trials}


RUNNER = '''"""Run the policy Morph learned, in MuJoCo (numpy + mujoco only).

    pip install mujoco numpy
    python run_policy.py            # runs one episode headless and prints where the object ended up
    python run_policy.py --view     # watch it (needs a display)

policy.json says exactly what the policy reads and outputs, so you can feed it from your own
simulator or robot: build the observation vector in that order, scale the output by action_scale,
add it to default_pose, and send the result to the joints as position targets at 1/control_dt Hz.
"""
import json, math, sys
import numpy as np, mujoco

spec = json.load(open("policy.json"))
m = mujoco.MjModel.from_xml_path("model.xml")
d = mujoco.MjData(m)
z = np.load("policy.npz")
W = [z[f"W{i}"] for i in range(len([k for k in z.files if k.startswith("W")]))]
B = [z[f"b{i}"] for i in range(len(W))]

def policy(obs):
    x = (obs - z["obs_mean"]) / z["obs_std"]
    for i, (w, b) in enumerate(zip(W, B)):
        x = x @ w + b
        if i < len(W) - 1:
            x = x / (1 + np.exp(-x))
    return np.tanh(x[: W[-1].shape[1] // 2])

ix = spec["indices"]
tips, mask = np.array(ix["tips"]), np.array(ix["pos_mask"])
q0, lo, hi = np.array(spec["default_pose"]), np.array(ix["act_lo"]), np.array(ix["act_hi"])
tx, ty, tz, tyaw = spec["target"]

def observe(last):
    bp = d.xpos[ix["block"]]; R = d.xmat[ix["block"]].reshape(3, 3); yaw = math.atan2(R[1, 0], R[0, 0])
    parts = [(d.actuator_length - q0) * mask, d.actuator_velocity * 0.1, bp - [tx, ty, tz], R[:, 2], [math.cos(yaw - tyaw), math.sin(yaw - tyaw)],
             d.qvel[ix["block_v"]:ix["block_v"] + 3] * 0.5, (d.xpos[tips] - bp).ravel(), last]
    if ix["base"] >= 0:
        Rb = d.xmat[ix["base"]].reshape(3, 3); v = ix["base_v"]
        parts += [d.xpos[ix["base"]] - bp, Rb.T @ [0, 0, -1.0], Rb.T @ d.qvel[v:v + 3] * 0.5, d.qvel[v + 3:v + 6] * 0.2]
    return np.concatenate(parts)

d.qpos[:] = spec["initial_qpos"]; d.ctrl[:] = np.clip(q0, lo, hi); mujoco.mj_forward(m, d)
steps = round(spec["control_dt"] / m.opt.timestep)
last, ctrl = np.zeros(m.nu), np.clip(q0, lo, hi)
viewer = None
if "--view" in sys.argv:
    import mujoco.viewer; viewer = mujoco.viewer.launch_passive(m, d)
for _ in range(int(spec["episode_s"] / spec["control_dt"])):
    last = policy(observe(last))
    want = np.clip(q0 + last * spec["action_scale"], lo, hi)
    ctrl = ctrl + np.clip(want - ctrl, -spec["max_step"], spec["max_step"])
    d.ctrl[:] = ctrl
    for _ in range(steps):
        mujoco.mj_step(m, d)
    if viewer: viewer.sync()
bp = d.xpos[ix["block"]]
print(f"object ended at ({bp[0]:.3f}, {bp[1]:.3f}, {bp[2]:.3f}); target ({tx:.3f}, {ty:.3f}, {tz:.3f})")
'''


def export(task_path: Path, dest: Path | None = None) -> Path:
    """A folder (and .zip) someone can take away and use without Morph."""
    task = load_task(task_path)
    src = task_path.parent
    m = build_task_model(task, accurate=True)
    ix = indices(m, task)
    dest = dest or (src / "export")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    shutil.copy(src / "policy.npz", dest / "policy.npz")
    (dest / "model.xml").write_text(_portable_xml(task, dest / "assets"))
    d = mujoco.MjData(m)
    d.qpos[ix.home_adr] = ix.home_val
    bx, by, byaw = task["start"]
    d.qpos[ix.block_q:ix.block_q + 7] = [bx, by, task["block"]["half_z"], math.cos(byaw / 2), 0, 0, math.sin(byaw / 2)]
    if ix.base >= 0:
        mx, my, myaw = task["mount"]
        d.qpos[ix.base_q:ix.base_q + 7] = [mx, my, task.get("base_z", 0.4), math.cos(myaw / 2), 0, 0, math.sin(myaw / 2)]
    t = task["target"]
    meta = json.loads((src / "train_meta.json").read_text()) if (src / "train_meta.json").exists() else {}
    policy_json = {
        "robot": task["robot"], "goal": task["conditions"], "target": [t["x"], t["y"], t["z"], t["yaw"]],
        "control_dt": CTRL_DT, "episode_s": task["episode_s"], "action_scale": task["action_scale"],
        "default_pose": list(map(float, ix.q0)),
        "joints": list(ix.joints),
        "commands": [_command_name(m, a) for a in range(m.nu)],
        "action": "one command per actuator (commands above): move towards default_pose + action_scale * output (clipped "
                  "to limits) by at most max_step per control step; a position target for position actuators "
                  "(pos_mask 1), else a speed",
        "max_step": MAX_STEP,
        "observation": ["(actuator length - default_pose) x pos_mask (n): joint angles, or tendon lengths",
                        "actuator speeds x 0.1 (n)", "object position - target (3)",
                        "object up axis (3)", "cos, sin of (object heading - target heading) (2)", "object velocity x 0.5 (3)",
                        f"end bodies relative to object ({len(ix.tips)} x 3: " +
                        ", ".join(m.body(int(b)).name.removeprefix("robot_") for b in ix.tips) + ")",
                        "last action (n)"] + (["base - object (3)", "gravity in base frame (3)", "base velocity in base frame x 0.5 (3)",
                                               "base angular velocity x 0.2 (3)"] if ix.base >= 0 else []),
        "network": "normalise with obs_mean/obs_std, 3 hidden layers with swish, tanh on the first half of the output",
        "indices": {"joint_qadr": ix.home_adr.tolist(), "joint_vadr": ix.joint_vadr.tolist(), "pos_mask": ix.pos_mask.tolist(),
                    "tips": ix.tips.tolist(), "block": ix.block,
                    "block_v": ix.block_v, "base": ix.base, "base_v": ix.base_v, "act_lo": ix.act_lo.tolist(),
                    "act_hi": ix.act_hi.tolist()},
        "initial_qpos": d.qpos.tolist(), "training": {k: meta.get(k) for k in ("steps", "wall_s")},
        "mount": list(task["mount"]), "randomize": meta.get("randomize", "off"),
    }
    (dest / "policy.json").write_text(json.dumps(policy_json, indent=1))
    (dest / "run_policy.py").write_text(RUNNER)
    goal_text = "; ".join(c.get("kind", "").replace("_", " ") + (f" {c['target']}" if c.get("target") else "")
                          for c in task["conditions"])
    from .ros_export import write_ros2_package
    ros = write_ros2_package(dest, policy_json, goal_text)
    (dest / "README.txt").write_text(
        f"Policy learned by Morph for {task['robot']}.\n\nGoal: {goal_text}" +
        "\n\nFiles:\n  policy.npz     the trained network (NumPy arrays)\n  policy.json    what it reads, what it outputs, joint order, rates\n"
        "  model.xml      the exact MuJoCo scene + robot it was trained in\n  run_policy.py  run it: python run_policy.py [--view]\n"
        f"  ros2/{ros.name}/  a ROS 2 package: a node that runs it on your robot (joint_states in, controller commands\n"
        "                 out) plus a simulated stand-in to try it first; see its README.md\n\n"
        "It was trained in simulation. On a real robot, expect to fine-tune: real motors, friction and sensing differ.\n")
    zp = shutil.make_archive(str(dest), "zip", dest)
    return Path(zp)


def _command_name(m: mujoco.MjModel, a: int) -> str:
    """What a controller would call this actuator's command: its joint's name, else the actuator's."""
    if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
        return m.joint(int(m.actuator_trnid[a, 0])).name.removeprefix("robot_")
    return m.actuator(a).name.removeprefix("robot_")


def _portable_xml(task: dict, assets: Path) -> str:
    """The training scene as one MJCF plus an assets/ folder holding every mesh and texture it uses."""
    from .task_model import build_task_spec
    robot = mujoco.MjSpec.from_file(task["robot_xml"])
    base = Path(task["robot_xml"]).parent
    meshdir = base / (robot.meshdir or "")
    texdir = base / (robot.texturedir or "")
    spec = build_task_spec(task, accurate=True)
    assets.mkdir(parents=True, exist_ok=True)
    used = {}
    for kind, items, folder in (("mesh", spec.meshes, meshdir), ("texture", spec.textures, texdir)):
        for it in items:
            if not it.file:
                continue
            src = Path(it.file) if Path(it.file).is_absolute() else folder / it.file
            if not src.exists():
                src = base / it.file
            name = src.name if src.name not in used or used[src.name] == src else f"{kind}_{len(used)}_{src.name}"
            used[name] = src
            shutil.copy(src, assets / name)
            it.file = name
    spec.meshdir = spec.texturedir = "assets"
    return spec.to_xml()
