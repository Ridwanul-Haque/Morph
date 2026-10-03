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
    gripper = None
    if prof.has_gripper:  # which motor is the gripper, and which end of its range is open / closed
        try:
            from .robots.auto_arm import auto_arm_config
            gcfg, _ = auto_arm_config(xml)
            gripper = {"actuator": gcfg.gripper_actuator, "open": gcfg.ctrl_open, "close": gcfg.ctrl_close,
                       "gain_scale": float(getattr(gcfg, "gripper_gain_scale", 1.0))}
        except Exception:  # noqa: BLE001  (not an arm Morph can set up: no closure measure, the rest still works)
            gripper = None
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
    if gripper:
        task["gripper"] = gripper
    if not floating:  # like the arm the programs run on (and real arm controllers): gravity compensated
        task["gravcomp"] = True
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


def rollout(env, m, ix, task, pol, start, params, seed, stop=None, watch=(), settle_s: float = 1.0) -> bool:
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
            ctrl = next_ctrl(np, ctrl, a, ctrl0, ix.act_scale, ix.act_lo, ix.act_hi, ix.act_step)
            env.advance(ctrl)
            if watch:
                v = measure(np, d.xpos, d.xmat, d.qvel, ix, task, held=contact_held(m, d, ix), ctrl=d.ctrl)
                for w in watch:
                    w.update(v)
                    if hasattr(w, "see"):
                        w.see(d)
        for _ in range(int(settle_s / CTRL_DT)):  # the checklist is judged once things settle: hold still a moment
            env.advance(ctrl)
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
    want = np.clip(q0 + last * np.asarray(spec["action_scale"]), lo, hi)
    ctrl = ctrl + np.clip(want - ctrl, -np.asarray(spec["max_step"]), np.asarray(spec["max_step"]))
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
        "control_dt": CTRL_DT, "episode_s": task["episode_s"], "action_scale": ix.act_scale.tolist(),
        "default_pose": list(map(float, ix.q0)),
        "joints": list(ix.joints),
        "commands": [_command_name(m, a) for a in range(m.nu)],
        "action": "one command per actuator (commands above): move towards default_pose + action_scale * output (both "
                  "per actuator; clipped to limits) by at most max_step (per actuator) per control step; a position target for position actuators "
                  "(pos_mask 1), else a speed",
        "max_step": ix.act_step.tolist(),
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


# ------------------------------------------------------------------ learning in rounds, with Nemotron revising the plan
class _Watch:
    """Follows a replay of the policy: how far through the steps it gets, and what it actually does."""

    def __init__(self, task: dict):
        from .goal_terms import uses
        self.task = task
        self.steps = task.get("steps", [])
        self.tracker = StepTracker(self.steps, task.get("fixtures", {}))
        self.vars = [sorted(uses(s["done_when"]) - {"abs", "min", "max", "sqrt", "hypot", "dist_to"}) for s in self.steps]
        self.best = [None] * len(self.steps)  # closest each step came (graded), and the values seen meanwhile
        self.seen: dict[int, dict] = {}
        self.ext = {"touch": 9.0, "grip": 0.0, "lifted": 0.0, "moved": 0.0, "speed": 0.0, "tilt": 0.0, "held_s": 0.0}
        self.frontier: dict[int, np.ndarray] = {}  # the state it was in when it came closest to each step
        self._improved = None

    def update(self, v: dict) -> None:
        from .goal_terms import parse, violation
        from .task_model import soft
        self.tracker.update(v)
        k = self.tracker.reached
        if k < len(self.steps):
            x = float(violation(np, parse(self.steps[k]["done_when"], self.task.get("fixtures", {})), soft(v)))
            if self.best[k] is None or x < self.best[k]:
                self.best[k] = x
                self._improved = k
            rng = self.seen.setdefault(k, {})
            for name in self.vars[k]:
                if name in v and not isinstance(v[name], dict):
                    val = float(v[name])
                    lo, hi = rng.get(name, (val, val))
                    rng[name] = (min(lo, val), max(hi, val))
        e = self.ext
        e["touch"] = min(e["touch"], float(v.get("touch_dist", 9.0)))
        e["grip"], e["lifted"] = max(e["grip"], float(v.get("grip", 0.0))), max(e["lifted"], float(v.get("lifted", 0.0)))
        e["moved"], e["speed"] = max(e["moved"], float(v["travelled"])), max(e["speed"], float(v["speed"]))
        e["tilt"] = max(e["tilt"], float(v.get("base_tilt_deg", 0.0)))
        e["held_s"] += CTRL_DT * (float(v["held"]) > 0.5)


    def see(self, d) -> None:
        if self._improved is not None:  # keep the state of its best moment so far on that step
            self.frontier[self._improved] = d.qpos.copy()
            self._improved = None


def diagnose(task_path: Path, policy_path: Path, n: int = 4) -> dict:
    """What the learned policy actually does, in numbers a person (or Nemotron) can act on."""
    from collections import Counter
    from .sim import PhysicalParams
    task = load_task(task_path)
    m = build_task_model(task, accurate=True)
    ix = indices(m, task)
    env = SimulationEnvironment(m, control_dt=CTRL_DT)
    pol = NumpyPolicy(policy_path)
    watches = []
    for seed in range(n):
        w = _Watch(task)
        rollout(env, m, ix, task, pol, task["start"], PhysicalParams(), seed, watch=(w,))
        watches.append(w)
    steps = task.get("steps", [])
    reached = [w.tracker.reached for w in watches]
    stuck = Counter(r for r in reached if r < len(steps)).most_common(1)
    out = {"replays": n, "steps_reached_each_replay": reached, "steps": [s["name"] for s in steps]}
    if stuck:
        k = stuck[0][0]
        bests = [w.best[k] for w in watches if w.best[k] is not None]
        ranges = {}
        for w in watches:
            for name, (lo, hi) in w.seen.get(k, {}).items():
                a, b = ranges.get(name, (lo, hi))
                ranges[name] = (round(min(a, lo), 3), round(max(b, hi), 3))
        out["stuck_at"] = {"step": k + 1, "name": steps[k]["name"], "done_when": steps[k]["done_when"],
                           "closest_it_came (0 = done; graded)": round(min(bests), 3) if bests else None,
                           "values_seen_while_trying (min, max)": ranges}
    if stuck:  # where it got closest on the step it's stuck at: the next round practises from there too
        k = stuck[0][0]
        starts = [w.frontier[k] for w in watches if k in w.frontier]
        if starts:
            f = task_path.parent / "starts.npz"
            qs, cs = np.array(starts), np.full((len(starts), m.nu), np.nan)  # no commands: hold the pose it's in
            if f.exists() and task.get("demo"):  # keep the program's moments too
                old = np.load(f)
                qs = np.concatenate([old["qpos"], qs])
                cs = np.concatenate([old["ctrl"] if "ctrl" in old else np.full((len(old["qpos"]), m.nu), np.nan), cs])
            np.savez(f, qpos=qs, ctrl=cs)
            task["starts"] = {"file": "starts.npz", "share": 0.5 if task.get("demo") else 0.3,
                              "from": f"its closest moments on step {k + 1}" + (" and the program's" if task.get("demo") else "")}
            task_path.write_text(json.dumps(task, indent=1))
            out["practice_from"] = f"{len(starts)} of its closest moments on step {k + 1}, for 30% of practice runs"
    e = {key: [w.ext[key] for w in watches] for key in watches[0].ext}
    out["what_the_robot_did"] = {
        "closest any of its end parts got to the object's surface (cm)": round(min(e["touch"]) * 100, 1),
        "best grip score (0..1; 1 = held from opposite sides and lifted)": round(max(e["grip"]), 2),
        "seconds it really held the object (both sides, nothing else touching)": round(max(e["held_s"]), 1),
        "highest it lifted the object (cm)": round(max(e["lifted"]) * 100, 1),
        "furthest it moved the object (cm)": round(max(e["moved"]) * 100, 1),
        "fastest the object moved (m/s)": round(max(e["speed"]), 2)}
    if task.get("floating"):
        out["what_the_robot_did"]["most its body tilted (degrees)"] = round(max(e["tilt"]), 0)
    return out


def revise_task_steps(task_path: Path, llm, diagnosis: dict, goal: GoalSpec, scene, rounds: list, log=print) -> dict | None:
    """Nemotron reads what the learner actually did and rewrites the steps (smaller first steps, reachable
    thresholds, graded quantities); Morph checks the new steps and writes them into the task."""
    from .introspect import profile_robot
    from .skill_steps import Step, check_steps, robot_brief, scales
    task = load_task(task_path)
    prof = profile_robot(task["robot_xml"])
    p = placement(goal, scene)
    errors, fix = [], None
    for _ in range(2):
        prop, rec = llm.revise_steps(robot_brief(prof), [c.describe() for c in goal.conditions], task.get("steps", []),
                                     diagnosis, scales(scene, p), rounds, fix)
        raw = [Step(**s) for s in prop.steps if isinstance(s, dict) and s.get("done_when")]
        new, errors = check_steps(raw, list(scene.fixtures), scene.object_name)
        if new and not errors:
            task["steps"] = [s.model_dump() for s in new]
            task["steps_from"] = f"nemotron, revised after round {len(rounds)}"
            task_path.write_text(json.dumps(task, indent=1))
            return {"steps": task["steps"], "rationale": prop.rationale}
        fix = {"your_previous_steps": [s.model_dump() for s in raw], "problems": errors}
    log(f"  (Nemotron's revised steps couldn't be measured: {'; '.join(errors[:2])}; keeping the old ones)")
    return None


def learn_in_rounds(pj: Project, goal: GoalSpec, task_path: Path, train, llm=None, log=print, emit=None,
                    total_steps: int = 60_000_000, rounds: int = 4, tries: int = 5, candidates: int = 3) -> dict:
    """Practise in short rounds. After each: try it; if it isn't there yet, measure what it actually does and
    (with Nemotron) revise the steps, then keep practising from where it got to. candidates > 0: instead of one
    revision, Nemotron proposes that many different plans and the best after brief practice is kept (Eureka-style).
    A round that did better than the one before keeps its plan: the plan only changes when progress stalls.
    train(task_path, steps, restore) -> bool runs the GPU trainer."""
    emit = emit or (lambda **e: None)
    per = max(5_000_000, total_steps // rounds)
    history, res, best, last_fit = [], {"success": 0, "n": 0}, None, None
    task = load_task(task_path)
    if task.get("demo"):  # copied from a working program: maybe that's already enough
        emit(type="learn_round", round=0, rounds=rounds, stage="practice", steps=0)
        if train(task_path, 0, None):
            res = try_policy(pj, goal, task_path, n=tries, log=log, emit=emit)
            log(f"  copying the program: {res['success']}/{res['n']}")
            emit(type="learn_round", round=0, rounds=rounds, stage="tried", success=res["success"], n=res["n"],
                 note="copied from Nemotron's program, before any practice")
            if res["success"] >= math.ceil(0.6 * tries):
                res["learned_by"] = "copying the program"
                res["rounds"] = history
                return res
    for r in range(1, rounds + 1):
        restore = task_path.parent / "params.pkl" if (r > 1 or task.get("demo")) else None
        emit(type="learn_round", round=r, rounds=rounds, stage="practice", steps=per)
        if not train(task_path, per, restore):
            res["why_not"] = "the trainer stopped (see the log)"
            break
        res = try_policy(pj, goal, task_path, n=tries, log=log, emit=emit)
        emit(type="learn_round", round=r, rounds=rounds, stage="tried", success=res["success"], n=res["n"])
        best = _keep_best(task_path, res, best, f"round {r}")
        if res["success"] >= math.ceil(0.6 * tries) or r == rounds:
            break
        fit = _fitness(task_path.parent, res)
        improving = fit > last_fit if last_fit is not None else fit[1] > 0  # first round: any time in the goal state
        if improving:  # still getting better: keep practising the same plan
            log(f"  round {r}: {res['success']}/{res['n']}, better than the round before: same plan, more practice")
            history.append({"round": r, "practice_steps": per * r, "tries_succeeded": f"{res['success']}/{res['n']}",
                            "improving": True})
            last_fit = fit
            continue
        last_fit = fit
        diag = diagnose(task_path, task_path.parent / "policy.npz")
        history.append({"round": r, "practice_steps": per * r, "tries_succeeded": f"{res['success']}/{res['n']}",
                        "diagnosis": diag})
        stuck = diag.get("stuck_at")
        log(f"  round {r}: {res['success']}/{res['n']}" + (f"; stuck at step {stuck['step']} ({stuck['name']})" if stuck else ""))
        emit(type="learn_round", round=r, rounds=rounds, stage="diagnosed", diagnosis=diag)
        if llm is not None and getattr(llm, "available", False) and candidates > 0:
            each = max(2_000_000, per // (candidates + 1))
            chosen = search_step_plans(pj, goal, task_path, train, llm, diag, history, each, tries, candidates, log,
                                       lambda **e: emit(**e, round=r, rounds=rounds))
            if chosen:
                history[-1]["revision"] = f"best of {chosen['tried']} plans: {chosen['idea']}"
                t = load_task(task_path)
                emit(type="skill_steps", robot=t["robot"], steps=t["steps"] + [t["steps_final"]], **{"from": t["steps_from"]},
                     rationale=chosen["idea"])
                best = _keep_best(task_path, chosen["result"], best, f"best plan after round {r}")
                if chosen["result"]["success"] >= math.ceil(0.6 * tries):  # the search itself got there
                    res = chosen["result"]
                    break
                continue
        if llm is not None and getattr(llm, "available", False):
            try:
                rev = revise_task_steps(task_path, llm, diag, goal, pj.scene, history, log)
            except Exception as e:  # noqa: BLE001  (no revision: keep practising with the same steps)
                rev = None
                log(f"  (Nemotron couldn't revise the steps: {e})")
            if rev:
                history[-1]["revision"] = rev["rationale"]
                log(f"  Nemotron revised the steps: {rev['rationale'][:300]}")
                t = load_task(task_path)
                emit(type="skill_steps", robot=t["robot"], steps=t["steps"] + [t["steps_final"]], **{"from": t["steps_from"]},
                     rationale=rev["rationale"])
                emit(type="learn_round", round=r, rounds=rounds, stage="revised", rationale=rev["rationale"])
    if best is not None and _fitness(task_path.parent, res) < best[0]:  # a later round did worse: go back
        for f, data in best[2].items():
            (task_path.parent / f).write_bytes(data)
        log(f"  kept the best result, from {best[3]}: {best[1]['success']}/{best[1]['n']}")
        res = best[1]
    res["rounds"] = history
    return res


# ------------------------------------------------------------------ learning from a working program
def demo_to_task(task_path: Path, ad, demos: list, start_every: int = 4, share: float = 0.5) -> dict:
    """A program's successful runs (recorded on the arm it ran on) in the learner's terms: the observation the
    policy would see at each moment and the action that gives the program's commands (for imitation), plus
    states along the way to start practice from (hand around it, gripping it, lifting it). The task's arm must
    stand where the program's arm stood (make_task stand_at)."""
    task = load_task(task_path)
    m = build_task_model(task)
    d = mujoco.MjData(m)
    src = ad.model
    jmap, amap = [], []
    for j in range(src.njnt):
        n = src.joint(j).name
        if n == "block_free":
            tj = m.joint("block_free").id
            jmap.append((src.jnt_qposadr[j], m.jnt_qposadr[tj], 7, src.jnt_dofadr[j], m.jnt_dofadr[tj], 6))
            continue
        try:
            tj = m.joint("robot_" + n).id
        except KeyError:
            continue
        jmap.append((src.jnt_qposadr[j], m.jnt_qposadr[tj], 1, src.jnt_dofadr[j], m.jnt_dofadr[tj], 1))
    for a in range(src.nu):
        try:
            amap.append((a, m.actuator("robot_" + src.actuator(a).name).id))
        except KeyError:
            continue
    # start where the demonstration started (the arm's own ready pose, hand open), and let "no action" mean
    # "keep the commands it started with": otherwise the copy meets a start it has never seen
    q_first, _, c_first = demos[0][0]
    for j in range(m.njnt):
        name = m.joint(j).name
        if name.startswith("robot_") and name.removeprefix("robot_") in task.get("home", {}):
            try:
                sj = src.joint(name.removeprefix("robot_")).id
            except KeyError:
                continue
            task["home"][name.removeprefix("robot_")] = float(q_first[src.jnt_qposadr[sj]])
    q0 = list(task["q0"])
    for sa, ta in amap:
        q0[ta] = float(c_first[sa])
    task["q0"] = q0
    task_path.write_text(json.dumps(task, indent=1))
    ix = indices(m, task)
    t = task["target"]
    tgt = (t["x"], t["y"], t["z"], t["yaw"])
    obs, acts, sq, sc = [], [], [], []
    for demo in demos:
        last = np.zeros(m.nu)
        for k, (qpos, qvel, ctrl) in enumerate(demo):
            d.qpos[:] = m.qpos0
            d.qvel[:] = 0
            for sa, ta, n, sv, tv, nv in jmap:
                d.qpos[ta:ta + n] = qpos[sa:sa + n]
                d.qvel[tv:tv + nv] = qvel[sv:sv + nv]
            mujoco.mj_forward(m, d)
            c = np.clip(ix.q0, ix.act_lo, ix.act_hi).copy()
            for sa, ta in amap:
                c[ta] = ctrl[sa]
            c = np.clip(c, ix.act_lo, ix.act_hi)
            a = np.clip((c - ix.q0) / ix.act_scale, -1.0, 1.0)
            obs.append(observe(np, d.qvel, d.actuator_length, d.actuator_velocity, d.xpos, d.xmat, last, ix, tgt))
            acts.append(a)
            last = a
            if k % start_every == 0:
                sq.append(d.qpos.copy())
                sc.append(c)
    np.savez(task_path.parent / "demo.npz", obs=np.array(obs), act=np.array(acts))
    np.savez(task_path.parent / "starts.npz", qpos=np.array(sq), ctrl=np.array(sc))
    task["demo"] = "demo.npz"
    task["starts"] = {"file": "starts.npz", "share": share, "from": "moments along the program that worked"}
    # enough time to do it at a learner's pace: the program's own duration, and then some
    longest = max(len(dm) for dm in demos) * CTRL_DT
    task["episode_s"] = round(max(task.get("episode_s", 8.0), 1.6 * longest), 1)
    if task.get("gripper"):  # the gripper strength the arm's grip test settled on (the program ran with it)
        task["gripper"]["gain_scale"] = float(getattr(ad.cfg, "gripper_gain_scale", 1.0))
    task_path.write_text(json.dumps(task, indent=1))
    return {"samples": len(obs), "starts": len(sq)}


def prepare_learning(pj: Project, goal: GoalSpec, robot_path: str, llm=None, log=print, emit=None, stand_at=None,
                     steps_from: Path | None = None) -> tuple[Path, dict]:
    """The task to learn, from a working program when there can be one: for an arm with a gripper, Nemotron writes a
    program (programs.solve), its successful runs are recorded, and the learner starts by copying them and
    practises from moments along them. Otherwise the task as usual. steps_from: a task.json whose steps were
    already planned (reused, no new planning)."""
    from .programs import solve
    emit = emit or (lambda **e: None)
    xml, _ = import_robot(robot_path)
    prof = profile_robot(xml)
    info: dict = {"demo": False}
    planned = load_task(steps_from) if steps_from and Path(steps_from).exists() else None
    if llm is not None and getattr(llm, "available", False) and prof.base == "fixed" and prof.has_gripper:
        emit(type="caption", text="Nemotron is writing a program to learn from")
        r = solve(pj, goal, robot_path, prof.name, llm, log=log, emit=emit, n=3, stand_at=stand_at)
        if r.get("demos"):
            ad = r["adapter"]
            mount = (ad.cfg.base_pos[0], ad.cfg.base_pos[1], getattr(ad.cfg, "base_yaw", 0.0))
            task = make_task(pj, goal, robot_path, llm=None if planned else llm, log=log, stand_at=mount,
                             plan=planned is None)
            if planned:  # the steps already shown
                t = load_task(task)
                for k in ("steps", "steps_final", "steps_from", "steps_rationale"):
                    if k in planned:
                        t[k] = planned[k]
                task.write_text(json.dumps(t, indent=1))
            d = demo_to_task(task, ad, r["demos"])
            log(f"  learning from Nemotron's program: {len(r['demos'])} successful runs, {d['samples']} moments to copy, "
                f"{d['starts']} to practise from")
            info = {"demo": True, "program": r["program"], "program_trials": f"{r['success']}/{r['n']}"}
            return task, info
        log("  (no working program to learn from: learning from scratch)")
    if planned:
        return Path(steps_from), info
    return make_task(pj, goal, robot_path, llm=llm, log=log, stand_at=stand_at), info


def _keep_best(task_path: Path, res: dict, best, label: str):
    """Remember the best policy so far (same yardstick as the plan search), so a later, worse round can't lose it."""
    fit = _fitness(task_path.parent, res)
    if best is None or fit > best[0]:
        files = {f: (task_path.parent / f).read_bytes() for f in ("task.json", "policy.npz", "params.pkl", "train_meta.json")
                 if (task_path.parent / f).exists()}
        return (fit, dict(res), files, label)
    return best


# ------------------------------------------------------------------ several step plans at once (Eureka-style)
def _fitness(folder: Path, res: dict) -> tuple:
    """One yardstick for every candidate, whatever its steps: real successes in the tries, then how much of the
    time the training evaluation spent in the goal state, then how close it got."""
    meta = json.loads((folder / "train_meta.json").read_text()) if (folder / "train_meta.json").exists() else {}
    last = (meta.get("curve") or [{}])[-1]
    in_goal = last.get("success", 0.0)
    in_goal = 0.0 if in_goal != in_goal else in_goal  # NaN
    dist = last.get("dist", 1e9)
    dist = 1e9 if dist != dist else dist
    return (res.get("success", 0), round(in_goal, 2), -round(dist, 2))


def search_step_plans(pj: Project, goal: GoalSpec, task_path: Path, train, llm, diagnosis: dict, rounds: list,
                      steps_each: int, tries: int = 5, k: int = 3, log=print, emit=None) -> dict | None:
    """Nemotron proposes k different step plans at once; each (and the current plan) practises briefly from where the
    learner is now, in its own folder; the best on the same yardstick becomes the plan and the policy. Returns what
    was chosen, or None if Nemotron couldn't propose any."""
    from .introspect import profile_robot
    from .skill_steps import Step, check_steps, robot_brief, scales
    emit = emit or (lambda **e: None)
    task = load_task(task_path)
    prof = profile_robot(task["robot_xml"])
    try:
        prop, _ = llm.step_candidates(robot_brief(prof), [c.describe() for c in goal.conditions], task.get("steps", []),
                                      diagnosis, scales(pj.scene, placement(goal, pj.scene)), rounds, k)
    except Exception as e:  # noqa: BLE001
        log(f"  (Nemotron couldn't propose step plans: {e})")
        return None
    plans = [{"idea": "keep the current steps", "steps": task.get("steps", [])}]
    for c in prop.candidates:
        raw = [Step(**s) for s in c.get("steps", []) if isinstance(s, dict) and s.get("done_when")]
        ok, errors = check_steps(raw, list(pj.scene.fixtures), pj.scene.object_name)
        if ok and not errors:
            plans.append({"idea": str(c.get("idea", ""))[:200], "steps": [s.model_dump() for s in ok]})
        else:
            log(f"  (a proposed plan couldn't be measured: {'; '.join(errors[:1])})")
    if len(plans) == 1:
        return None
    emit(type="learn_round", stage="candidates", ideas=[p["idea"] for p in plans])
    main = task_path.parent
    results = []
    for i, plan in enumerate(plans):
        cdir = main / f"candidate_{i}"
        shutil.rmtree(cdir, ignore_errors=True)
        cdir.mkdir()
        for f in ("demo.npz", "starts.npz"):  # the same demonstration and practice starts
            if (main / f).exists():
                shutil.copy(main / f, cdir / f)
        t = dict(task, steps=plan["steps"])
        (cdir / "task.json").write_text(json.dumps(t, indent=1))
        log(f"  candidate {i + 1}/{len(plans)}: {plan['idea']}")
        if not train(cdir / "task.json", steps_each, main / "params.pkl"):
            continue
        res = try_policy(pj, goal, cdir / "task.json", n=tries, log=lambda s: None)
        fit = _fitness(cdir, res)
        results.append((fit, i, res))
        log(f"    -> {res['success']}/{res['n']} tries; {fit[1]} of the evaluation's moments in the goal state")
        emit(type="learn_round", stage="candidate", index=i, idea=plan["idea"], success=res["success"], n=res["n"],
             in_goal=fit[1])
    if not results:
        return None
    fit, best, res = max(results, key=lambda r: r[0])
    win = main / f"candidate_{best}"
    for f in ("task.json", "policy.npz", "params.pkl", "train_meta.json"):
        if (win / f).exists():
            shutil.copy(win / f, main / f)
    t = load_task(task_path)  # keep the main folder's own paths and practice starts
    t["steps_from"] = f"nemotron, best of {len(plans)} plans in round {len(rounds)}"
    task_path.write_text(json.dumps(t, indent=1))
    for i in range(len(plans)):
        shutil.rmtree(main / f"candidate_{i}", ignore_errors=True)
    chosen = {"index": best, "idea": plans[best]["idea"], "tries": f"{res['success']}/{res['n']}", "in_goal": fit[1],
              "tried": len(plans), "rationale": prop.rationale}
    log(f"  kept candidate {best + 1}: {plans[best]['idea']} ({res['success']}/{res['n']})")
    emit(type="learn_round", stage="chosen", **chosen)
    return chosen | {"result": res}
