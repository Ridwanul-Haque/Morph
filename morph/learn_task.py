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
from .reward_parts import view as reward_view
from .goals import GoalSpec, check, lets_go, placement, release_spot
from .introspect import home_qpos, profile_robot
from .project import Project, _free_spot, _params, _start_poses
from .robot_import import import_robot
from .sim import SimulationEnvironment, SimulationUnstable
from .skill_steps import StepTracker, plan_steps
from .task_model import (CTRL_DT, MAX_STEP, Residual, build_task_model, contact_held, indices, load_task, measure,
                         next_ctrl, observe, write_task)

ROOT = Path(__file__).resolve().parents[1]
LEARNED = ROOT / "runs" / "learned"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40]


def make_task(pj: Project, goal: GoalSpec, robot_path: str, llm=None, log=print, plan: bool = True,
              stand_at: tuple | None = None, folder: Path | None = None) -> Path:
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
        placed = True
    else:
        placed = False
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
    out = folder or task_dir(prof.name, goal)
    task = write_task(out / "task.json", robot_xml=str(xml), block=b, start=pj.scene.start, target=target,
                      mount=(mx, my, myaw), floating=floating, q0=q0, conditions=[c.model_dump() for c in goal.conditions],
                      scene_path=pj.scene.path, robot_name=prof.name, home=home,
                      fixtures=fixtures_geometry(pj.scene))
    if floating:  # standing height from the model's keyframe; without one, the height at which its lowest point
        # touches the floor (a wheeled base sits at about 0: "or 0.4" once dropped the Stretch from 40 cm)
        fj = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
        z = float(q[m.jnt_qposadr[fj] + 2])
        task["base_z"] = z if z > 0.02 else _resting_height(m, q, fj)
    if gripper:
        task["gripper"] = gripper
        # a gripper snaps shut (0.2 s), as on the arm the programs run on; at the old 1 s the arm lifted before the
        # fingers had closed and every copied grasp slipped (xArm7 lift: 0/5 replayed, 4/5 with this)
        task["gripper_ticks"] = 10
    if not floating:  # like the arm the programs run on (and real arm controllers): gravity compensated
        task["gravcomp"] = True
    task["placed_by_user"] = placed
    task["critic_extras"] = True  # the critic sees grip, progress and this copy's physics (train_task._critic_extras)
    if plan:
        steps, info = plan_steps(prof, goal, pj.scene, p, llm, log)
        task["steps"] = [st.model_dump() for st in steps[:-1]]  # the last step is the goal itself
        task["steps_final"] = steps[-1].model_dump()
        task["steps_from"] = info["from"]
        if info.get("rationale"):
            task["steps_rationale"] = info["rationale"]
    (out / "task.json").write_text(json.dumps(task, indent=1))
    physics_check(out / "task.json", log)
    return out / "task.json"


def _resting_height(m: mujoco.MjModel, q, fj: int) -> float:
    """The free base's height at which the robot's lowest collision shape just touches the floor (z = 0)."""
    d = mujoco.MjData(m)
    d.qpos[:] = q
    d.qpos[m.jnt_qposadr[fj] + 2] = 0.0
    mujoco.mj_forward(m, d)
    low = min((float(d.geom_xpos[g][2] - _half_height(m, d, g)) for g in range(m.ngeom)
               if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE),
              default=0.0)
    return round(max(0.0, -low) + 0.002, 4)


def _half_height(m, d, g: int) -> float:
    """How far a geom reaches below its centre (its bounding box, turned as it is)."""
    h = m.geom_aabb[g][3:]
    R = d.geom_xmat[g].reshape(3, 3)
    return float(abs(R[2, 0]) * h[0] + abs(R[2, 1]) * h[1] + abs(R[2, 2]) * h[2] - R[2] @ m.geom_aabb[g][:3])


def physics_check(task_path: Path, log=print, seconds: float = 3.0) -> dict:
    """Does the robot stay still in the learner's fast physics when told to hold still? The fast physics (4 ms steps, 4
    solver iterations, strong grip friction) suits arms, but on the Stretch's light wheels it didn't settle: they spun
    themselves up to 300 rad/s and every practice run was the robot sliding in circles. If it drifts, the learner's
    physics uses ordinary friction for this robot (stable at the same speed); the final tries keep the accurate physics."""
    task = load_task(task_path)
    tried = [{}, {"impratio": 1.0}, {"impratio": 1.0, "iterations": 20, "ls_iterations": 10}]
    result = {}
    for fix in tried:
        task["physics"] = fix
        try:
            moved, fastest = _idle_drift(task, seconds)
        except Exception:  # noqa: BLE001  (a model that can't even be built here: leave it to the learner)
            return {}
        result = {"settings": fix, "moved_m": round(moved, 3), "fastest_joint": round(fastest, 1)}
        if moved < 0.05 and fastest < 50:
            break
    t = json.loads(Path(task_path).read_text())
    if result.get("settings"):
        t["physics"] = result["settings"]
        log(f"  the learning physics wasn't stable for this robot (it drifted on its own); using "
            + ", ".join(f"{k} {v:g}" for k, v in result["settings"].items()) + f": still now ({result['moved_m'] * 100:.1f} cm)")
    else:
        t.pop("physics", None)
    t["physics_check"] = result
    Path(task_path).write_text(json.dumps(t, indent=1))
    return result


def _idle_drift(task: dict, seconds: float) -> tuple[float, float]:
    """(how far the robot's base, or else the object, moved; the fastest any robot joint turned) holding still."""
    m = build_task_model(task)
    ix = indices(m, task)
    d = mujoco.MjData(m)
    d.qpos[ix.home_adr] = ix.home_val
    bx, by, byaw = task["start"]
    d.qpos[ix.block_q:ix.block_q + 7] = [bx, by, task["block"]["half_z"], math.cos(byaw / 2), 0, 0, math.sin(byaw / 2)]
    if ix.base >= 0:
        mx, my, myaw = task["mount"]
        d.qpos[ix.base_q:ix.base_q + 7] = [mx, my, task.get("base_z", 0.4), math.cos(myaw / 2), 0, 0, math.sin(myaw / 2)]
    d.ctrl[:] = np.clip(ix.q0, ix.act_lo, ix.act_hi)
    mujoco.mj_forward(m, d)
    body = ix.base if ix.base >= 0 else ix.block
    p0, fastest = d.xpos[body].copy(), 0.0
    robot_dofs = [m.jnt_dofadr[j] for j in range(m.njnt) if m.joint(j).name.startswith("robot_")
                  and m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)]
    for k in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if k > int(0.5 / m.opt.timestep):  # after it has settled onto the floor
            v = d.qvel[robot_dofs] if robot_dofs else np.zeros(1)
            if not np.all(np.isfinite(v)):
                return 1e9, 1e9
            fastest = max(fastest, float(np.max(np.abs(v))))
    moved = float(np.linalg.norm(d.xpos[body][:2] - p0[:2]))
    return (moved if np.isfinite(moved) else 1e9), fastest


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


def learning_status(folder: Path) -> dict:
    """How learning in this folder went: {"state": "learning" | "done" | "none", "success", "n", "good"}. A skill is
    "good" (runs use it) only if learning finished and it succeeded in at least half of its final tries. Folders from
    before status.json was kept count as good if their learning ran to the end (train_meta.json, no plans left over
    from an interrupted search) and its last evaluation reached the goal state at all."""
    folder = Path(folder)
    if not (folder / "policy.npz").exists() or not (folder / "task.json").exists():
        return {"state": "none", "good": False}
    f = folder / "status.json"
    if f.exists():
        st = json.loads(f.read_text())
        st["good"] = st.get("state") == "done" and st.get("n", 0) > 0 and 2 * st.get("success", 0) >= st["n"]
        return st
    finished = (folder / "train_meta.json").exists() and not list(folder.glob("candidate_*"))
    reached = False
    if finished:  # it must have reached the goal in its last evaluation at least some of the time
        curve = json.loads((folder / "train_meta.json").read_text()).get("curve") or [{}]
        reached = (curve[-1].get("success") or 0) > 0
    return {"state": "done" if finished else "learning", "good": finished and reached}


def learned_for(robot_name: str, goal: GoalSpec, robot_xml: str | Path | None = None) -> Path | None:
    """task.json of a learned skill for this robot and goal that runs should use: learning finished and it worked
    (learning_status good), and (given robot_xml) it was learned on this very robot file, not another one with the same
    name (a hand re-imported with its meshes is a different body from the one without). Else None."""
    d = task_dir(robot_name, goal)
    if not learning_status(d)["good"]:
        return None
    if robot_xml is not None:
        try:
            if Path(json.loads((d / "task.json").read_text())["robot_xml"]).resolve() != Path(robot_xml).resolve():
                return None
        except (OSError, KeyError, ValueError):
            return None
    return d / "task.json"


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


def rollout(env, m, ix, task, pol, start, params, seed, stop=None, watch=(), settle_s: float = 1.0,
            residual: "Residual | None" = None, disturb: tuple | None = None) -> bool:
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
    if residual is None and task.get("residual"):
        residual = Residual.of(task, task.get("_dir", "."))
    try:
        for t in range(int(task["episode_s"] / CTRL_DT)):
            if stop is not None and stop():
                break
            o = observe(np, d.qvel, d.actuator_length, d.actuator_velocity, d.xpos, d.xmat, last, ix, tgt)
            if residual is not None:  # corrections on top of the recipe: it also sees how far through it is
                o = np.concatenate([o, residual.phase(np, t)])
            a = pol(o)
            last = a
            if residual is not None:
                a = residual.action(np, t, a)
            ctrl = next_ctrl(np, ctrl, a, ctrl0, ix.act_scale, ix.act_lo, ix.act_hi, ix.act_step)
            if disturb is not None and t == disturb[0]:  # a bump: the object suddenly gets this velocity (m/s)
                d.qvel[ix.block_v:ix.block_v + 3] += disturb[1]
            env.advance(ctrl)
            for w in watch:
                w.t = t
            if watch:
                v = measure(np, d.xpos, d.xmat, d.qvel, ix, task, held=contact_held(m, d, ix), ctrl=d.ctrl, qpos=d.qpos)
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
    w = _Watch(task)
    blew = rollout(env, m, ix, task, NumpyPolicy(policy_path), task["start"], PhysicalParams(), seed, stop, (tr, w))
    p = env.block_pose()
    t = task["target"]
    e = w.ext
    return {"dist_cm": round(math.hypot(p.x - t["x"], p.y - t["y"]) * 100, 1), "blew_up": blew,
            "tilt_deg": round(math.degrees(p.tilt)), "steps_reached": tr.reached, "steps_total": len(tr.steps),
            "stuck_at": tr.steps[tr.reached][0] if tr.reached < len(tr.steps) else "",
            # what it did, in numbers (for comparing the tries a person liked with the ones they didn't)
            "did": {"closest_to_object_cm": round(e["touch"] * 100, 1), "best_grip_0_to_1": round(e["grip"], 2),
                    "seconds_really_held": round(e["held_s"], 1), "highest_lift_cm": round(e["lifted"] * 100, 1),
                    "moved_object_cm": round(e["moved"] * 100, 1), "fastest_object_m_s": round(e["speed"], 2),
                    "ended_cm_from_target": round(math.hypot(p.x - t["x"], p.y - t["y"]) * 100, 1),
                    "object_tilt_deg": round(math.degrees(p.tilt)), "steps_reached": tr.reached}}


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


def robustness(pj: Project, goal: GoalSpec, task_path: Path, n: int = 10, at: float = 0.25, speed: float = 0.7,
               log=print) -> dict:
    """The difference learning makes: the same tries with the object bumped part-way through (it suddenly slides at
    `speed` m/s in a random direction, `at` of the way through the recipe), once with the recipe alone (the policy's
    corrections switched off) and once with the learned skill. A recipe follows fixed moves and doesn't notice; the
    learned skill sees where the object is and corrects."""
    task = load_task(task_path)
    m = build_task_model(task, accurate=True)
    ix = indices(m, task)
    env = SimulationEnvironment(m, control_dt=CTRL_DT)
    res = Residual.of(task, task_path.parent)
    T = res.T if res is not None else int(task["episode_s"] / CTRL_DT / 1.6)
    rng = np.random.default_rng(7)
    bumps = [(int(at * T), np.array([*(speed * np.array([math.cos(a), math.sin(a)])), 0.0]))
             for a in rng.uniform(0, 2 * math.pi, n)]
    pols = {"learned skill": NumpyPolicy(task_path.parent / "policy.npz")}
    if res is not None:
        pols = {"recipe alone": lambda o: np.zeros(m.nu)} | pols
    out = {"n": n, "bump": f"the {pj.scene.object_name} is knocked at {speed:.1f} m/s in a random direction, "
                            f"{at:.0%} of the way through the task"}
    for name, pol in pols.items():
        ok = 0
        for i, (start, bump) in enumerate(zip(_start_poses(pj.scene, n), bumps)):
            hist = History([c.model_dump() for c in goal.conditions], task.get("fixtures", {}),
                           (task["block"]["half_x"], task["block"]["half_y"], task["block"]["half_z"]))
            blew = rollout(env, m, ix, task, pol, start, _params(i), i, watch=(hist,), residual=res, disturb=bump)
            held = contact_held(m, env.data, ix) > 0.5
            r = check(goal, pj.scene, m, env.data, env.block_pose(), env.block_speed(), not held, history=hist)
            ok += bool(r["success"]) and not blew
        out[name] = ok
        log(f"  bumped mid-task, {name}: {ok}/{n}")
    return out


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

res = spec.get("residual")  # corrections on top of a recipe (recipe.npy): action = recipe[t] + scale * output
recipe = np.load(res["file"]) if res else None

def observe(last, t):
    bp = d.xpos[ix["block"]]; R = d.xmat[ix["block"]].reshape(3, 3); yaw = math.atan2(R[1, 0], R[0, 0])
    parts = [(d.actuator_length - q0) * mask, d.actuator_velocity * 0.1, bp - [tx, ty, tz], R[:, 2], [math.cos(yaw - tyaw), math.sin(yaw - tyaw)],
             d.qvel[ix["block_v"]:ix["block_v"] + 3] * 0.5, (d.xpos[tips] - bp).ravel(), last]
    if ix["base"] >= 0:
        Rb = d.xmat[ix["base"]].reshape(3, 3); v = ix["base_v"]
        parts += [d.xpos[ix["base"]] - bp, Rb.T @ [0, 0, -1.0], Rb.T @ d.qvel[v:v + 3] * 0.5, d.qvel[v + 3:v + 6] * 0.2]
    if recipe is not None:
        parts.append([min(t / len(recipe), 1.5)])
    return np.concatenate(parts)

d.qpos[:] = spec["initial_qpos"]; d.ctrl[:] = np.clip(q0, lo, hi); mujoco.mj_forward(m, d)
steps = round(spec["control_dt"] / m.opt.timestep)
last, ctrl = np.zeros(m.nu), np.clip(q0, lo, hi)
viewer = None
if "--view" in sys.argv:
    import mujoco.viewer; viewer = mujoco.viewer.launch_passive(m, d)
for t in range(int(spec["episode_s"] / spec["control_dt"])):
    last = policy(observe(last, t))
    a = last if recipe is None else np.clip(recipe[min(t, len(recipe) - 1)] + res["scale"] * last, -1, 1)
    want = np.clip(q0 + a * np.asarray(spec["action_scale"]), lo, hi)
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
    res = Residual.of(task, src)
    if res is not None:  # the recipe the policy corrects (its action per control step)
        np.save(dest / "recipe.npy", res.base)
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
                        "last action (n): the network's own output"] + (["base - object (3)", "gravity in base frame (3)", "base velocity in base frame x 0.5 (3)",
                                               "base angular velocity x 0.2 (3)"] if ix.base >= 0 else [])
                       + (["how far through the recipe: control steps / recipe length, at most 1.5 (1)"] if res is not None else []),
        "network": "normalise with obs_mean/obs_std, 3 hidden layers with swish, tanh on the first half of the output",
        "indices": {"joint_qadr": ix.home_adr.tolist(), "joint_vadr": ix.joint_vadr.tolist(), "pos_mask": ix.pos_mask.tolist(),
                    "tips": ix.tips.tolist(), "block": ix.block,
                    "block_v": ix.block_v, "base": ix.base, "base_v": ix.base_v, "act_lo": ix.act_lo.tolist(),
                    "act_hi": ix.act_hi.tolist()},
        "initial_qpos": d.qpos.tolist(), "training": {k: meta.get(k) for k in ("steps", "wall_s")},
        "mount": list(task["mount"]), "randomize": meta.get("randomize", "off"),
        "residual": {"file": "recipe.npy", "scale": res.scale,
                     "meaning": "corrections on top of a recipe: action = recipe[t] + scale * output (t = control steps "
                                "since the start, clipped to the recipe's last); the observation ends with t / len(recipe)"}
        if res is not None else None,
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
        self.frontier_t: dict[int, int] = {}
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
        if self._improved is not None:  # keep the state of its best moment so far on that step (and when it was)
            self.frontier[self._improved] = d.qpos.copy()
            self.frontier_t[self._improved] = getattr(self, "t", 0)
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
        when = [w.frontier_t[k] for w in watches if k in w.frontier]
        if starts:
            f = task_path.parent / "starts.npz"
            qs, cs = np.array(starts), np.full((len(starts), m.nu), np.nan)  # no commands: hold the pose it's in
            ts = np.array(when, dtype=np.int32)
            if f.exists() and task.get("demo"):  # keep the program's moments too
                old = np.load(f)
                qs = np.concatenate([old["qpos"], qs])
                cs = np.concatenate([old["ctrl"] if "ctrl" in old else np.full((len(old["qpos"]), m.nu), np.nan), cs])
                ts = np.concatenate([old["t"] if "t" in old else np.zeros(len(old["qpos"]), np.int32), ts])
            np.savez(f, qpos=qs, ctrl=cs, t=ts)
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
    terms = term_stats(task_path.parent, task)
    if terms:
        out["reward_terms"] = terms
    out.update(reward_part_stats(task_path.parent, task))
    if task.get("dropped_terms"):  # so the next revision writes them in a form Morph can measure
        out["your_reward_terms_morph_couldnt_use"] = task["dropped_terms"]
    return out


def reward_part_stats(folder: Path, task: dict) -> dict:
    """What each part of the reward (reward_parts) paid over a practice run at the start and the end of the last round,
    the weights in use, and a possible loophole (the reward rising while the checklist doesn't)."""
    from .reward_parts import REWARD_PARTS, loophole, part_totals, weights
    meta = json.loads((folder / "train_meta.json").read_text()) if (folder / "train_meta.json").exists() else {}
    curve = [c for c in meta.get("curve", []) if c.get("parts")]
    if not curve:
        return {}
    a, b = part_totals(curve[0], task), part_totals(curve[-1], task)
    w = weights(task)
    out = {"reward_parts": [{"part": k, "what": REWARD_PARTS[k][0] if k in REWARD_PARTS else "Nemotron's term",
                             "weight": w.get(k, 1.0), "total_per_run_at_start": a.get(k), "total_per_run_at_end": b[k]}
                            for k in b]}
    hole = loophole(curve, task)
    if hole:
        out["possible_loophole"] = hole["text"]
    return out


def term_stats(folder: Path, task: dict) -> list[dict]:
    """How each of Nemotron's reward terms behaved in the last practice: its average value per tick in the first and
    last evaluation (Eureka's "reward reflection": a term that never moves isn't helping)."""
    terms = task.get("reward_terms") or []
    meta = json.loads((folder / "train_meta.json").read_text()) if (folder / "train_meta.json").exists() else {}
    curve = [c for c in meta.get("curve", []) if c.get("terms") and len(c["terms"]) == len(terms)]
    if not terms or not curve:
        return []
    return [{"name": t["name"], "formula": t["formula"], "weight": t["weight"],
             "average_per_tick_at_start": curve[0]["terms"][k], "average_per_tick_at_end": curve[-1]["terms"][k]}
            for k, t in enumerate(terms)]


def revise_task_steps(task_path: Path, llm, diagnosis: dict, goal: GoalSpec, scene, rounds: list, log=print,
                      advice: list[str] | None = None, ratings: list[dict] | None = None) -> dict | None:
    """Nemotron reads what the learner actually did and rewrites the steps (smaller first steps, reachable
    thresholds, graded quantities); Morph checks the new steps and writes them into the task."""
    from .introspect import profile_robot
    from .skill_steps import Step, body_only, check_steps, check_terms, object_steps, robot_brief, scales
    task = load_task(task_path)
    prof = profile_robot(task["robot_xml"])
    p = placement(goal, scene)
    errors, fix = [], None
    for attempt in range(2):
        prop, rec = llm.revise_steps(robot_brief(prof), [c.describe() for c in goal.conditions], task.get("steps", []),
                                     diagnosis, scales(scene, p), rounds, fix, advice=advice, ratings=ratings)
        raw = [Step(**s) for s in prop.steps if isinstance(s, dict) and s.get("done_when")]
        new, errors = check_steps(raw, list(scene.fixtures), scene.object_name)
        terms, t_err = check_terms(prop.reward_terms, list(scene.fixtures), scene.object_name)
        if attempt == 0:  # asked once to fix them; after that the plan is kept without the terms Morph can't measure
            errors += t_err
        elif t_err:
            log(f"  (left out a reward term Morph couldn't measure: {t_err[0]})")
        from .reward_parts import check_weights
        wts, w_err = check_weights(prop.reward_weights, task)
        errors += w_err
        if body_only(goal):
            errors += object_steps(new)
        if new and not errors:
            task["steps"] = [s.model_dump() for s in new]
            task["reward_terms"] = terms
            task["dropped_terms"] = t_err if attempt else []
            if wts != (task.get("reward_weights") or {}):
                log("  reward weights now: " + ", ".join(f"{k} ×{v:g}" for k, v in wts.items()))
            task["reward_weights"] = wts
            task["steps_from"] = f"nemotron, revised after round {len(rounds)}"
            task_path.write_text(json.dumps(task, indent=1))
            return {"steps": task["steps"], "rationale": prop.rationale}
        fix = {"your_previous_steps": [s.model_dump() for s in raw], "your_previous_reward_terms": prop.reward_terms,
               "problems": errors}
    log(f"  (Nemotron's revised steps couldn't be measured: {'; '.join(errors[:2])}; keeping the old ones)")
    return None


def learn_in_rounds(pj: Project, goal: GoalSpec, task_path: Path, train, llm=None, log=print, emit=None,
                    total_steps: int = 60_000_000, rounds: int = 4, tries: int = 10, candidates: int = 3,
                    advice=None) -> dict:
    """Learning in rounds (see _learn_rounds), with its state kept in status.json: runs only use the skill once
    learning has finished and it worked (learning_status)."""
    folder = task_path.parent
    for c in folder.glob("candidate_*"):  # left over from an interrupted plan search
        shutil.rmtree(c, ignore_errors=True)
    (folder / "status.json").write_text(json.dumps({"state": "learning"}))
    res = _learn_rounds(pj, goal, task_path, train, llm, log, emit, total_steps, rounds, tries, candidates, advice)
    (folder / "status.json").write_text(json.dumps({"state": "done", "success": int(res.get("success", 0)),
                                                    "n": int(res.get("n", 0)), "tries": tries}))
    return res


def _learn_rounds(pj: Project, goal: GoalSpec, task_path: Path, train, llm=None, log=print, emit=None,
                  total_steps: int = 60_000_000, rounds: int = 4, tries: int = 10, candidates: int = 3,
                  advice=None) -> dict:
    """Practise in short rounds. After each: try it; if it isn't there yet, measure what it actually does and
    (with Nemotron) revise the steps, then keep practising from where it got to. candidates > 0: instead of one
    revision, Nemotron proposes that many different plans and the best after brief practice is kept (Eureka-style).
    A round that did better than the one before keeps its plan: the plan only changes when progress stalls.
    train(task_path, steps, restore) -> bool runs the GPU trainer. advice() -> list[str]: what the person watching has
    said since it was last asked (e.g. "it grabs the top edge and it slips"); it goes to Nemotron with the measurements
    at the end of the round, and the plan is revised from both."""
    emit = emit or (lambda **e: None)

    def take_advice(r: int, history: list) -> bool:
        got = advice() if advice else []
        said = [a.strip() for a in got if isinstance(a, str) and a.strip()]
        rated = [a for a in got if isinstance(a, dict) and a.get("rating") in ("good", "bad")]
        if rated and not said:
            g, b = sum(a["rating"] == "good" for a in rated), sum(a["rating"] == "bad" for a in rated)
            said = [f"(no words: rated {g} practice tries as looking right and {b} as looking wrong)"]
        if not said:
            return False
        if llm is None or not getattr(llm, "available", False):
            emit(type="learn_round", round=r, rounds=rounds, stage="advice", said=said,
                 rationale="Nemotron is off, so suggestions can't be turned into steps")
            return False
        diag = diagnose(task_path, task_path.parent / "policy.npz")
        history.append({"round": r, "tries_succeeded": f"{res['success']}/{res['n']}", "diagnosis": diag, "person_said": said})
        log(f"  you said: {' / '.join(said)}")
        try:
            rev = revise_task_steps(task_path, llm, diag, goal, pj.scene, history, log, advice=said, ratings=rated)
        except Exception as e:  # noqa: BLE001
            rev = None
            log(f"  (Nemotron's revision couldn't be used: {e})")
        why = rev["rationale"] if rev else ("Nemotron's answer couldn't be turned into steps Morph can measure, so the plan "
                                            "stays as it was (see the Log)")
        history[-1]["revision"] = why
        emit(type="learn_round", round=r, rounds=rounds, stage="advice", said=said, rationale=why)
        if rev:
            t = load_task(task_path)
            emit(type="skill_steps", **reward_view(t), robot=t["robot"], steps=t["steps"] + [t["steps_final"]], **{"from": t["steps_from"]},
                 rationale=why)
        return bool(rev)

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
            # how far to trust the program while practising: the less the copy works, the freer practice is to change
            t = load_task(task_path)
            t["copy_trust"] = round(res["success"] / max(res["n"], 1), 2)
            task_path.write_text(json.dumps(t, indent=1))
            if t["copy_trust"] < 0.3:
                log(f"  the copy only worked {res['success']}/{res['n']}: practice is free to move well away from the program")
            take_advice(0, history)
    stuck_before = None
    for r in range(1, rounds + 1):
        restore = task_path.parent / "params.pkl" if (r > 1 or task.get("demo")) else None
        _easy_first(task_path, r, rounds, log)
        emit(type="learn_round", round=r, rounds=rounds, stage="practice", steps=per)
        if not train(task_path, per, restore):
            res["why_not"] = "the trainer stopped (see the log)"
            break
        res = try_policy(pj, goal, task_path, n=tries, log=log, emit=emit)
        emit(type="learn_round", round=r, rounds=rounds, stage="tried", success=res["success"], n=res["n"])
        best = _keep_best(task_path, res, best, f"round {r}")
        if res["success"] >= math.ceil(0.6 * tries) or r == rounds:
            break
        if take_advice(r, history):  # the person watching said what's wrong: that revision is this round's
            t = load_task(task_path)  # and loosen the pull towards the program, so the change can show
            t["copy_trust"] = round(min(float(t.get("copy_trust", 1.0)), 0.5), 2)
            task_path.write_text(json.dumps(t, indent=1))
            last_fit = _fitness(task_path.parent, res)
            continue
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
        moved = _remount(task_path, diag, log)
        if moved:  # it couldn't reach: stand closer and start again (nothing it learned carries over)
            history[-1]["remounted"] = moved
            emit(type="learn_round", round=r, rounds=rounds, stage="remounted", text=moved)
            last_fit, stuck_before = None, None
            continue
        if stuck and stuck_before == stuck["step"] and diag.get("practice_from"):
            _focus(task_path, stuck, log)  # stuck at the same step twice: practise mostly from there
        stuck_before = stuck["step"] if stuck else None
        emit(type="learn_round", round=r, rounds=rounds, stage="diagnosed", diagnosis=diag)
        if llm is not None and getattr(llm, "available", False) and candidates > 0:
            each = max(2_000_000, per // (candidates + 1))
            chosen = search_step_plans(pj, goal, task_path, train, llm, diag, history, each, tries, candidates, log,
                                       lambda **e: emit(**e, round=r, rounds=rounds))
            if chosen:
                history[-1]["revision"] = f"best of {chosen['tried']} plans: {chosen['idea']}"
                t = load_task(task_path)
                emit(type="skill_steps", **reward_view(t), robot=t["robot"], steps=t["steps"] + [t["steps_final"]], **{"from": t["steps_from"]},
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
                emit(type="skill_steps", **reward_view(t), robot=t["robot"], steps=t["steps"] + [t["steps_final"]], **{"from": t["steps_from"]},
                     rationale=rev["rationale"])
                emit(type="learn_round", round=r, rounds=rounds, stage="revised", rationale=rev["rationale"])
    if best is not None and _fitness(task_path.parent, res) < best[0]:  # a later round did worse: go back
        for f, data in best[2].items():
            (task_path.parent / f).write_bytes(data)
        log(f"  kept the best result, from {best[3]}: {best[1]['success']}/{best[1]['n']}")
        res = best[1]
    res["rounds"] = history
    return res


def _remount(task_path: Path, diag: dict, log=print) -> str | None:
    """A bolted-down robot that never got near the object because it stands too far away: move its mount closer
    (once), so the object is well inside its reach, and practise again from scratch. Not when the user placed it,
    and not when it learns from a recipe (the recipe was recorded where it stands)."""
    task = load_task(task_path)
    did = diag.get("what_the_robot_did", {})
    touch = did.get("closest any of its end parts got to the object's surface (cm)", 0.0) / 100
    if task.get("floating") or task.get("demo") or task.get("placed_by_user") or task.get("remounted") or touch < 0.08:
        return None
    reach = max((v["max_horizontal"] for v in profile_robot(task["robot_xml"]).reach_m.values()), default=0.0)
    mx, my, myaw = task["mount"]
    sx, sy = task["start"][:2]
    dist = math.hypot(sx - mx, sy - my)
    if reach <= 0 or dist < 0.8 * reach:
        return None  # in reach: it's the skill that's missing, not the spot
    step = dist - 0.6 * reach
    task["mount"] = [mx + (sx - mx) / dist * step, my + (sy - my) / dist * step, myaw]
    task["remounted"] = True
    task_path.write_text(json.dumps(task, indent=1))
    for f in ("params.pkl", "starts.npz"):
        (task_path.parent / f).unlink(missing_ok=True)
    task = load_task(task_path)
    task.pop("starts", None)
    task_path.write_text(json.dumps(task, indent=1))
    msg = (f"it never got within {touch * 100:.0f} cm of the {task.get('object', 'object')}: the object is {dist:.2f} m away and "
           f"it reaches about {reach:.2f} m, so it now stands {step:.2f} m closer")
    log(f"  {msg}")
    return msg


def _easy_first(task_path: Path, r: int, rounds: int, log=print) -> None:
    """Easy-first practice (a reverse curriculum, Florensa et al. 2017): with a recipe to start from, early rounds
    practise from moments late in it (already holding it, nearly there), and each round moves the starts back
    towards the beginning, so the end of the task is mastered first and every later skill builds on it."""
    task = load_task(task_path)
    st = task.get("starts")
    if not task.get("demo") or not st or st.get("focus"):
        return
    z = np.load(task_path.parent / st["file"])
    if "t" not in z or not len(z["t"]):
        return
    frac = max(0.0, 0.5 - 0.25 * (r - 1)) if rounds > 1 else 0.0  # 50% of the way in, then 25%, then all of it
    st["from_t"] = int(frac * int(z["t"].max()))
    task["starts"] = st
    task_path.write_text(json.dumps(task, indent=1))
    if st["from_t"]:
        log(f"  round {r}: practising from the later part of the recipe first (from {frac:.0%} of the way in)")


def _focus(task_path: Path, stuck: dict, log=print) -> None:
    """Stuck at the same step two rounds running: most practice starts where it got closest on that step, so it
    practises that step on its own rather than spending most of its time on what it can already do."""
    task = load_task(task_path)
    if task.get("starts"):
        task["starts"].update(share=0.7, from_t=0, focus=stuck["step"])
        task_path.write_text(json.dumps(task, indent=1))
        log(f"  stuck at step {stuck['step']} twice: practising mostly from its closest moments on that step")


# ------------------------------------------------------------------ learning from a working program
def demo_to_task(task_path: Path, ad, demos: list, start_every: int = 4, share: float = 0.5,
                 residual: float | None = 0.3) -> dict:
    """A program's successful runs in the learner's terms: the observation the policy would see at each moment and
    the action that gives the program's commands (for imitation), plus states along the way to start practice from
    (hand around it, gripping it, lifting it). ad: the adapter whose model the runs were recorded on (an arm the
    program ran on; it must stand where the task's arm stands, make_task stand_at), or None if they were recorded on
    the task's own model (body_programs).

    residual (default): the learner outputs corrections on top of the first run's actions (Residual), scaled by this;
    the targets to copy are then the corrections each run needed (zero along the first run). None: it outputs whole
    actions, copied from the runs."""
    task = load_task(task_path)
    m = build_task_model(task)
    d = mujoco.MjData(m)
    src = ad.model if ad is not None else m
    jmap, amap = [], []
    for j in range(src.njnt):
        n = src.joint(j).name
        if n == "block_free":
            tj = m.joint("block_free").id
            jmap.append((src.jnt_qposadr[j], m.jnt_qposadr[tj], 7, src.jnt_dofadr[j], m.jnt_dofadr[tj], 6))
            continue
        try:
            tj = m.joint(n if ad is None else "robot_" + n).id
        except KeyError:
            continue
        span = 7 if int(src.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE) else 1
        jmap.append((src.jnt_qposadr[j], m.jnt_qposadr[tj], span, src.jnt_dofadr[j], m.jnt_dofadr[tj], 6 if span == 7 else 1))
    for a in range(src.nu):
        try:
            amap.append((a, m.actuator(src.actuator(a).name if ad is None else "robot_" + src.actuator(a).name).id))
        except KeyError:
            continue
    # start where the demonstration started (the arm's own ready pose, hand open), and let "no action" mean
    # "keep the commands it started with": otherwise the copy meets a start it has never seen
    q_first, _, c_first = demos[0][0]
    for j in range(m.njnt):
        name = m.joint(j).name
        if name.startswith("robot_") and name.removeprefix("robot_") in task.get("home", {}):
            try:
                sj = src.joint(name if ad is None else name.removeprefix("robot_")).id
            except KeyError:
                continue
            task["home"][name.removeprefix("robot_")] = float(q_first[src.jnt_qposadr[sj]])
    q0 = list(task["q0"])
    for sa, ta in amap:
        q0[ta] = float(c_first[sa])
    task["q0"] = q0
    task["residual"] = {"scale": residual} if residual else None
    task_path.write_text(json.dumps(task, indent=1))
    ix = indices(m, task)
    # a joint's command can only go so far from its start (0.8 rad by default) and so fast (2.5 rad/s): if the recipe
    # goes further or faster (a fist bends fingers 90°, a throw swings hard), widen that joint's range so the learner
    # can do what the recipe did (else even a perfect copy stops half-way)
    need_far, need_fast = np.zeros(m.nu), np.zeros(m.nu)
    for demo in demos:
        prev = None
        for _, _, ctrl in demo:
            c = np.clip(ix.q0, ix.act_lo, ix.act_hi).copy()
            for sa, ta in amap:
                c[ta] = ctrl[sa]
            need_far = np.maximum(need_far, np.abs(c - ix.q0))
            if prev is not None:
                need_fast = np.maximum(need_fast, np.abs(c - prev))
            prev = c
    wide = {str(a): [round(float(need_far[a] * 1.15), 4), round(float(need_fast[a] * 1.2), 4)] for a in range(m.nu)
            if need_far[a] > 0.95 * ix.act_scale[a] or need_fast[a] > ix.act_step[a]}
    if wide:
        task["action_units"] = wide
        task_path.write_text(json.dumps(task, indent=1))
        ix = indices(m, task)
    t = task["target"]
    tgt = (t["x"], t["y"], t["z"], t["yaw"])

    def frames(demo):  # each moment of a run in the task model: (state set in d, the absolute action)
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
            yield k, c, np.clip((c - ix.q0) / ix.act_scale, -1.0, 1.0)

    base = np.array([a for _, _, a in frames(demos[0])]) if residual else None
    obs, acts, sq, sc, st = [], [], [], [], []
    for demo in demos:
        last = np.zeros(m.nu)
        for k, c, a in frames(demo):
            o = observe(np, d.qvel, d.actuator_length, d.actuator_velocity, d.xpos, d.xmat, last, ix, tgt)
            if residual:  # what the policy outputs here: the correction from the first run's action at this moment
                o = np.concatenate([o, [min(k / len(base), 1.5)]])
                a = np.clip((a - base[min(k, len(base) - 1)]) / residual, -1.0, 1.0)
            obs.append(o)
            acts.append(a)
            last = a
            if k % start_every == 0:
                sq.append(d.qpos.copy())
                sc.append(c)
                st.append(k)
    extra = {"base": base} if residual else {}
    np.savez(task_path.parent / "demo.npz", obs=np.array(obs), act=np.array(acts), **extra)
    np.savez(task_path.parent / "starts.npz", qpos=np.array(sq), ctrl=np.array(sc), t=np.array(st, dtype=np.int32))
    task["demo"] = "demo.npz"
    task["starts"] = {"file": "starts.npz", "share": share, "from": "moments along the program that worked"}
    # enough time to do it at a learner's pace: the program's own duration, and then some
    longest = max(len(dm) for dm in demos) * CTRL_DT
    task["episode_s"] = round(max(task.get("episode_s", 8.0), 1.6 * longest), 1)
    if task.get("gripper") and ad is not None:  # the gripper strength the arm's grip test settled on (the program ran with it)
        task["gripper"]["gain_scale"] = float(getattr(ad.cfg, "gripper_gain_scale", 1.0))
    task_path.write_text(json.dumps(task, indent=1))
    return {"samples": len(obs), "starts": len(sq), "residual": bool(residual)}


def attach_grasp(task_path: Path, pj: Project, goal: GoalSpec, log=print, program=None) -> dict | None:
    """Morph's planned grip (contact_plan.plan_grasp: the one that held best when tried) on the task, for the learner
    to aim at: where the fingers go, which sides they close on, which way the hand comes in. Only when the goal needs
    the object held and the robot has a gripper, and only when it agrees with the program it copies (a program that
    gripped its own way is copied as it is)."""
    from .body_programs import BodyRunner, needs_hold
    if not needs_hold(goal, pj.scene) or (program is not None and not any(s.get("pick") for s in program)):
        return None
    task = load_task(task_path)
    try:
        rn = BodyRunner(task)
    except Exception:  # noqa: BLE001
        return None
    hand = next((n for n, p_ in rn.parts.items() if p_.can_grip), None)
    if hand is None:
        return None
    from .contact_plan import plan_grasp
    g = plan_grasp(rn, pj.scene, hand, log=log)
    if not g or not g.get("geom"):
        return g
    task["grasp"] = dict(g["geom"], desc=g["desc"])
    task.pop("_dir", None)
    task_path.write_text(json.dumps(task, indent=1))
    log(f"  the learner aims for Morph's planned grip: {g['desc']}")
    return g


def prepare_learning(pj: Project, goal: GoalSpec, robot_path: str, llm=None, log=print, emit=None, stand_at=None,
                     steps_from: Path | None = None) -> tuple[Path, dict]:
    """The task to learn, from a working program when there can be one. For an arm with a gripper, Nemotron writes a
    program from the hand-down blocks (programs.solve); its runs are recorded, and kept if they still work replayed in
    the learner's own physics. Otherwise (or for any other bolted-down robot) a program from the general blocks, run on
    the learner's own model (body_programs.solve_body). The learner then starts from the program and learns
    corrections on top of it. Without a working program: the task as usual (learning from scratch). steps_from: a
    task.json whose steps were already planned (reused, no new planning)."""
    from .programs import solve
    emit = emit or (lambda **e: None)
    xml, _ = import_robot(robot_path)
    prof = profile_robot(xml)
    info: dict = {"demo": False}
    planned = load_task(steps_from) if steps_from and Path(steps_from).exists() else None
    from .programs import too_flat_for_hand_down
    if llm is not None and getattr(llm, "available", False) and prof.base == "fixed" and prof.has_gripper \
            and not too_flat_for_hand_down(pj.scene):  # a flat object: straight to the general blocks (below)
        emit(type="caption", text="Nemotron is writing a program to learn from")
        r = solve(pj, goal, robot_path, prof.name, llm, log=log, emit=emit, n=8, stand_at=stand_at)
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
            ok = replay_check(pj, goal, task)
            # replayed move for move in the learner's physics (no anti-slip contacts there) a grasp may slip; practice
            # learns the corrections that fix it (xArm7 lift: 0/10 copied, 10/10 after one round), so keep it
            log(f"  learning from Nemotron's program: {len(r['demos'])} successful runs, {d['samples']} moments to copy, "
                f"{d['starts']} to practise from (replayed in the learner's physics: {ok}/3"
                + ("; practice will correct the rest)" if ok < 3 else ")"))
            info = {"demo": True, "program": r["program"], "program_trials": f"{r['success']}/{r['n']}", "replay": f"{ok}/3"}
            return task, info
        else:
            log("  (the hand-down building blocks didn't give a working program: trying the general ones)")
    if planned:
        task = Path(steps_from)
    else:
        task = make_task(pj, goal, robot_path, llm=llm, log=log, stand_at=stand_at)
    if llm is not None and getattr(llm, "available", False) and prof.base == "fixed":
        # any bolted-down body: a program over its parts, run on the learner's own model, so its runs are copied as is
        from .body_programs import solve_body
        emit(type="caption", text="Nemotron is writing a program to learn from (moving the robot's parts)")
        r = solve_body(pj, goal, task, llm, log=log, emit=emit, n=8, attempts=3 if prof.has_gripper else 5)
        if r.get("demos"):
            d = demo_to_task(task, None, r["demos"])
            log(f"  learning from Nemotron's program: {len(r['demos'])} successful runs, {d['samples']} moments to copy, "
                f"{d['starts']} to practise from")
            attach_grasp(task, pj, goal, log=log, program=r.get("program"))
            return task, {"demo": True, "program": r["program"], "program_trials": f"{r['success']}/{r['n']}"}
    if info.get("demo") is False and prof.base == "fixed" and prof.has_gripper:
        # no program worked: before an hour of learning from scratch, can this gripper pick the object up at all?
        from .body_programs import BodyRunner, needs_hold, pick_test
        if needs_hold(goal, pj.scene):
            rn = BodyRunner(load_task(task))
            if rn.parts and any(p_.can_grip for p_ in rn.parts.values()):
                ok_pick, said = pick_test(rn, pj.scene)
                if not ok_pick:
                    raise ValueError(f"{prof.name} can't learn this: {said}. Learning wouldn't change that (it would "
                                     "practise for an hour and fail), so Morph stops here. Try a robot whose gripper fits "
                                     "this object (the Franka Panda picks up a pen), or a goal that doesn't need lifting it.")
                attach_grasp(task, pj, goal, log=log)  # learning from scratch: aim for the grip that held
    if info.get("demo") is False and prof.base == "fixed" and llm is not None and getattr(llm, "available", False):
        log("  (no working program to learn from: learning from scratch)")
    return task, info


def replay_check(pj: Project, goal: GoalSpec, task_path: Path, n: int = 3) -> int:
    """How many of n tries the recorded program succeeds when replayed move for move in the learner's own physics
    (the learner's starting point: corrections of zero)."""
    task = load_task(task_path)
    res = Residual.of(task, task_path.parent)
    if res is None:
        return 0
    m = build_task_model(task, accurate=True)
    ix = indices(m, task)
    env = SimulationEnvironment(m, control_dt=CTRL_DT)
    ok = 0
    for i, start in enumerate(_start_poses(pj.scene, n)):
        hist = History([c.model_dump() for c in goal.conditions], task.get("fixtures", {}),
                       (task["block"]["half_x"], task["block"]["half_y"], task["block"]["half_z"]))
        blew = rollout(env, m, ix, task, lambda o: np.zeros(m.nu), start, _params(i), i, watch=(hist,), residual=res)
        held = contact_held(m, env.data, ix) > 0.5
        r = check(goal, pj.scene, m, env.data, env.block_pose(), env.block_speed(), not held, history=hist)
        ok += bool(r["success"]) and not blew
    return ok


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
    from .skill_steps import Step, check_steps, check_terms, robot_brief, scales
    emit = emit or (lambda **e: None)
    task = load_task(task_path)
    prof = profile_robot(task["robot_xml"])
    try:
        prop, _ = llm.step_candidates(robot_brief(prof), [c.describe() for c in goal.conditions], task.get("steps", []),
                                      diagnosis, scales(pj.scene, placement(goal, pj.scene)), rounds, k)
    except Exception as e:  # noqa: BLE001
        log(f"  (Nemotron couldn't propose step plans: {e})")
        return None
    plans = [{"idea": "keep the current steps", "steps": task.get("steps", []), "reward_terms": task.get("reward_terms", [])}]
    for c in prop.candidates:
        raw = [Step(**s) for s in c.get("steps", []) if isinstance(s, dict) and s.get("done_when")]
        ok, errors = check_steps(raw, list(pj.scene.fixtures), pj.scene.object_name)
        terms, t_err = check_terms(c.get("reward_terms"), list(pj.scene.fixtures), pj.scene.object_name)
        if ok and not errors:  # a reward term Morph can't measure is left out; the plan itself is still tried
            plans.append({"idea": str(c.get("idea", ""))[:200], "steps": [s.model_dump() for s in ok], "reward_terms": terms,
                          "dropped_terms": t_err})
            if t_err:
                log(f"  (a proposed plan is tried without a reward term Morph couldn't measure: {t_err[0]})")
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
        t = dict(task, steps=plan["steps"], reward_terms=plan["reward_terms"], dropped_terms=plan.get("dropped_terms", []))
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
              "tried": len(plans), "rationale": prop.rationale, "reward_terms": plans[best]["reward_terms"]}
    log(f"  kept candidate {best + 1}: {plans[best]['idea']} ({res['success']}/{res['n']})")
    emit(type="learn_round", stage="chosen", **chosen)
    return chosen | {"result": res}
