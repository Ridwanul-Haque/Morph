"""Morph in the browser: `morph app` serves a one-page UI on http://127.0.0.1:8765.

Same code as the command line (morph.project, morph.planner, ...); this file only adds:
  * a live picture: every simulation step goes through one hook (SimulationEnvironment.advance),
    which renders a frame now and then, paces trials to the chosen speed, and handles pause/stop
  * progress events for the page (Server-Sent Events), fed by morph.project.emit
  * small JSON endpoints for scenes, robots, goals, runs, replays of failed trials and gait training
  * videos (morph.video): tries are recorded during runs and rendered on request; learning videos are rendered
    when learning finishes

Standard library only (http.server); everything runs on this machine and listens on localhost.
"""

from __future__ import annotations

import copy
import io
import json
import re
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("MUJOCO_GL", "egl")  # headless GPU rendering; works from worker threads

from PIL import Image  # noqa: E402

from .. import project as P  # noqa: E402
from ..sim import DEFAULT_VIEW, SimulationEnvironment, nudge_view, view_adjusted  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MENAGERIE = ROOT / "third_party" / "mujoco_menagerie"
STATIC = Path(__file__).resolve().parent / "static"
UPLOADS = ROOT / "runs" / "app_uploads"

GROUPS = {"Franka Panda": "Arms", "UFactory xArm7": "Arms", "AgileX Piper": "Arms", "Kinova Gen3": "Arms", "UR5e": "Arms",
          "UFactory Lite6": "Arms", "SO-ARM100": "Arms", "Unitree Go1": "Legged", "Unitree Go2": "Legged",
          "ANYmal C": "Legged", "Spot with arm": "Legged", "Unitree H1": "Legged", "Unitree G1": "Legged"}
KNOWN_ROBOTS = [
    ("Franka Panda", "franka_emika_panda/panda.xml"), ("UFactory xArm7", "ufactory_xarm7/xarm7.xml"),
    ("AgileX Piper", "agilex_piper/piper.xml"), ("Kinova Gen3", "kinova_gen3/gen3.xml"),
    ("UR5e", "universal_robots_ur5e/ur5e.xml"), ("UFactory Lite6", "ufactory_lite6/lite6.xml"),
    ("SO-ARM100", "trs_so_arm100/so_arm100.xml"), ("Unitree Go1", "unitree_go1/go1.xml"),
    ("Unitree Go2", "unitree_go2/go2.xml"), ("ANYmal C", "anybotics_anymal_c/anymal_c.xml"),
    ("Spot with arm", "boston_dynamics_spot/spot_arm.xml"), ("Unitree H1", "unitree_h1/h1.xml"),
    ("Unitree G1", "unitree_g1/g1.xml"), ("Skydio X2", "skydio_x2/x2.xml"),
    ("Hello Robot Stretch", "hello_robot_stretch/stretch.xml"),
]


class Cancelled(Exception):
    pass


# ------------------------------------------------------------------ live picture + pacing
class Live:
    def __init__(self):
        self.cv = threading.Condition()
        self.jpeg: bytes | None = None
        self.seq = 0
        self.paused = False
        self.speed = 4.0  # sim seconds per wall second during trials; 0 = as fast as possible
        self.camera = "track"
        self.pace = False  # only trials are paced; setup experiments run flat out
        self.cancel = False
        self.env = None
        self.anchor = None
        self.last_render = 0.0
        self.env_seen = 0.0  # when the simulation on screen last stepped
        self.tick = 0
        self.recording: list | None = None
        self.view = (1280, 800)  # the page's viewer size in device pixels
        self.quality = "sharp"  # sharp: viewer size (<= 1920x1200) | balanced: <= 1280x800 | fast: <= 720x450
        self.renderer_name: str | None = None
        self.orbit = copy.deepcopy(DEFAULT_VIEW)  # the user's drag / zoom / pan on top of the chosen view
        self.dirty = False  # the view changed while paused: draw it again
        self.idle_lock = threading.Lock()

    def frame_size(self) -> tuple[int, int]:
        cap = {"sharp": (1920, 1200), "balanced": (1280, 800), "fast": (720, 450)}[self.quality]
        w, h = self.view
        k = min(1.0, cap[0] / w, cap[1] / h)
        return max(320, int(w * k) // 8 * 8), max(200, int(h * k) // 8 * 8)

    def install(self):
        orig = SimulationEnvironment.advance
        live = self

        def advance(env, *a, **k):
            r = orig(env, *a, **k)
            live.after(env)
            return r
        SimulationEnvironment.advance = advance

    def after(self, env):
        if getattr(env, "offscreen", False):  # video rendering, not for the live view
            return
        if self.cancel:
            raise Cancelled()
        while self.paused and not self.cancel:
            if self.dirty and env is self.env:
                self.dirty = False
                self.publish(self.render(env))
            time.sleep(0.05)
        now = time.time()
        if env is not self.env:
            # Two simulations can run at once (a practice replay while the learner's tries run; quick setup experiments):
            # stay on the one on screen while it's still going, and let the other run unseen and unslowed. Switching on
            # every step showed bits and pieces of both, rebuilding the renderer each time.
            if self.env is not None and now - self.env_seen < 0.6:
                return
            old, self.env, self.anchor = self.env, env, None
            if old is not None and getattr(old, "_render_thread", None) == threading.get_ident():
                try:
                    old.close()  # free its renderer (only from the thread that made it: GL contexts are per thread)
                except Exception:  # noqa: BLE001
                    pass
        self.env_seen = now
        self.tick += 1
        # ~12 frames per wall-clock second while trials play, ~8 during setup; rendering shares the job
        # thread with the physics, so a fixed wall-clock rate keeps fast-forward fast
        due = now - self.last_render > (1 / 12 if self.pace else 0.12)
        if due:
            self.last_render = now
            self.publish(self.render(env))
        if self.pace and self.speed > 0:
            t = float(env.data.time)
            if self.anchor is None or t < self.anchor[0]:
                self.anchor = (t, time.time())
            wait = self.anchor[1] + (t - self.anchor[0]) / self.speed - time.time()
            if wait > 0:
                time.sleep(min(wait, 0.25))
            elif wait < -0.5:
                self.anchor = (t, time.time())  # fell behind (rendering): don't try to catch up

    def render(self, env) -> bytes:
        cam = self.camera
        try:
            if cam != "track":
                env.model.camera(cam)
        except Exception:  # noqa: BLE001  (this scene has no such camera)
            cam = "track"
        w, h = self.frame_size()
        g = env.model.vis.global_
        if g.offwidth < w or g.offheight < h:  # scenes default to a 1280x720 offscreen buffer
            g.offwidth, g.offheight = max(g.offwidth, w), max(g.offheight, h)
            env.close()
        img = env.render(cam, w, h, view=self.orbit)
        env._render_thread = threading.get_ident()
        if self.renderer_name is None:
            try:
                from OpenGL import GL
                self.renderer_name = GL.glGetString(GL.GL_RENDERER).decode()
            except Exception:  # noqa: BLE001
                self.renderer_name = "unknown"
        b = io.BytesIO()
        fast = self.quality == "fast"
        Image.fromarray(img).save(b, "JPEG", quality=80 if fast else 90, subsampling=2 if fast else 0)
        return b.getvalue()

    def redraw_idle(self, draft: bool = False):
        """Draw the last scene again with the new view when nothing is running (the finished run's final
        state). Uses its own renderer: the old one belongs to the finished job's thread."""
        from .preview import jpeg, service
        env = self.env
        if env is None or not self.idle_lock.acquire(blocking=False):
            return  # a redraw is already under way; the next drag step brings it up to date
        try:
            w, h = self.frame_size()
            if draft:
                w, h = max(320, w // 2) // 8 * 8, max(200, h // 2) // 8 * 8
            img = service().draw(env.model, env.data, env.camera(self.camera, self.orbit), (w, h))
            self.publish(jpeg(img, draft))
        finally:
            self.idle_lock.release()

    def publish(self, jpeg: bytes):
        if self.recording is not None and self.pace:
            self.recording.append(jpeg)
        with self.cv:
            self.jpeg, self.seq = jpeg, self.seq + 1
            self.cv.notify_all()


# ------------------------------------------------------------------ session state + events
class App:
    def __init__(self):
        self.live = Live()
        self.live.install()
        self.subs: list[queue.Queue] = []
        self.history: list[dict] = []
        self.lock = threading.Lock()
        self.project: P.Project | None = None
        self.goal = None
        self.goal_info: dict = {}
        self.job: threading.Thread | None = None
        self.result: dict | None = None
        self.replays: dict = {}
        self.profiles: dict = {}
        self.custom_robots: list[tuple[str, str]] = []
        self.proc: subprocess.Popen | None = None
        self.compute = {"mode": "local", "gpu": "H100"}  # where gait training runs
        self.training_task: Path | None = None  # task being learned (for practice runs of checkpoints)
        self.suggestions: list = []  # what the person watching says (text) or rates (dicts), taken at the end of a round
        self.practice_seen: dict[int, dict] = {}  # practice replay id -> what it did
        self.suggest_lock = threading.Lock()
        self.planned: dict = {}  # (robot path, goal) -> task.json whose steps were already planned and shown
        self.placements: dict = {}  # robot path -> (x, y, heading rad) where the user put it; else Morph chooses
        self.latest_ckpt = None
        self.demo: threading.Thread | None = None
        self.llm = None
        self.videos = None  # TrialRecorder for the current job
        P.LISTENER = self.on_event

    # events
    def send(self, **e):
        e.setdefault("t", round(time.time(), 2))
        with self.lock:
            self.history.append(e)
            for q in self.subs:
                q.put(e)

    def on_event(self, e: dict):
        t = e.get("type")
        rec = self.videos
        if rec is not None:
            item = rec.on_event(e)
            if item is not None:
                e = e | {"video": item["id"]}
        if t == "trial_start":
            self.live.pace, self.live.recording, self.live.anchor = True, [], None
        elif t == "stage" and e.get("stage") == "setup":
            self.live.pace, self.live.recording = False, None
        elif t == "trial":
            frames, self.live.recording = self.live.recording or [], None
            if not e["success"] or not any(k[0] == e["robot"] for k in self.replays):
                self.replays[(e["robot"], e["i"])] = frames  # failures (and one success per robot) for replay
            e = e | {"replay": bool(frames)}
        self.send(**e)

    def on_checkpoint(self, e: dict):
        """A training checkpoint arrived: play it in the viewer (keeps replaying until a newer one comes)."""
        self.latest_ckpt = (Path(e["checkpoint"]), int(e["step"]))
        if self.training_task is not None and (self.demo is None or not self.demo.is_alive()):
            self.demo = threading.Thread(target=self._practice_loop, daemon=True)
            self.demo.start()

    def place(self, b: dict) -> dict:
        """Where the robot stands: set by the user (x, y, heading; or a point picked on the picture), or reset
        to Morph's choice. Returns where it now stands."""
        import math
        from ..goals import placement as goal_placement
        from .preview import floor_point, placement_of
        if self.project is None:
            raise ValueError("pick a scene first")
        path = b["path"]
        goal = None
        if self.goal is not None:
            gp = goal_placement(self.goal, self.project.scene)
            goal = (round(gp["x"], 4), round(gp["y"], 4), round(gp["yaw"], 4))
        robot = self.resolve(path)
        now = placement_of(self.project.scene, robot, goal, self.placements.get(path))
        if b.get("query"):
            return now
        if b.get("reset"):
            self.placements.pop(path, None)
        elif "u" in b:  # a point on the picture -> the floor there
            p = floor_point(float(b["u"]), float(b["v"]), float(b.get("aspect", 1.6)))
            if p is None:
                raise ValueError("that's above the horizon: pick a point on the floor")
            self.placements[path] = (p[0], p[1], math.radians(now["yaw_deg"]))
        else:
            self.placements[path] = (float(b.get("x", now["x"])), float(b.get("y", now["y"])),
                                     math.radians(float(b.get("yaw_deg", now["yaw_deg"]))))
        self.profiles, self.planned = {}, {}
        return placement_of(self.project.scene, robot, goal, self.placements.get(path))

    def plan_steps(self, path: str, use_llm: bool) -> dict:
        """How this robot will go about the goal (Nemotron's steps, or Morph's own), before learning."""
        from ..learn_task import make_task
        if self.goal is None or self.project is None:
            raise ValueError("understand a goal first")
        logs = []
        task = make_task(self.project, self.goal, self.resolve(path), llm=self.reasoning() if use_llm else None,
                         log=logs.append, stand_at=self.placements.get(path))
        self.planned[(path, id(self.goal))] = task
        return steps_view(task, self.project.scene.object_name) | {"log": logs}

    def _practice_loop(self):
        from ..learn_task import practice_run
        k = 0
        while self.training_task is not None and not self.live.cancel:
            path, step = self.latest_ckpt
            robot = json.loads(self.training_task.read_text())["robot"]
            self.send(type="caption", robot=robot, text=f"Practice try {k + 1}: " + (
                f"what it knows after {step / 1e6:.1f}M practice steps this round" if step else "before this round's practice"))
            self.live.pace, self.live.anchor = True, None
            try:
                r = practice_run(self.training_task, path, seed=k,
                                 stop=lambda p=path: self.live.cancel or self.training_task is None or self.latest_ckpt[0] != p,
                                 on_step=lambda i, name, rb=robot: self.send(type="caption", robot=rb,
                                                                             text=f"Step {i} done: {name}"))
            except Cancelled:
                break
            except Exception as ex:  # noqa: BLE001  (a bad checkpoint shouldn't stop training)
                self.send(type="log", robot=robot, text=f"practice run skipped: {ex}")
                time.sleep(2)
                continue
            self.practice_seen[k] = r.get("did", {})  # so a rating of this try can say what it did
            self.send(type="practice", robot=robot, step=step, id=k, **r)
            k += 1
        self.live.pace = False

    def reasoning(self):
        if self.llm is None:
            from ..reasoning import ReasoningService
            self.llm = ReasoningService(log_path=ROOT / "runs" / "app_inference.jsonl")
        return self.llm

    # scenes / robots
    def projects(self) -> list[dict]:
        out = []
        for base in (ROOT / "examples", UPLOADS / "scenes"):
            if base.exists():
                for d in sorted(base.iterdir(), key=lambda d: (d.name != "parcel", d.name)):  # the parcel cell first
                    if (d / "morph.toml").exists() or (d / "scene.xml").exists():
                        out.append({"name": d.name, "path": str(d), "uploaded": base != ROOT / "examples"})
        return out

    def robots(self) -> list[dict]:
        out = [{"name": n, "path": str(MENAGERIE / rel), "present": (MENAGERIE / rel).exists(), "group": GROUPS.get(n, "Other")}
               for n, rel in KNOWN_ROBOTS]
        return [{"name": n, "path": p, "present": True, "custom": True, "group": "Yours"} for n, p in self.custom_robots] + out

    def load_scene(self, path: str) -> dict:
        self.project = P.load_project(path)
        self.goal, self.goal_info, self.profiles, self.planned = None, {}, {}, {}
        s = self.project.scene
        return {"name": self.project.name, "summary": s.summary(), "description": self.project.description,
                "demo": self.project.demo, "fixtures": [{"name": f.name, "kind": f.kind, "text": f.describe()}
                                                         for f in s.fixtures.values()]}

    def resolve(self, path: str) -> str:
        """URDF -> converted MJCF (cached); MJCF unchanged."""
        from ..robot_import import import_robot
        if not hasattr(self, "_imports"):
            self._imports = {}
        if path not in self._imports:
            xml, notes = import_robot(path)
            self._imports[path] = (str(xml), notes)
        return self._imports[path][0]

    def profile(self, path: str) -> dict:
        from ..goals import needs_learning, placement, reach_problem, to_taskspec
        from ..introspect import capabilities_from_profile, profile_robot
        from ..learn_task import learned_for
        from ..planner import body_goal, check_templates, learning_only, rule_plan
        from ..task import default_spec
        key = (path, id(self.goal))
        if key in self.profiles:
            return self.profiles[key]
        prof = profile_robot(self.resolve(path))
        caps = capabilities_from_profile(prof)
        if self.goal is not None and self.project is not None:
            p = placement(self.goal, self.project.scene)
            checks = body_goal(learning_only(check_templates(prof, to_taskspec(self.goal, self.project.scene),
                                                             self.project.scene.block, lift_reason=p["lift_reason"]),
                                             needs_learning(self.goal), reach_problem(prof, self.goal, self.project.scene)),
                               self.goal)
        else:
            checks = check_templates(prof, default_spec(0.4, 0.12, 0.0))
        plan = rule_plan(checks)
        lift = placement(self.goal, self.project.scene)["lift_reason"] if self.goal is not None and self.project else ""
        why = f"needs lifting ({lift}); it can only push" if lift and not caps.prehensile else ""
        r = {"name": prof.name, "summary": prof.summary(), "kind": prof.kind, "dof": prof.dof, "base": prof.base,
             "mass_kg": round(prof.mass_kg, 1), "size_m": {k: round(v, 2) for k, v in prof.size_m.items()},
             "reach_m": round(max((v["max_horizontal"] for v in prof.reach_m.values()), default=0), 2),
             "gripper": prof.gripper.get("type", "none"), "opening_cm": round(prof.gripper.get("max_opening_m", 0) * 100, 1),
             "legs": prof.n_legs, "abilities": caps.contact_modes,
             "chains": [c for c in prof.chains if c["role"] != "other"],
             "plan": plan["plan_id"], "why_not": why or ("" if plan["plan_id"] != "unsupported" else plan.get("rationale", "")),
             "templates": [{"id": c.id, "ok": c.applicable, "met": c.met, "missing": c.missing} for c in checks],
             "import_notes": self._imports.get(path, ("", []))[1] if hasattr(self, "_imports") else [],
             "learned": (lambda t: t.parent.name if t else None)(learned_for(prof.name, self.goal, self.resolve(path))
                                                                  if self.goal else None),
             "recipes": P.program_body(prof)}
        if self.goal is not None and not r["learned"]:  # an earlier attempt that didn't work (or didn't finish)
            from ..learn_task import learning_status, task_dir
            st = learning_status(task_dir(prof.name, self.goal))
            if st["state"] != "none":
                r["attempt"] = {"state": "other_file" if st["good"] else st["state"], "success": st.get("success"),
                                "n": st.get("n")}
        if r["learned"]:
            r["plan"], r["why_not"] = "learned_skill", ""
        self.profiles[key] = r
        return r

    # goals
    def interpret(self, description: str, demo: str | None, use_llm: bool, robots: list[str] | None = None) -> dict:
        logs = []
        self.goal, self.goal_info = P.resolve_goal(self.project, description, demo,
                                                   self.reasoning() if use_llm else None, log=logs.append,
                                                   robots=[self.resolve(r) for r in (robots or [])][:6])
        self.profiles, self.planned = {}, {}
        return self.goal_view() | {"log": logs}

    def goal_view(self) -> dict:
        from ..goals import checked_when, explain
        return {"conditions": [c.model_dump() | {"text": c.describe(), "detail": explain(c, self.project.scene)}
                               for c in self.goal.conditions], "checked_when": checked_when(self.goal),
                "notes": self.goal.notes, "warnings": self.goal.warnings, "from": self.goal_info.get("goal_from", ""), "demo": self.goal_info.get("demo")}

    def set_conditions(self, conds: list[dict]) -> dict:
        from ..goals import Condition, GoalSpec, validate
        g = GoalSpec(object=self.project.scene.object_name,
                     conditions=[Condition(**{k: v for k, v in c.items() if k in Condition.model_fields}) for c in conds],
                     notes=self.goal.notes if self.goal else [])
        errs = validate(g, self.project.scene)
        if errs:
            raise ValueError("; ".join(errs))
        self.goal, self.profiles, self.planned = g, {}, {}
        self.goal_info["goal_from"] = (self.goal_info.get("goal_from", "") + ", edited").lstrip(", ")
        return self.goal_view()

    # jobs
    def busy(self) -> bool:
        return self.job is not None and self.job.is_alive()

    def start(self, target, *args, kind: str = "run", robot: str = ""):
        if self.busy():
            j = getattr(self, "job_info", None) or {}
            what = {"learn": "learning", "train": "learning to walk"}.get(j.get("kind"), "a run")
            raise RuntimeError(f"{what}{' (' + j['robot'] + ')' if j.get('robot') else ''} is already going: wait for it "
                               "to finish, or press Stop first")
        self.job_info = {"kind": kind, "robot": robot, "since": time.time()}
        with self.lock:
            self.history = []
        self.live.cancel, self.live.paused = False, False
        if kind in ("run", "learn"):
            from ..video import TrialRecorder, new_run_dir
            self.videos = TrialRecorder(new_run_dir(kind, robot.replace(", ", "+")))
        self.job = threading.Thread(target=self._guard, args=(target, *args), daemon=True)
        self.job.start()

    def _guard(self, target, *args):
        try:
            target(*args)
        except Cancelled:
            self.live.pace = False
            self.send(type="stopped")
        except Exception as e:  # noqa: BLE001
            self.send(type="error", text=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])
        finally:
            if self.videos is not None:
                self.videos.close()
                self.videos = None

    def run(self, robots: list[str], n: int, use_llm: bool):
        # this run's scene and goal, as they were when it started: changing them in the page meanwhile (loading a scene
        # clears the goal) doesn't change what the robots still to come are asked to do
        pj, goal, placements = self.project, self.goal, dict(self.placements)
        self.replays, result = {}, {"project": pj.name, "goal": goal.model_dump(),
                                    "goal_info": self.goal_info, "robots": []}
        self.send(type="run_start", robots=robots, n=n)
        for path in robots:
            name = Path(path).stem
            self.send(type="robot_start", robot=name, path=path)

            def log(text, _name=name):
                text = text.strip()
                if text:
                    self.send(type="log", robot=_name, text=text)
                    if text.startswith("mount"):
                        self.send(type="caption", robot=_name, text=f"setting up: trying {text}")
                    elif text.startswith("grip test"):
                        self.send(type="caption", robot=_name, text="Grip test: a practice pick to find how firmly to hold"
                                  + (" (shown on an empty floor)" if "empty floor" in text else ""))
            r = P.run_on_robot(pj, goal, path, n=n, llm=self.reasoning() if use_llm else None, log=log,
                               stand_at=placements.get(path))
            result["robots"].append(r)
        self.live.pace = False
        self.result = result
        self.send(type="done", summary=[{"robot": r["robot"], "plan": r["plan"]["plan_id"], "success": r.get("success", 0),
                                         "n": r.get("n", 0),
                                         "moveit": Path(r["moveit_export"]).stem if r.get("moveit_export") else None}
                                        for r in result["robots"]])

    def train(self, path: str):
        from ..introspect import profile_robot
        from ..planner import WALK_DIR
        from .compute import cloud_job
        prof = profile_robot(path)
        if self.compute["mode"] == "cloud":
            job = cloud_job(path, prof.name, self.compute["gpu"])
            self.send(type="cloud_job", robot=prof.name, **job)
            return
        py = ROOT / ".venv-rl" / "bin" / "python"
        out = WALK_DIR / prof.name
        if (out / "walk_policy.npz").exists():  # never overwrite a trained gait from the UI
            self.send(type="error", text=f"{prof.name} already has a walking policy ({out}); move it away to retrain")
            return
        cmd = [str(py), str(ROOT / "scripts" / "train_walk.py"), path, "--steps", "300000000", "--out", str(out)]
        if not py.exists():
            self.send(type="error", text="the RL environment (.venv-rl) isn't installed; see requirements-rl.lock")
            return
        try:
            code, oom = _stream_training(self, cmd, prof.name, 300_000_000, "walk")
        except Cancelled:
            if out.exists() and not any(out.iterdir()):
                out.rmdir()  # nothing was saved
            raise
        ok = code == 0 and (out / "walk_policy.npz").exists()
        self.profiles = {}
        self.send(type="training_done", robot=prof.name, ok=ok,
                  why="" if ok else ("the GPU ran out of memory" if oom else "the trainer stopped with an error"))


NOISE = ("Module", "external/", "W0", "I0", "WARNING: Attach", "Warp ", "CUDA Toolkit", "Devices:", '"cpu"', '"cuda:',
         "Kernel cache", "/home/", "noslip", "impratio", "integrator", "return build")


def robot_label(p: Path) -> str:
    return re.sub(r"(\.urdf)?\.(xacro|urdf|xml)$", "", p.name)


def _stream_training(app: "App", cmd: list, robot: str, steps: int, kind: str, quiet_start: bool = False):
    """Run a trainer, forward its progress; returns (exit code, ran out of GPU memory?)."""
    if not quiet_start:
        app.send(type="training_start", robot=robot, steps=steps, kind=kind, cmd=" ".join(cmd[1:]))
    else:  # a later round: its own number of practice steps (the copy round before it had none)
        app.send(type="training_round_start", robot=robot, steps=steps, kind=kind)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(ROOT), env=dict(os.environ))
    app.proc = proc
    oom, in_trace = False, False
    for line in proc.stdout:
        if app.live.cancel:
            proc.terminate()
            raise Cancelled()
        line = line.strip()
        if "out of memory" in line or "Failed to allocate" in line:
            oom = True
        if line.startswith('{"step"'):
            app.send(type="training", robot=robot, kind=kind, **json.loads(line))
        elif line.startswith('{"checkpoint"'):
            app.on_checkpoint(json.loads(line))
        elif line.startswith('{"obs"'):
            d = json.loads(line)
            app.send(type="log", robot=robot, text=f"learning with {d['actions']} motors; it watches {len(d['tips'])} end "
                                                   f"parts ({', '.join(t.removeprefix('robot_') for t in d['tips'])})")
        elif line.startswith('{"task"'):
            d = json.loads(line)
            app.send(type="log", robot=robot, text=f"practice finished: {d['steps'] / 1e6:.0f}M steps in "
                                                   f"{d['wall_s'] / 60:.1f} min ({d['num_envs']} copies at once)")
        elif line.startswith("Traceback"):
            in_trace = True  # stack traces go to the report, not the page; the cause is summarised below
        elif line and not in_trace and not line.startswith(NOISE):
            app.send(type="log", robot=robot, text=line[:300])
    code = proc.wait()
    app.proc = None
    if app.live.cancel:
        raise Cancelled()
    return code, oom


def steps_view(task: Path, object_name: str = "the object") -> dict:
    from ..skill_steps import steps_with_words
    t = json.loads(task.read_text())
    steps = steps_with_words(t.get("steps", []) + ([t["steps_final"]] if t.get("steps_final") else []), object_name)
    from ..reward_parts import view
    return {"robot": t["robot"], "steps": steps, "from": t.get("steps_from", ""), "rationale": t.get("steps_rationale", ""),
            **view(t)}


def _learn(app: "App", path: str, steps: int = 60_000_000, randomize: str = "light", use_llm: bool = True):
    from ..learn_task import learn_in_rounds, prepare_learning
    from .compute import cloud_task_job
    if app.goal is None or app.project is None:
        raise ValueError("understand a goal first")
    # what this learning is for, as it was when it started: learning takes an hour or more, and loading another scene
    # meanwhile clears the page's goal (that once crashed it at the very end, after all the practice)
    pj, goal = app.project, app.goal
    planned = app.planned.get((path, id(goal)))  # the steps already shown, if they were planned first
    app.send(type="caption", text="Working out how to go about it")
    task, prep = prepare_learning(pj, goal, app.resolve(path), llm=app.reasoning() if use_llm else None,
                                  log=lambda t: app.send(type="log", text=t.strip()), emit=lambda **e: app.on_event(e),
                                  stand_at=app.placements.get(path), steps_from=planned)
    if prep.get("demo"):
        app.send(type="log", text=f"Learning starts from Nemotron's program (it worked {prep['program_trials']}); "
                                  "the learner copies it first, then practises")
    robot = json.loads(task.read_text())["robot"]
    app.send(type="skill_steps", **steps_view(task, pj.scene.object_name))
    if app.compute["mode"] == "cloud":
        app.send(type="cloud_job", robot=robot, **cloud_task_job(str(task), robot, app.compute["gpu"], steps, randomize))
        return
    py = ROOT / ".venv-rl" / "bin" / "python"
    if not py.exists():
        raise ValueError("the RL environment (.venv-rl) isn't installed; see requirements-rl.lock")
    app.training_task, app.latest_ckpt = task, None
    state = {"envs": 1024, "first": True, "oom": False}

    def train(task_path, n_steps, restore) -> bool:
        """One round of practice on this GPU (fewer copies at once if it runs out of memory)."""
        app.training_task = task_path
        while True:
            cmd = [str(py), str(ROOT / "scripts" / "train_task.py"), str(task_path), "--steps", str(n_steps),
                   "--envs", str(state["envs"]), "--randomize", randomize] + (["--restore", str(restore)] if restore else []) \
                + (["--imitate-only"] if n_steps == 0 else [])
            code, oom = _stream_training(app, cmd, robot, n_steps, "task", quiet_start=not state["first"])
            state["first"], state["oom"] = False, oom
            if code == 0 or not oom or state["envs"] <= 128:
                break
            state["envs"] //= 2  # detailed collision meshes need more GPU memory per copy: practise with fewer at once
            app.send(type="log", robot=robot, text=f"The GPU ran out of memory (this robot has detailed collision shapes). "
                                                   f"Retrying with {state['envs']} practice copies at once: slower, same result.")
            app.send(type="training_retry", robot=robot, envs=state["envs"])
        app.training_task = None
        if app.demo is not None:
            app.demo.join(timeout=30)  # let the last practice run notice and stop
        ok = code == 0 and (task_path.parent / "policy.npz").exists()
        if ok:
            app.send(type="robot_start", robot=robot, path=path)
            app.send(type="stage", robot=robot, stage="trials", text="trying what it has learned so far")
        return ok

    def emit(**e):
        if e.get("type") == "learn_round":
            e["robot"] = robot
            app.send(**e)
        else:
            app.on_event(e)

    def advice() -> list[str]:  # the suggestions typed since the last round ended
        with app.suggest_lock:
            said, app.suggestions = app.suggestions, []
        return said

    with app.suggest_lock:
        app.suggestions = []
    res = learn_in_rounds(pj, goal, task, train, llm=app.reasoning() if use_llm else None,
                          log=lambda t: app.send(type="log", robot=robot, text=t.strip()), emit=emit, total_steps=steps,
                          advice=advice)
    app.training_task = None
    if not (task.parent / "policy.npz").exists():
        why = "the GPU ran out of memory even with 128 copies" if state["oom"] else "the trainer stopped with an error (see the report)"
        app.send(type="training_done", robot=robot, ok=False, why=why)
        return
    app.send(type="training_done", robot=robot, ok=True, kind="task", folder=task.parent.name)
    app.live.pace = False
    app.result = {"project": pj.name, "goal": goal.model_dump(), "robots": [{"robot": robot, "learned": task.parent.name, **res}]}
    app.send(type="learn_done", robot=robot, success=res["success"], n=res["n"], folder=task.parent.name,
             rounds=len(res.get("rounds", [])) + 1)
    app.send(type="done", summary=[{"robot": robot, "plan": "learned_skill", "success": res["success"], "n": res["n"]}])
    _learning_videos(app, task.parent, robot)


def _learning_videos(app: "App", folder: Path, robot: str):
    """Render the learning videos right after learning so the download buttons are instant. Takes a few minutes;
    Stop skips the rest (finished clips are kept)."""
    from ..video import learning_video
    app.send(type="learn_video", robot=robot, folder=folder.name, state="drawing")
    app.send(type="caption", text="Rendering the learning videos (a few minutes, Stop skips this)")
    try:
        learning_video(folder, stop=lambda: app.live.cancel,
                       log=lambda t: app.send(type="log", robot=robot, text=t.strip()))
    except InterruptedError:
        app.send(type="learn_video", robot=robot, folder=folder.name, state="stopped")
        return
    except Exception as e:  # noqa: BLE001  learning already succeeded, don't fail the job
        app.send(type="log", robot=robot, text=f"learning video skipped: {e}")
        app.send(type="learn_video", robot=robot, folder=folder.name, state="failed", why=str(e))
        return
    app.send(type="learn_video", robot=robot, folder=folder.name, state="ready")
    app.send(type="caption", text="Done. Videos are in the Learning tab")


APP: App | None = None


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):  # quiet unless MORPH_APP_DEBUG is set
        if os.environ.get("MORPH_APP_DEBUG"):
            sys.stderr.write(f"{time.strftime('%H:%M:%S')} {self.command} {self.path} {fmt % a}\n")

    def _json(self, obj, code=200):
        b = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                b = (STATIC / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            elif u.path == "/api/state":
                from ..reasoning import ReasoningService
                from .compute import estimates, nemotron_cost, task_estimates
                self._json({"nemotron": ReasoningService().available, "projects": APP.projects(), "robots": APP.robots(),
                            "busy": APP.busy(), "scene": APP.project.name if APP.project else None,
                            "goal": APP.goal_view() if APP.goal else None,
                            "control": {"paused": APP.live.paused, "speed": APP.live.speed, "camera": APP.live.camera,
                                        "orbit_moved": view_adjusted(APP.live.orbit),
                                        "quality": APP.live.quality},
                            "renderer": APP.live.renderer_name, "compute": APP.compute, "estimates": estimates(),
                            "task_estimates": task_estimates(),
                            "llm_cost": nemotron_cost(APP.llm.records if APP.llm else [])})
            elif u.path == "/api/job":  # what's running now (the page shows it, and keeps Run / Learn from starting another)
                j = getattr(APP, "job_info", None) or {}
                self._json({"busy": APP.busy(), "kind": j.get("kind"), "robot": j.get("robot", ""),
                            "seconds": round(time.time() - j["since"]) if APP.busy() and j.get("since") else 0,
                            "stopping": bool(APP.busy() and APP.live.cancel)})
            elif u.path == "/api/costs":
                from .compute import nemotron_cost
                self._json({"llm_cost": nemotron_cost(APP.llm.records if APP.llm else []), "renderer": APP.live.renderer_name})
            elif u.path == "/api/events":
                self._sse()
            elif u.path == "/api/stream.mjpg":
                self._mjpeg()
            elif u.path == "/api/replay.mjpg":
                self._replay(q.get("robot", ""), int(q.get("i", 0)))
            elif u.path == "/api/preview.jpg":
                from ..goals import placement
                from .preview import render_preview
                if APP.project is None:
                    raise ValueError("no scene")
                goal = None
                if APP.goal is not None:
                    p = placement(APP.goal, APP.project.scene)
                    goal = (round(p["x"], 4), round(p["y"], 4), round(p["yaw"], 4))
                if q.get("orbit"):  # a drag step sent along with the picture request (one round trip)
                    APP.live.orbit = nudge_view(APP.live.orbit, json.loads(q["orbit"]))
                w = min(1920, max(320, int(float(q.get("w", 1280))))) // 8 * 8
                h = min(1200, max(200, int(float(q.get("h", 800))))) // 8 * 8
                b = render_preview(APP.project.scene, APP.resolve(q["robot"]) if q.get("robot") else None, goal, (w, h),
                                   APP.live.orbit, APP.placements.get(q.get("robot")), draft=q.get("draft") == "1")
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(b)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(b)
            elif u.path == "/api/export":
                from ..learn_task import LEARNED, export
                folder = LEARNED / Path(q.get("task", "")).name  # a folder name only, nothing outside runs/learned
                if not (folder / "policy.npz").exists():
                    raise ValueError("nothing learned under that name yet")
                zp = export(folder / "task.json")
                data = zp.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", f"attachment; filename={folder.name}_policy.zip")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif u.path == "/api/export_moveit":
                from ..moveit_export import MOVEIT
                zp = MOVEIT / (Path(q.get("name", "")).name + ".zip")  # a name only, nothing outside runs/moveit
                if not zp.exists():
                    raise ValueError("no MoveIt export under that name (it's made when a scripted arm run succeeds)")
                data = zp.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", f"attachment; filename={zp.stem}_moveit.zip")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif u.path == "/api/videos":
                from ..video import listing
                self._json(listing())
            elif u.path == "/api/video":  # ?item=try:<id> or learn:<folder>:<clip>, rendered on first request
                from ..video import resolve
                path, title = resolve(q.get("item", ""))
                self._file(path, "video/mp4", f"{title}.mp4" if q.get("dl") else None)
            elif u.path == "/api/report":
                b = json.dumps(APP.result or {}, indent=2, default=str).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Disposition", "attachment; filename=morph_report.json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            else:
                self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{e}"}, 400)

    def do_POST(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/api/upload":
                self._json(self._upload(q.get("kind", ""), Path(q.get("name", "file")).name))
                return
            b = self._body()
            if u.path == "/api/scene":
                self._json(APP.load_scene(b["path"]))
            elif u.path == "/api/profile":
                self._json(APP.profile(b["path"]))
            elif u.path == "/api/robot_path":
                p = Path(b["path"]).expanduser()
                if not p.exists():
                    raise ValueError(f"{p} doesn't exist")
                APP.resolve(str(p))  # converts a URDF now, so problems show up here
                APP.custom_robots.append((robot_label(p), str(p)))
                self._json({"robots": APP.robots()})
            elif u.path == "/api/goal":
                if APP.project is None:
                    raise ValueError("pick a scene first")
                self._json(APP.interpret(b.get("description", ""), b.get("demo") or None, b.get("llm", True),
                                         b.get("robots") or []))
            elif u.path == "/api/conditions":
                self._json(APP.set_conditions(b["conditions"]))
            elif u.path == "/api/run":
                if APP.goal is None:
                    raise ValueError("interpret a goal first")
                APP.start(APP.run, b["robots"], int(b.get("n", 5)), b.get("llm", True), kind="run",
                          robot=", ".join(Path(r).stem for r in b["robots"]))
                self._json({"ok": True})
            elif u.path == "/api/learn":
                steps = int(min(max(int(b.get("steps", 60_000_000)), 1_000_000), 500_000_000))
                rnd = b.get("randomize", "light")
                if rnd not in ("off", "light", "full"):
                    raise ValueError("randomize must be off, light or full")
                APP.start(_learn, APP, b["path"], steps, rnd, bool(b.get("llm", True)), kind="learn",
                          robot=Path(b["path"]).stem)
                self._json({"ok": True})
            elif u.path == "/api/placement":
                self._json(APP.place(b))
            elif u.path == "/api/steps":
                self._json(APP.plan_steps(b["path"], bool(b.get("llm", True))))
            elif u.path == "/api/suggest":  # while it learns: what the person watching sees it doing wrong
                text = str(b.get("text", "")).strip()[:500]
                if not text:
                    raise ValueError("say what it's doing wrong, e.g. \"it grabs the top edge and the parcel slips\"")
                if APP.training_task is None:
                    raise ValueError("nothing is learning right now")
                with APP.suggest_lock:
                    APP.suggestions.append(text)
                now = bool(b.get("now"))
                if now:  # end the current round at its next snapshot, so the change applies sooner
                    (Path(APP.training_task).parent / "stop_round").write_text("1")
                APP.send(type="suggestion", text=text, now=now)
                self._json({"ok": True, "now": now})
            elif u.path == "/api/rate":  # 👍 / 👎 on the practice try just watched
                if APP.training_task is None:
                    raise ValueError("nothing is learning right now")
                k = int(b.get("id", -1))
                if k not in APP.practice_seen:
                    raise ValueError("that practice try is no longer known")
                rating = "good" if b.get("good") else "bad"
                with APP.suggest_lock:
                    APP.suggestions.append({"rating": rating, "did": APP.practice_seen[k], "id": k})
                self._json({"ok": True, "rating": rating})
            elif u.path == "/api/robustness":  # knock the object mid-task: the recipe alone vs the learned skill
                from ..learn_task import LEARNED, robustness
                if APP.goal is None or APP.project is None:
                    raise ValueError("understand a goal first")
                folder = LEARNED / Path(b.get("task", "")).name  # a folder name only, nothing outside runs/learned
                if not (folder / "policy.npz").exists():
                    raise ValueError("nothing learned under that name yet")
                self._json(robustness(APP.project, APP.goal, folder / "task.json", n=int(min(max(int(b.get("n", 10)), 2), 30)),
                                      log=lambda s: None))
            elif u.path == "/api/videos/bundle":
                from ..video import bundle
                items = [str(x) for x in (b.get("items") or [])][:60]
                as_zip = bool(b.get("zip"))
                path = bundle(items, as_zip)
                self._file(path, "application/zip" if as_zip else "video/mp4", path.name)
            elif u.path == "/api/train":
                APP.start(APP.train, b["path"], kind="train", robot=Path(b["path"]).stem)
                self._json({"ok": True})
            elif u.path == "/api/control":
                if "paused" in b:
                    APP.live.paused = bool(b["paused"])
                if "speed" in b:
                    APP.live.speed, APP.live.anchor = float(b["speed"]), None
                if "camera" in b:
                    APP.live.camera = b["camera"]
                if "quality" in b and b["quality"] in ("sharp", "balanced", "fast"):
                    APP.live.quality = b["quality"]
                if "orbit" in b or b.get("orbit_reset"):
                    APP.live.orbit = (copy.deepcopy(DEFAULT_VIEW) if b.get("orbit_reset")
                                      else nudge_view(APP.live.orbit, b["orbit"]))
                    if APP.busy():
                        APP.live.dirty = APP.live.paused  # a running job draws it on its next frame anyway
                    elif APP.live.env is not None and b.get("live"):
                        APP.live.redraw_idle(draft=bool(b.get("draft")))
                if "view" in b:
                    v = b["view"]
                    APP.live.view = (max(320, int(v["w"] * v.get("dpr", 1))), max(200, int(v["h"] * v.get("dpr", 1))))
                self._json({"paused": APP.live.paused, "speed": APP.live.speed, "camera": APP.live.camera,
                            "orbit": APP.live.orbit})
            elif u.path == "/api/compute":
                from .compute import CLOUD_GPUS
                if b.get("mode") in ("local", "cloud"):
                    APP.compute["mode"] = b["mode"]
                if b.get("gpu") in CLOUD_GPUS:
                    APP.compute["gpu"] = b["gpu"]
                self._json(APP.compute)
            elif u.path == "/api/stop":
                APP.live.cancel, APP.live.paused = True, False
                if APP.proc is not None:
                    APP.proc.terminate()
                self._json({"ok": True})
            else:
                self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{e}"}, 400)

    def _upload(self, kind: str, name: str) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(n)
        if kind == "scene" and Path(name).suffix.lower() in (".stl", ".obj"):  # an object's 3D model: a scene around it
            from ..object_import import object_scene
            src = UPLOADS / "objects" / Path(name).name
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_bytes(data)
            info = object_scene(src)
            return {"path": info["path"], "projects": APP.projects(), "note": info["note"]}
        if kind == "scene":
            d = UPLOADS / "scenes" / Path(name).stem
            d.mkdir(parents=True, exist_ok=True)
            (d / "scene.xml").write_bytes(data)
            (d / "morph.toml").write_text(f'[project]\nname = "{d.name}"\nscene = "scene.xml"\n')
            return {"path": str(d), "projects": APP.projects()}
        if kind == "robot":
            d = UPLOADS / "robots"
            d.mkdir(parents=True, exist_ok=True)
            if name.lower().endswith(".zip"):  # a description folder: URDF/MJCF plus its meshes
                import zipfile
                dest = d / Path(name).stem
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    for member in z.namelist():  # no paths that escape the folder
                        if member.startswith("/") or ".." in Path(member).parts:
                            raise ValueError(f"unsafe path in zip: {member}")
                    z.extractall(dest)
                cands = (sorted(dest.rglob("*.urdf")) or sorted(dest.rglob("*.urdf.xacro"))
                         or [x for x in sorted(dest.rglob("*.xacro")) if b"<link" in x.read_bytes()]
                         or [x for x in sorted(dest.rglob("*.xml")) if b"<mujoco" in x.read_bytes()[:4000]])
                if not cands:
                    raise ValueError("no .urdf, .xacro or MuJoCo .xml robot found in the zip")
                path = cands[0]
            else:
                path = d / name
                path.write_bytes(data)
            APP.resolve(str(path))
            APP.custom_robots.append((robot_label(path), str(path)))
            return {"path": str(path), "robots": APP.robots()}
        if kind == "demo":
            d = UPLOADS / "demos"
            d.mkdir(parents=True, exist_ok=True)
            (d / name).write_bytes(data)
            return {"path": str(d / name)}
        raise ValueError(f"unknown upload kind {kind!r}")

    def _file(self, path: Path, ctype: str, download: str | None = None):
        """Serve a file. Supports Range so <video> can seek."""
        size = path.stat().st_size
        start, end = 0, size - 1
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start, end = int(m.group(1)), min(int(m.group(2) or end), end)
            else:
                start = max(0, size - int(m.group(2)))
        self.send_response(206 if m and (m.group(1) or m.group(2)) else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        if self.headers.get("Range") and m:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            name = re.sub(r"[^A-Za-z0-9_.()+ -]+", "_", download.replace(" · ", " - ").replace(": ", " - ")).strip() \
                or "morph_video.mp4"
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        q: queue.Queue = queue.Queue()
        with APP.lock:
            backlog = list(APP.history)
            APP.subs.append(q)
        try:
            for e in backlog:
                self.wfile.write(f"data: {json.dumps(e, default=str)}\n\n".encode())
            self.wfile.flush()
            while True:
                try:
                    e = q.get(timeout=15)
                    self.wfile.write(f"data: {json.dumps(e, default=str)}\n\n".encode())
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                self.wfile.flush()
        finally:
            with APP.lock:
                APP.subs.remove(q)

    def _mjpeg_head(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def _frame(self, jpeg: bytes):
        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode()
                         + b"\r\n\r\n" + jpeg + b"\r\n")
        self.wfile.flush()

    def _mjpeg(self):
        self._mjpeg_head()
        live, seen = APP.live, -1
        while True:
            with live.cv:
                live.cv.wait_for(lambda: live.seq != seen, timeout=2.0)
                jpeg, seen = live.jpeg, live.seq
            if jpeg:
                self._frame(jpeg)

    def _replay(self, robot: str, i: int):
        frames = APP.replays.get((robot, i)) or []
        self._mjpeg_head()
        for f in frames:
            self._frame(f)
            time.sleep(1 / 25)


def _open_browser(url: str) -> None:
    """The default browser; on WSL that's the Windows one (Linux's opener there only prints "gio: Operation not
    supported")."""
    try:
        wsl = "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        wsl = False
    if wsl:
        try:
            subprocess.Popen(["explorer.exe", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except OSError:
            pass
    import webbrowser
    webbrowser.open(url)


def serve(port: int = 8765, open_browser: bool = True) -> None:
    global APP
    APP = App()
    from ..video import cleanup_tmp
    cleanup_tmp()
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as e:
        if e.errno == 98:  # address in use
            print(f"Port {port} is already in use: Morph (or something else) is still running there.\n"
                  f"  Open http://127.0.0.1:{port} if it's Morph, or stop it first:\n"
                  f"    fuser -k {port}/tcp        (or: kill %1, if you paused it with Ctrl+Z)\n"
                  f"  or start this one on another port:  morph app --port {port + 1}")
            sys.exit(1)
        raise
    srv.daemon_threads = True
    url = f"http://127.0.0.1:{port}"
    print(f"Morph app: {url}   (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.8, lambda: _open_browser(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        APP.live.cancel = True
        proc = getattr(APP, "proc", None)
        if proc is not None and proc.poll() is None:  # a training run it started: don't leave it on the GPU
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        srv.server_close()
        sys.stdout.flush()
        # skip Python's exit-time cleanup: the GPU renderer's (EGL on WSL's D3D12) and PyTorch's can hang there,
        # which left Ctrl+C stuck at "stopped"
        os._exit(0)
