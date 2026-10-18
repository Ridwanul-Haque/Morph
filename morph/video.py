"""Videos of runs and of learning.

  TrialRecorder   records each try as qpos/mocap/camera at 25 Hz (<run>/<robot>/try_NN.npz + the model as .mjb).
                  render_try() turns it into an mp4 later. Rendering is ~50 ms/frame under WSL, too slow to do live.
  learning_video  round_N.mp4, final.mp4, compare.mp4 and learning_progress.mp4 in <learned>/videos/
  bundle          several videos -> one mp4 or a zip

1280x720, 25 fps, real time, same tracking camera as the live view.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
import zipfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .sim import LAST_CAM, SimulationEnvironment

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
VIDEOS = RUNS / "videos"
SIZE = (1280, 720)
FPS = 25
HOLD_S = 1.6  # seconds the result banner stays up

BG = (10, 15, 26)
TEXT, DIM = (231, 237, 246), (148, 163, 184)
OK, BAD, ACCENT = (52, 211, 153), (248, 113, 113), (106, 168, 255)


# ------------------------------------------------------------------ recording
class Tape:
    """State, camera and caption at 25 Hz."""

    def __init__(self):
        self.t, self.qpos, self.mpos, self.mquat, self.cam, self.cap = [], [], [], [], [], []
        self.captions: list[str] = []
        self.caption = ""

    def add(self, env: SimulationEnvironment) -> None:
        d = env.data
        if self.t and 0 <= d.time - self.t[-1] < 1 / FPS - 1e-9:
            return
        saved = dict(LAST_CAM)
        cam = env.camera("track")  # camera() also updates LAST_CAM, which the viewer uses; restore it
        LAST_CAM.clear()
        LAST_CAM.update(saved)
        if not self.captions or self.captions[-1] != self.caption:
            self.captions.append(self.caption)
        self.t.append(float(d.time))
        self.qpos.append(d.qpos.copy())
        self.mpos.append(d.mocap_pos.copy())
        self.mquat.append(d.mocap_quat.copy())
        self.cam.append([*cam.lookat, cam.distance, cam.azimuth, cam.elevation])
        self.cap.append(len(self.captions) - 1)

    def save(self, path: Path, **meta) -> None:
        np.savez_compressed(path, t=np.array(self.t), qpos=np.array(self.qpos), mocap_pos=np.array(self.mpos),
                            mocap_quat=np.array(self.mquat), cam=np.array(self.cam), cap=np.array(self.cap, int),
                            captions=json.dumps(self.captions), meta=json.dumps(meta, default=str))

    @classmethod
    def load(cls, path: Path) -> tuple["Tape", dict]:
        z = np.load(path)
        tp = cls()
        tp.t, tp.qpos, tp.mpos, tp.mquat, tp.cam = list(z["t"]), list(z["qpos"]), list(z["mocap_pos"]), \
            list(z["mocap_quat"]), list(z["cam"])
        tp.cap, tp.captions = list(z["cap"]), json.loads(str(z["captions"]))
        return tp, json.loads(str(z["meta"]))

    def __len__(self):
        return len(self.t)


def _taping(env: SimulationEnvironment, tape: Tape):
    orig = env.advance
    env.offscreen = True

    def advance(*a, **k):
        r = orig(*a, **k)
        tape.add(env)
        return r
    env.advance = advance


class TrialRecorder:
    """Records each try between the trial_start and trial events. Only the thread that started the try is recorded,
    since the app runs practice replays in other threads."""

    _installed = False
    _active: list["TrialRecorder"] = []

    def __init__(self, out: Path):
        self.out = Path(out)
        self.cur: dict | None = None
        self.items: list[dict] = []
        self.models: dict[int, tuple] = {}  # id(model) -> (model, file); keep a ref so the id isn't reused
        self.passes: dict[str, int] = {}
        self.lock = threading.Lock()
        TrialRecorder._install()
        TrialRecorder._active.append(self)

    @classmethod
    def _install(cls):
        if cls._installed:
            return
        orig = SimulationEnvironment.advance

        def advance(env, *a, **k):
            r = orig(env, *a, **k)
            for rec in cls._active:
                rec._after(env)
            return r
        SimulationEnvironment.advance = advance
        cls._installed = True

    def _after(self, env):
        c = self.cur
        if c is None or c["thread"] != threading.get_ident() or getattr(env, "offscreen", False):
            return
        if c["env"] is None:
            c["env"] = env
        if env is c["env"]:
            c["tape"].add(env)

    def on_event(self, e: dict) -> dict | None:
        t = e.get("type")
        if t == "trial_start":
            robot = str(e.get("robot") or "robot")
            if e.get("i", 0) == 0:
                self.passes[robot] = self.passes.get(robot, 0) + 1  # e.g. program tries after the built-in skill
            self.cur = {"robot": robot, "i": int(e.get("i", 0)), "n": int(e.get("n", 0)), "thread": threading.get_ident(),
                        "env": None, "tape": Tape(), "pass": self.passes.get(robot, 1)}
        elif t == "caption" and self.cur is not None and e.get("robot", self.cur["robot"]) == self.cur["robot"]:
            self.cur["tape"].caption = str(e.get("text", ""))
        elif t == "trial" and self.cur is not None:
            c, self.cur = self.cur, None
            if c["env"] is None or not len(c["tape"]):
                return None
            return self._save(c, bool(e.get("success")), list(e.get("reasons") or []))
        return None

    def _save(self, c: dict, success: bool, reasons: list[str]) -> dict:
        d = self.out / _safe(c["robot"])
        d.mkdir(parents=True, exist_ok=True)
        m = c["env"].model
        with self.lock:
            if id(m) not in self.models:
                f = d / f"model_{len(self.models) + 1}.mjb"
                mujoco.mj_saveModel(m, str(f))
                self.models[id(m)] = (m, f)
            model_file = self.models[id(m)][1]
        name = f"try_{c['i'] + 1:02d}" + (f"_set{c['pass']}" if c["pass"] > 1 else "")
        path = d / f"{name}.npz"
        title = f"{c['robot']} · try {c['i'] + 1}" + (f" of {c['n']}" if c["n"] else "")
        c["tape"].save(path, model=model_file.name, robot=c["robot"], i=c["i"], n=c["n"], success=success,
                       reasons=reasons, title=title, set=c["pass"])
        item = {"id": video_id(path), "path": str(path), "robot": c["robot"], "i": c["i"], "n": c["n"], "success": success,
                "reasons": reasons, "title": title, "set": c["pass"]}
        self.items.append(item)
        (self.out / "index.json").write_text(json.dumps({"made": time.strftime("%Y-%m-%d %H:%M"), "tries": self.items},
                                                        indent=1))
        return item

    def close(self) -> None:
        self.cur = None
        if self in TrialRecorder._active:
            TrialRecorder._active.remove(self)
        self.models = {}


def video_id(path: Path) -> str | None:
    """Path relative to runs/, or None if it's outside runs/."""
    p = Path(path).resolve()
    return p.relative_to(RUNS.resolve()).as_posix() if RUNS.resolve() in p.parents else None


def from_id(vid: str) -> Path:
    """Inverse of video_id. Refuses anything outside runs/."""
    p = (RUNS / vid).resolve()
    if RUNS.resolve() not in p.parents:
        raise ValueError("not a Morph video")
    return p


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "robot"


# ------------------------------------------------------------------ drawing
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_LOCK = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    with _LOCKS_LOCK:
        return _LOCKS.setdefault(str(path), threading.Lock())


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for p in (name, f"/usr/share/fonts/truetype/dejavu/{name}"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _fit(draw: ImageDraw.ImageDraw, text: str, font, width: float) -> str:
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text.rstrip() + "…"


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: float, lines: int = 2) -> list[str]:
    out, cur = [], ""
    for w in text.split():
        if draw.textlength((cur + " " + w).strip(), font=font) <= width:
            cur = (cur + " " + w).strip()
        else:
            out.append(cur)
            cur = w
    out.append(cur)
    out = [x for x in out if x]
    if len(out) > lines:
        out = out[:lines - 1] + [_fit(draw, " ".join(out[lines - 1:]), font, width)]
    return out


def overlay(img: np.ndarray, title: str, caption: str = "", t: float | None = None,
            result: tuple[bool, str] | None = None) -> np.ndarray:
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    W, H = im.size
    s = H / 720
    ft, fc = _font(int(24 * s), bold=True), _font(int(19 * s))
    pad = int(14 * s)
    lines = [_fit(dr, title, ft, W - 4 * pad)]
    if caption:
        lines.append(_fit(dr, caption, fc, W - 4 * pad))
    w = max(dr.textlength(x, font=ft if k == 0 else fc) for k, x in enumerate(lines)) + 2 * pad
    h = pad + int(30 * s) + (int(26 * s) if caption else 0) + pad // 2
    dr.rounded_rectangle([pad, pad, pad + w, pad + h], radius=int(10 * s), fill=(10, 15, 26, 190))
    dr.text((2 * pad, pad + int(8 * s)), lines[0], fill=TEXT, font=ft)
    if caption:
        dr.text((2 * pad, pad + int(40 * s)), lines[1], fill=DIM, font=fc)
    if t is not None:
        tt = f"{t:4.1f} s"
        tw = dr.textlength(tt, font=fc)
        dr.rounded_rectangle([W - pad - tw - 2 * pad, H - pad - int(34 * s), W - pad, H - pad], radius=int(8 * s),
                             fill=(10, 15, 26, 170))
        dr.text((W - 2 * pad - tw, H - pad - int(29 * s)), tt, fill=DIM, font=fc)
    if result is not None:
        ok, why = result
        fb = _font(int(30 * s), bold=True)
        msg = "✓ Success" if ok else "✕ Failed"
        rows = [msg] + (_wrap(dr, why, fc, W - 6 * pad) if why else [])
        bh = int(56 * s) + int(26 * s) * (len(rows) - 1)
        dr.rectangle([0, H - bh - int(52 * s), W, H - int(52 * s)], fill=(*(OK if ok else BAD), 225))
        dr.text((3 * pad, H - bh - int(52 * s) + int(10 * s)), msg, fill=(10, 15, 26), font=fb)
        for k, row in enumerate(rows[1:]):
            dr.text((3 * pad, H - bh - int(52 * s) + int(50 * s) + int(26 * s) * k), row, fill=(10, 15, 26), font=fc)
    return np.asarray(im)


def card(lines: list[str], size=SIZE, note: str = "") -> np.ndarray:
    W, H = size
    s = H / 720
    im = Image.new("RGB", size, BG)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, W, int(6 * s)], fill=ACCENT)
    ft, fl, fn = _font(int(46 * s), bold=True), _font(int(24 * s)), _font(int(18 * s))
    rows = [(x, ft, TEXT, int(60 * s)) for x in _wrap(dr, lines[0], ft, W * 0.84)]
    rows[-1] = (*rows[-1][:3], int(78 * s))
    for ln in lines[1:]:
        rows += [(x, fl, DIM, int(36 * s)) for x in _wrap(dr, ln, fl, W * 0.84, lines=3)]
    y = (H - sum(r[3] for r in rows)) // 2
    for text, f, color, h in rows:
        dr.text(((W - dr.textlength(text, font=f)) / 2, y), text, fill=color, font=f)
        y += h
    if note:
        dr.text(((W - dr.textlength(note, font=fn)) / 2, H - int(50 * s)), note, fill=(100, 116, 139), font=fn)
    return np.asarray(im)


def _writer(path: Path):
    import imageio.v2 as imageio
    return imageio.get_writer(path, fps=FPS, codec="libx264", macro_block_size=8, quality=7,
                              output_params=["-movflags", "+faststart"], ffmpeg_log_level="error")


def _fit_frame(img: np.ndarray, size=SIZE) -> np.ndarray:
    if img.shape[1] == size[0] and img.shape[0] == size[1]:
        return img
    im = Image.fromarray(img)
    k = min(size[0] / im.width, size[1] / im.height)
    im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))))
    out = Image.new("RGB", size, BG)
    out.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return np.asarray(out)


def _frames(model: mujoco.MjModel, tape: Tape, size=SIZE):
    g = model.vis.global_
    g.offwidth, g.offheight = max(g.offwidth, size[0]), max(g.offheight, size[1])
    d = mujoco.MjData(model)
    r = mujoco.Renderer(model, size[1], size[0])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    try:
        for k in range(len(tape)):
            d.qpos[:] = tape.qpos[k]
            if model.nmocap:
                d.mocap_pos[:], d.mocap_quat[:] = tape.mpos[k], tape.mquat[k]
            mujoco.mj_forward(model, d)
            c = tape.cam[k]
            cam.lookat[:] = c[:3]
            cam.distance, cam.azimuth, cam.elevation = float(c[3]), float(c[4]), float(c[5])
            r.update_scene(d, camera=cam)
            yield r.render()
    finally:
        r.close()


def _write(path: Path, frames) -> Path:
    """Write to a temp file and rename, so a partial file is never served."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{threading.get_ident()}.mp4")
    w = _writer(tmp)
    try:
        for f in frames:
            w.append_data(_fit_frame(f))
    except BaseException:
        w.close()
        tmp.unlink(missing_ok=True)
        raise
    w.close()
    tmp.replace(path)
    return path


def _played(model, tape: Tape, title: str, result: tuple[bool, str] | None, size=SIZE, clock: bool = True):
    last = None
    t0 = tape.t[0] if tape.t else 0.0
    for k, img in enumerate(_frames(model, tape, size)):
        last = img
        yield overlay(img, title, tape.captions[tape.cap[k]] if tape.captions else "", tape.t[k] - t0 if clock else None)
    if last is not None and result is not None:
        held = overlay(last, title, "", tape.t[-1] - t0 if clock else None, result)
        for _ in range(int(HOLD_S * FPS)):
            yield held


def render_try(npz: Path) -> Path:
    """npz -> mp4 next to it (cached)."""
    npz = Path(npz)
    out = npz.with_suffix(".mp4")
    with _lock_for(out):
        if out.exists() and out.stat().st_mtime >= npz.stat().st_mtime:
            return out
        tape, meta = Tape.load(npz)
        model = mujoco.MjModel.from_binary_path(str(npz.parent / meta["model"]))
        why = "; ".join(meta.get("reasons") or [])
        return _write(out, _played(model, tape, meta.get("title", npz.stem), (bool(meta.get("success")), why)))


# ------------------------------------------------------------------ learning videos
def _goal_of(task: dict):
    from .goals import Condition, GoalSpec
    from .user_scene import load_scene
    scene = load_scene(task["scene"])
    return GoalSpec(object=scene.object_name, conditions=[Condition(**c) for c in task.get("conditions", [])]), scene


def _policy_try(task: dict, policy, disturb=None, stop=None, label_steps=True) -> tuple:
    """Run the policy once from task["start"] and record it. disturb=(tick, vel) pushes the object at that tick.
    Returns (model, tape, (success, reason))."""
    from .goal_terms import History
    from .goals import check
    from .learn_task import rollout
    from .sim import PhysicalParams
    from .skill_steps import StepTracker
    from .task_model import CTRL_DT, build_task_model, contact_held, indices
    m = build_task_model(task, accurate=True)
    ix = indices(m, task)
    env = SimulationEnvironment(m, control_dt=CTRL_DT)
    tape = Tape()
    _taping(env, tape)
    steps = task.get("steps", [])

    def on_step(k, name):
        tape.caption = f"step {k} of {len(steps)} done: {name}"
    tracker = StepTracker(steps, task.get("fixtures", {}), on_step if label_steps else None)
    if label_steps and steps:
        tape.caption = f"step 1 of {len(steps)}: {steps[0]['name']}"
    watch: list = [tracker]
    goal = scene = None
    try:
        goal, scene = _goal_of(task)
        watch.append(History([c.model_dump() for c in goal.conditions], task.get("fixtures", {}),
                             (task["block"]["half_x"], task["block"]["half_y"], task["block"]["half_z"])))
    except Exception:  # noqa: BLE001  scene file missing, fall back to distance
        pass
    blew = rollout(env, m, ix, task, policy, task["start"], PhysicalParams(), 0, stop, tuple(watch), disturb=disturb)
    if stop is not None and stop():
        raise InterruptedError("stopped")
    if goal is not None:
        held = contact_held(m, env.data, ix) > 0.5
        r = check(goal, scene, m, env.data, env.block_pose(), env.block_speed(), not held, history=watch[1])
        ok, why = bool(r["success"]) and not blew, "; ".join(r["reasons"][:2])
    else:
        p, t = env.block_pose(), task["target"]
        dist = math.hypot(p.x - t["x"], p.y - t["y"])
        ok, why = dist < t.get("pos_tol", 0.03) and not blew, f"ended {dist * 100:.0f} cm from the target"
    if blew:
        why = "the simulation became unstable (it hit the object too hard)"
    if not ok and tracker.steps:
        why = (why + "; " if why else "") + tracker.summary()
    env.close()
    return m, tape, (ok, why)


def _load_round_task(folder: Path, file: Path) -> dict:
    from .task_model import load_task
    task = load_task(file)
    task["_dir"] = str(folder)  # demo.npz is in the main folder, not rounds/
    return task


def rounds_of(folder: Path) -> list[dict]:
    """Policies to show, in order. Uses rounds/ if learn_task saved it, otherwise falls back to the trainer's
    checkpoints (older runs, last round only)."""
    folder = Path(folder)
    out = []
    idx = folder / "rounds" / "index.json"
    if idx.exists():
        history = {}
        if (folder / "rounds" / "history.json").exists():
            for h in json.loads((folder / "rounds" / "history.json").read_text()):
                history.setdefault(int(h.get("round", -1)), h)
        for e in json.loads(idx.read_text()):
            pol = folder / "rounds" / f"{e['key']}.npz"
            if not pol.exists():
                continue
            h = history.get(int(e.get("round", -1)), {}) if not e["key"].endswith("_plan") else {}
            note = f"{e['success']}/{e['n']} succeeded" if e.get("n") else ""
            out.append({"key": e["key"], "label": e["label"], "note": note, "detail": e.get("note", ""),
                        "revision": h.get("revision", ""), "policy": pol, "task": folder / "rounds" / f"{e['key']}.task.json"})
        return out
    ck = sorted((folder / "checkpoints").glob("step_*.npz"))
    if len(ck) > 4:
        ck = [ck[round(k * (len(ck) - 1) / 3)] for k in range(4)]
    for f in ck:
        steps = int(f.stem.split("_")[1])
        out.append({"key": f"ckpt_{steps}", "label": f"Checkpoint at {steps / 1e6:.1f}M steps (last round)",
                    "note": "", "detail": "", "revision": "", "policy": f,
                    "task": folder / "task.json"})
    return out


def clips_of(folder: Path) -> list[dict]:
    folder = Path(folder)
    out = [{"clip": "progress", "label": "Learning progress (all clips)"}]
    out += [{"clip": r["key"], "label": r["label"]} for r in rounds_of(folder)]
    out += [{"clip": "final", "label": "Final policy"}]
    out += [{"clip": "compare", "label": "Program vs learned policy (knocked mid-task)" if _has_residual(folder)
             else "Before vs after"}]
    for c in out:
        c["made"] = _fresh(clip_path(folder, c["clip"]), folder)
    return out


def _has_residual(folder: Path) -> bool:
    try:
        t = json.loads((Path(folder) / "task.json").read_text())
        return bool(t.get("residual") and t.get("demo") and (Path(folder) / t["demo"]).exists())
    except (OSError, ValueError):
        return False


def _fresh(path: Path, folder: Path) -> bool:
    pol = Path(folder) / "policy.npz"
    return path.exists() and (not pol.exists() or path.stat().st_mtime >= pol.stat().st_mtime)


def clip_path(folder: Path, clip: str) -> Path:
    return Path(folder) / "videos" / ("learning_progress.mp4" if clip == "progress" else f"{_safe(clip)}.mp4")


def learning_clip(folder: Path, clip: str, stop=None) -> Path:
    from .learn_task import NumpyPolicy
    folder = Path(folder)
    out = clip_path(folder, clip)
    with _lock_for(out):
        if _fresh(out, folder):
            return out
        if clip == "progress":
            return _progress(folder, out, stop)
        if clip == "compare":
            return _compare(folder, out, stop)
        if clip == "final":
            task = _load_round_task(folder, folder / "task.json")
            m, tape, res = _policy_try(task, NumpyPolicy(folder / "policy.npz"), stop=stop)
            st = _status(folder)
            title = f"{task['robot']} · final policy" + (f" ({st['success']}/{st['n']} succeeded)"
                                                               if st.get("n") else "")
            return _write(out, _played(m, tape, title, res))
        r = next((r for r in rounds_of(folder) if r["key"] == clip), None)
        if r is None:
            raise ValueError(f"no round called {clip!r} here")
        task = _load_round_task(folder, r["task"])
        try:
            pol = NumpyPolicy(r["policy"])
            m, tape, res = _policy_try(task, pol, stop=stop)
        except InterruptedError:
            raise
        except Exception as e:  # noqa: BLE001  e.g. obs size changed
            raise ValueError(f"{r['label']}: this snapshot can't be played with the skill's current setup ({e})") from e
        title = f"{task['robot']} · {r['label']}" + (f" · {r['note']}" if r["note"] else "")
        return _write(out, _played(m, tape, title, res))


def _status(folder: Path) -> dict:
    try:
        return json.loads((Path(folder) / "status.json").read_text())
    except (OSError, ValueError):
        return {}


def _compare(folder: Path, out: Path, stop=None) -> Path:
    """Side by side from the same start. Residual skills: program alone vs learned, with the object knocked at 25%
    (same knocks as learn_task.robustness). Otherwise first round vs final."""
    from .learn_task import NumpyPolicy
    from .task_model import CTRL_DT, Residual
    task = _load_round_task(folder, folder / "task.json")
    res = Residual.of(task, folder)
    learned = NumpyPolicy(folder / "policy.npz")
    half = (SIZE[0] // 2, SIZE[1])
    if res is not None:
        # use the first knock (of 5) the learned policy recovers from; the title card says how many were tried
        rng = np.random.default_rng(7)
        tried, picked = 0, None
        for a in rng.uniform(0, 2 * math.pi, 5):
            bump = (int(0.25 * res.T), np.array([0.7 * math.cos(a), 0.7 * math.sin(a), 0.0]))
            tried += 1
            picked = (bump, _policy_try(task, learned, disturb=bump, stop=stop, label_steps=False))
            if picked[1][2][0]:
                break
        bump, right = picked
        left = _policy_try(task, lambda o: np.zeros(learned.act_dim), disturb=bump, stop=stop, label_steps=False)
        runs = [("Program only", left, bump), ("Learned policy", right, bump)]
        head = "Object knocked at 0.7 m/s, 25% into the task"
        (folder / "videos").mkdir(exist_ok=True)
        (folder / "videos" / "compare.json").write_text(json.dumps({"knocks_tried": tried, "recovered": bool(right[2][0])}))
    else:
        rs = rounds_of(folder)
        if not rs:
            raise ValueError("nothing from before practice was kept for this skill")
        first = rs[0]
        before = _policy_try(_load_round_task(folder, first["task"]), NumpyPolicy(first["policy"]), stop=stop,
                             label_steps=False)
        runs = [(f"Before: {first['label']}", before, None),
                ("After: the learned skill", _policy_try(task, learned, stop=stop, label_steps=False), None)]
        head = "Same start"
    sides = []
    for k, (title, (m, tape, r), bump) in enumerate(runs):
        if bump is not None:
            t_bump = tape.t[0] + bump[0] * CTRL_DT
            tape.captions.append("knocked!")
            for j, t in enumerate(tape.t):
                if t_bump <= t < t_bump + 1.5:
                    tape.cap[j] = len(tape.captions) - 1
        sides.append(_played(m, tape, title, r, half, clock=k == 1))

    def frames():
        a = b = None
        ft = _font(20, bold=True)
        while True:
            a2, b2 = next(sides[0], None), next(sides[1], None)
            if a2 is None and b2 is None:
                return
            a, b = a2 if a2 is not None else a, b2 if b2 is not None else b
            im = Image.fromarray(np.concatenate([a, np.full((SIZE[1], 4, 3), 10, np.uint8), b[:, 4:]], axis=1))
            dr = ImageDraw.Draw(im, "RGBA")
            w = dr.textlength(head, font=ft)
            dr.rounded_rectangle([(SIZE[0] - w) / 2 - 14, SIZE[1] - 46, (SIZE[0] + w) / 2 + 14, SIZE[1] - 12], radius=8,
                                 fill=(10, 15, 26, 200))
            dr.text(((SIZE[0] - w) / 2, SIZE[1] - 41), head, fill=TEXT, font=ft)
            yield np.asarray(im)
    return _write(out, frames())


def _clip_frames(path: Path):
    import imageio.v2 as imageio
    r = imageio.get_reader(path)
    try:
        for f in r:
            yield _fit_frame(f)
    finally:
        r.close()


def _hold(img: np.ndarray, seconds: float):
    for _ in range(int(seconds * FPS)):
        yield img


def _progress(folder: Path, out: Path, stop=None) -> Path:
    task = json.loads((folder / "task.json").read_text())
    st = _status(folder)
    rs = rounds_of(folder)
    parts = [(card([f"{task['robot']}: learning progress",
                    "Goal: " + task.get("steps_final", {}).get("how", "").split(": ", 1)[-1],
                    f"Final: {st['success']}/{st['n']} tries succeeded" if st.get("n") else ""],
                   note="Morph, MuJoCo"), 3.0, None)]
    for r in rs:
        sub = [x for x in (r["note"], r["detail"]) if x]
        if r["revision"]:
            sub.append("Nemotron's revision: " + r["revision"])
        try:
            parts.append((card([r["label"], *sub]), 2.4, learning_clip(folder, r["key"], stop)))
        except ValueError:
            pass
    parts.append((card(["Final policy", "best of all rounds"]), 2.0,
                  learning_clip(folder, "final", stop)))
    try:
        cmp_path = learning_clip(folder, "compare", stop)
        cmp_title = ["Before vs after", "same start"]
        if _has_residual(folder):
            k = json.loads((folder / "videos" / "compare.json").read_text()) if (folder / "videos" / "compare.json").exists() else {}
            cmp_title = ["Program vs learned policy", "object knocked at 0.7 m/s, 25% into the task"]
            if k.get("knocks_tried", 1) > 1:
                cmp_title.append(f"(knock {k['knocks_tried']} of the ones tried; the first it recovered from)"
                                 if k.get("recovered") else f"(failed all {k['knocks_tried']} knocks tried)")
        parts.append((card(cmp_title), 2.6, cmp_path))
    except ValueError:
        pass

    def frames():
        for img, secs, clip in parts:
            yield from _hold(img, secs)
            if clip is not None:
                yield from _clip_frames(clip)
    return _write(out, frames())


def learning_video(folder: Path, stop=None, log=print) -> Path:
    """Render all clips, then the progress video. Returns the progress video's path."""
    folder = Path(folder)
    for c in clips_of(folder):
        if c["clip"] == "progress":
            continue
        t0 = time.time()
        try:
            learning_clip(folder, c["clip"], stop)
            log(f"  video: {c['label']} ({time.time() - t0:.0f} s)")
        except ValueError as e:
            log(f"  video skipped: {e}")
    p = learning_clip(folder, "progress", stop)
    log(f"  video: {p}")
    return p


# ------------------------------------------------------------------ picker
def resolve(item: str, stop=None) -> tuple[Path, str]:
    """Picker item (try:<video id> or learn:<folder>:<clip>) -> (mp4 path, title)."""
    from .learn_task import LEARNED
    kind, _, rest = item.partition(":")
    if kind == "try":
        npz = from_id(rest)
        if npz.suffix != ".npz" or not npz.exists():
            raise ValueError(f"no such try: {rest}")
        meta = json.loads(str(np.load(npz)["meta"]))
        return render_try(npz), f"{meta.get('title', npz.stem)}: {'success' if meta.get('success') else 'failed'}"
    if kind == "learn":
        name, _, clip = rest.rpartition(":")
        folder = LEARNED / Path(name).name
        if not (folder / "task.json").exists():
            raise ValueError(f"no learned skill called {name}")
        label = next((c["label"] for c in clips_of(folder) if c["clip"] == clip), clip)
        return learning_clip(folder, clip, stop), f"{folder.name.split('__')[0]}: {label}"
    raise ValueError(f"unknown video {item!r}")


def bundle(items: list[str], as_zip: bool, out_dir: Path | None = None) -> Path:
    if not items:
        raise ValueError("pick at least one video")
    got = [resolve(i) for i in items]
    out_dir = Path(out_dir or VIDEOS / "bundles")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if as_zip:
        out = out_dir / f"morph_videos_{stamp}.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
            for k, (p, title) in enumerate(got, 1):
                z.write(p, f"{k:02d}_{_safe(title)[:80]}.mp4")
        return out
    out = out_dir / f"morph_videos_{stamp}.mp4"

    def frames():
        for k, (p, title) in enumerate(got, 1):
            if len(got) > 1:
                yield from _hold(card([title], note=f"{k} of {len(got)}"), 1.6)
            yield from _clip_frames(p)
    return _write(out, frames())


def listing(limit_runs: int = 12) -> dict:
    from .learn_task import LEARNED
    runs = []
    if VIDEOS.exists():
        for d in sorted((d for d in VIDEOS.iterdir() if (d / "index.json").exists()), reverse=True)[:limit_runs]:
            idx = json.loads((d / "index.json").read_text())
            tries = [t | {"item": f"try:{t['id']}", "made": (from_id(t["id"]).with_suffix(".mp4")).exists()}
                     for t in idx.get("tries", []) if t.get("id") and from_id(t["id"]).exists()]
            if tries:
                runs.append({"run": d.name, "made": idx.get("made", ""), "tries": tries})
    learned = []
    if LEARNED.exists():
        for f in sorted(LEARNED.iterdir(), key=lambda f: -f.stat().st_mtime):
            if (f / "policy.npz").exists() and (f / "task.json").exists():
                learned.append({"folder": f.name, "robot": f.name.split("__")[0],
                                "clips": [c | {"item": f"learn:{f.name}:{c['clip']}"} for c in clips_of(f)]})
    return {"runs": runs, "learned": learned}


def new_run_dir(kind: str, label: str = "") -> Path:
    return VIDEOS / f"{time.strftime('%Y%m%d-%H%M%S')}_{kind}{'_' + _safe(label)[:40] if label else ''}"


def cleanup_tmp() -> None:
    """Remove temp files left by an interrupted render."""
    for p in list(VIDEOS.rglob(".*.mp4")) + list((RUNS / "learned").glob("*/videos/.*.mp4")):
        p.unlink(missing_ok=True)

