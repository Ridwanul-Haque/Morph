"""Still pictures for the app: the scene alone, or a robot standing in it, before anything runs.

Where the robot stands: wherever the user put it (placement x, y, heading), else Morph's guess: a fixed
arm behind the work area at about half its reach (a run then finds the actual mount by experiment);
anything with a floating base (legged, wheeled, flying) a body length or so back from the object, clear
of the fixtures, in the pose of its model's keyframe.

Turning the view must feel instant, so a scene is built once and kept, and all drawing happens on one
long-lived thread that keeps its OpenGL renderers (making a renderer takes far longer than drawing).
"""

from __future__ import annotations

import io
import json
import math
import threading
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from ..introspect import home_qpos, profile_robot
from ..scene import FLOOR_FRICTION, _block_xml, _common_assets, _goal_xml
from ..sim import PhysicalParams, SimulationEnvironment, apply_view, remember, view_adjusted

_LOCK = threading.Lock()  # one scene build at a time
_BUILT: dict = {}  # (scene, robot, goal, placement) -> (model, data, robot base x/y/heading, fixed?)


class RenderService:
    """One thread that does all the app's drawing outside a run, keeping its renderers between frames."""

    def __init__(self):
        import queue
        self.q: "queue.Queue" = queue.Queue()
        self.renderers: dict = {}
        threading.Thread(target=self._loop, daemon=True, name="morph-render").start()

    def _loop(self):
        while True:
            fn, box, done = self.q.get()
            try:
                box["out"] = fn(self._renderer)
            except Exception as e:  # noqa: BLE001
                box["err"] = e
            done.set()

    def _renderer(self, m: mujoco.MjModel, w: int, h: int) -> mujoco.Renderer:
        key = (id(m), w, h)
        if key not in self.renderers:
            if len(self.renderers) >= 4:  # keep a few (preview sizes, the last run's scene)
                old = next(iter(self.renderers))
                self.renderers.pop(old)[1].close()
            g = m.vis.global_
            g.offwidth, g.offheight = max(g.offwidth, w), max(g.offheight, h)
            self.renderers[key] = (m, mujoco.Renderer(m, h, w))  # m kept alive with its renderer
        return self.renderers[key][1]

    def draw(self, m, d, camera, size) -> np.ndarray:
        def job(get):
            r = get(m, size[0], size[1])
            r.update_scene(d, camera=camera)
            return r.render().copy()
        box, done = {}, threading.Event()
        self.q.put((job, box, done))
        done.wait(30)
        if "err" in box:
            raise box["err"]
        return box["out"]


_SERVICE: RenderService | None = None


def service() -> RenderService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = RenderService()
    return _SERVICE


def jpeg(img: np.ndarray, draft: bool = False) -> bytes:
    b = io.BytesIO()
    Image.fromarray(img).save(b, "JPEG", quality=72 if draft else 90, subsampling=2 if draft else 0)
    return b.getvalue()


def _scene_spec(block) -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(f"""<mujoco model="morph_preview">
  <compiler angle="radian" autolimits="true"/>
  {_common_assets(block)}
  <worldbody>
    <light pos="0.6 -0.5 2.6" dir="-0.15 0.2 -1" cutoff="45" exponent="2" castshadow="true" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" friction="{FLOOR_FRICTION} 0.005 0.0001"/>
    {_goal_xml()}
    {_block_xml(block)}
    <frame name="mount"/>
  </worldbody>
</mujoco>""")


def _framed_camera(m: mujoco.MjModel, d: mujoco.MjData, view=None) -> mujoco.MjvCamera:
    """Camera that fits everything but the floor (robot, object, fixtures), seen from the front-left."""
    pts = []
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or m.geom_rgba[g][3] == 0 or d.geom_xpos[g][2] < -0.02:
            continue
        r = float(m.geom_rbound[g]) if m.geom_rbound[g] > 0 else 0.02
        pts += [d.geom_xpos[g] - min(r, 0.3), d.geom_xpos[g] + min(r, 0.3)]
    pts = np.array(pts)
    lo, hi = pts.min(0), pts.max(0)
    lo[2] = 0.0
    c = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo)) / 2
    cam = mujoco.MjvCamera()
    cam.lookat[:] = c
    cam.azimuth, cam.elevation = 150.0, -24.0
    cam.distance = max(0.8, radius / math.sin(math.radians(m.vis.global_.fovy / 2)) * 0.85)
    apply_view(cam, view)
    remember(cam)
    return cam


def auto_placement(scene, robot_path: str, goal_xy_yaw=None) -> tuple[float, float, float]:
    """Morph's guess at where the robot stands (x, y, heading), before anyone moves it."""
    prof = profile_robot(robot_path)
    rm = mujoco.MjModel.from_xml_path(str(robot_path))
    free = any(rm.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in range(rm.njnt))
    ox, oy, _ = scene.start
    tx, ty = (goal_xy_yaw[0], goal_xy_yaw[1]) if goal_xy_yaw else (ox, oy)
    cx, cy = (ox + tx) / 2, (oy + ty) / 2
    if not free:  # fixed arm: behind the work area at ~half its reach, facing it
        reach = max((v["max_horizontal"] for v in prof.reach_m.values()), default=0.6)
        return (cx - 0.55 * reach, cy, 0.0)
    length = max(prof.size_m.get("length", 0.5), 0.3)  # mobile: back from the object, away from the goal
    d = np.array([tx - ox, ty - oy])
    ang = math.atan2(-d[1], -d[0]) if np.linalg.norm(d) > 1e-3 else -math.pi / 2
    xy, a = np.array([ox, oy]), ang
    for k in range(12):
        a = ang + (k + 1) // 2 * (0.35 if k % 2 else -0.35)
        xy = np.array([ox, oy]) + (0.3 + 0.6 * length) * np.array([math.cos(a), math.sin(a)])
        if all(not (f.lo[0] - 0.25 < xy[0] < f.hi[0] + 0.25 and f.lo[1] - 0.25 < xy[1] < f.hi[1] + 0.25)
               for f in scene.fixtures.values()):
            break
    return (float(xy[0]), float(xy[1]), a + math.pi)


def _build(scene, robot_path, goal_xy_yaw, placement):
    """The preview scene (model + data), built once per scene / robot / goal / placement."""
    key = (scene.path, hash(scene.block.fixtures_xml), robot_path, goal_xy_yaw, placement)
    if key in _BUILT:
        return _BUILT[key]
    with _LOCK:
        spec = _scene_spec(scene.block)
        ox, oy, _ = scene.start
        base, free, rm, q0 = None, False, None, None
        if robot_path:
            robot = mujoco.MjSpec.from_file(str(robot_path))
            rm = robot.compile()
            q0 = home_qpos(rm)
            root_id = [b for b in range(1, rm.nbody) if rm.body_parentid[b] == 0][0]
            free = any(rm.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in range(rm.njnt))
            base = tuple(placement) if placement else auto_placement(scene, robot_path, goal_xy_yaw)
            spec.option.impratio, spec.option.cone = robot.option.impratio, robot.option.cone
            spec.option.integrator = robot.option.integrator
            for k in ("timestep", "density", "viscosity", "noslip_iterations", "iterations", "ls_iterations", "tolerance",
                      "solver"):  # a picture only: the scene's settings, without MuJoCo warning about each one
                setattr(robot.option, k, getattr(spec.option, k))
            frame = spec.worldbody.frames[0]
            if not free:
                from ..introspect import floor_offset
                frame.pos = [base[0], base[1], floor_offset(robot_path)]
                frame.quat = [math.cos(base[2] / 2), 0, 0, math.sin(base[2] / 2)]
            frame.attach_body(robot.body(rm.body(root_id).name), "robot_", "")
        m = spec.compile()
        env = SimulationEnvironment(m)
        env.reset(0, scene.start, PhysicalParams())
        if goal_xy_yaw:
            env.set_goal_marker(*goal_xy_yaw)
        else:
            env.data.mocap_pos[env._goal_mocap] = [ox, oy, -0.05]  # no goal yet: tuck the marker under the floor
        if robot_path:
            # the robot's own keyframe pose, joint by joint (names were prefixed on attach)
            for j in range(rm.njnt):
                name = rm.joint(j).name
                jj = m.joint("robot_" + name).id if name else None
                if jj is None:
                    continue
                a, b = m.jnt_qposadr[jj], rm.jnt_qposadr[j]
                n = 7 if rm.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 4 if rm.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL else 1
                env.data.qpos[a:a + n] = q0[b:b + n]
            if free:  # place the floating base
                fj = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                      and m.body(m.jnt_bodyid[j]).name.startswith("robot_")][0]
                a = m.jnt_qposadr[fj]
                env.data.qpos[a:a + 3] = [base[0], base[1], env.data.qpos[a + 2]]
                env.data.qpos[a + 3:a + 7] = [math.cos(base[2] / 2), 0, 0, math.sin(base[2] / 2)]
        mujoco.mj_forward(m, env.data)
        built = (m, env.data, base, not free)
        _BUILT[key] = built
        if len(_BUILT) > 12:
            _BUILT.pop(next(iter(_BUILT)))
        return built


def render_preview(scene, robot_path: str | None, goal_xy_yaw=None, size=(1280, 800), view=None,
                   placement=None, draft: bool = False) -> bytes:
    """JPEG of the scene (and robot). `scene` is a morph.user_scene.UserScene. draft: half size, quicker
    (while the user is dragging the view or the robot)."""
    m, d, _, _ = _build(scene, robot_path, goal_xy_yaw, tuple(placement) if placement else None)
    if draft:
        size = (max(320, size[0] // 2) // 8 * 8, max(200, size[1] // 2) // 8 * 8)
    return jpeg(service().draw(m, d, _framed_camera(m, d, view), size), draft)


def placement_of(scene, robot_path: str, goal_xy_yaw=None, placement=None) -> dict:
    """Where the robot stands in the preview (the user's placement, or Morph's guess)."""
    _, _, base, fixed = _build(scene, robot_path, goal_xy_yaw, tuple(placement) if placement else None)
    return {"x": round(base[0], 3), "y": round(base[1], 3), "yaw_deg": round(math.degrees(base[2]), 1), "fixed": fixed,
            "custom": placement is not None}


def floor_point(u: float, v: float, aspect: float, fovy: float = 45.0) -> tuple[float, float] | None:
    """Where on the floor a point of the last picture shows (u, v from 0 to 1 across and down it), using
    the camera that drew it. None if that point is sky."""
    from ..sim import LAST_CAM
    az, el = math.radians(LAST_CAM["az"]), math.radians(LAST_CAM["el"])
    f = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    r = np.array([math.sin(az), -math.cos(az), 0.0])
    up = np.cross(r, f)
    pos = np.array(LAST_CAM.get("lookat", [0.4, 0.0, 0.1])) - LAST_CAM["dist"] * f
    th = math.tan(math.radians(fovy) / 2)
    ray = f + (2 * u - 1) * th * aspect * r + (1 - 2 * v) * th * up
    if ray[2] >= -1e-6:
        return None
    t = -pos[2] / ray[2]
    p = pos + t * ray
    return float(p[0]), float(p[1])
