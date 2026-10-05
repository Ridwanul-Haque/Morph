"""Thin, deterministic wrapper around one MuJoCo scene.

Reads simulator ground truth directly; there's no perception model.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

from .geometry import quat_from_yaw, quat_to_mat, tilt_from_mat, yaw_from_mat


_BAD_WARNINGS = (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQVEL,
                 mujoco.mjtWarning.mjWARN_BADQPOS)


class SimulationUnstable(RuntimeError):
    """MuJoCo reported NaN/Inf; don't score the episode."""


@dataclass(frozen=True)
class PhysicalParams:
    block_mass: float = 0.2
    block_friction: float = 0.6

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class BlockPose:
    x: float
    y: float
    z: float
    yaw: float
    tilt: float

    def as_dict(self) -> dict:
        return {k: round(float(v), 5) for k, v in asdict(self).items()}


@dataclass
class ContactEvent:
    time: float
    geom: str
    body: str
    normal_force: float


@dataclass
class StepLog:
    """What happened during one control interval (several physics steps)."""

    max_step_disp: float = 0.0  # largest block displacement in a single physics step (m)
    robot_contacts: list[ContactEvent] = field(default_factory=list)


class SimulationEnvironment:
    def __init__(self, model: mujoco.MjModel, control_dt: float = 0.02, robot_prefix_bodies=None):
        self.model = model
        self.data = mujoco.MjData(model)
        self.control_dt = control_dt
        self.n_sub = max(1, int(round(control_dt / model.opt.timestep)))
        self._block_body = model.body("block").id
        self._block_geom = model.geom("block_geom").id
        self._floor_geom = model.geom("floor").id
        jnt = model.joint("block_free")
        self._bq = jnt.qposadr[0]
        self._bv = jnt.dofadr[0]
        self._goal_mocap = model.body("goal").mocapid[0]
        self._renderer: mujoco.Renderer | None = None
        self.track_azimuth = 150.0
        self.track_extra = None  # callable -> robot xy to keep in frame
        self.params = PhysicalParams()
        self.seed = 0

    # ------------------------------------------------------------------ setup
    def reset(self, seed: int, block_xy_yaw: tuple[float, float, float], params: PhysicalParams,
              robot_qpos: dict[str, float] | None = None) -> None:
        self.seed = seed
        self.params = params
        mujoco.mj_resetData(self.model, self.data)
        self.model.body_mass[self._block_body] = params.block_mass
        self.model.geom_friction[self._block_geom, 0] = params.block_friction
        # inertia must follow mass, for the object's own shape
        m = params.block_mass
        size, kind = self.model.geom_size[self._block_geom], int(self.model.geom_type[self._block_geom])
        if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            r = size[0]
            self.model.body_inertia[self._block_body] = [0.4 * m * r * r] * 3
            hz = r
        elif kind in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
            r, h = size[0], 2 * size[1] + (2 * size[0] if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE) else 0.0)
            side = m * (3 * r * r + h * h) / 12
            self.model.body_inertia[self._block_body] = [side, side, 0.5 * m * r * r]
            hz = h / 2
        else:
            sx, sy, sz = (2 * size)
            self.model.body_inertia[self._block_body] = [
                m / 12 * (sy**2 + sz**2), m / 12 * (sx**2 + sz**2), m / 12 * (sx**2 + sy**2)]
            hz = size[2]
        x, y, yaw = block_xy_yaw
        self.data.qpos[self._bq:self._bq + 3] = [x, y, hz]
        self.data.qpos[self._bq + 3:self._bq + 7] = quat_from_yaw(yaw)
        if robot_qpos:
            for name, v in robot_qpos.items():
                self.data.qpos[self.model.joint(name).qposadr[0]] = v
        mujoco.mj_forward(self.model, self.data)

    def set_goal_marker(self, x: float, y: float, yaw: float) -> None:
        self.data.mocap_pos[self._goal_mocap] = [x, y, 0.0005]
        self.data.mocap_quat[self._goal_mocap] = quat_from_yaw(yaw)

    # ---------------------------------------------------------------- stepping
    def advance(self, ctrl: np.ndarray | None = None, ctrl_fn=None) -> StepLog:
        """Apply `ctrl` (or call ctrl_fn(data) every physics step) for one control interval."""
        log = StepLog()
        if ctrl is not None:
            self.data.ctrl[:] = ctrl
        prev = self.data.qpos[self._bq:self._bq + 3].copy()
        for _ in range(self.n_sub):
            if ctrl_fn is not None:
                ctrl_fn(self.data)
            mujoco.mj_step(self.model, self.data)
            if any(self.data.warning[w].number > 0 for w in _BAD_WARNINGS):
                raise SimulationUnstable(f"MuJoCo instability at t={self.data.time:.3f}s (seed {self.seed})")
            cur = self.data.qpos[self._bq:self._bq + 3]
            log.max_step_disp = max(log.max_step_disp, float(np.linalg.norm(cur - prev)))
            prev = cur.copy()
            log.robot_contacts.extend(self._block_robot_contacts())
        return log

    def _block_robot_contacts(self) -> list[ContactEvent]:
        out = []
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if self._block_geom not in (g1, g2):
                continue
            other = g2 if g1 == self._block_geom else g1
            if other == self._floor_geom:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, f6)
            out.append(ContactEvent(
                time=float(self.data.time),
                geom=self.model.geom(other).name or f"geom{other}",
                body=self.model.body(self.model.geom_bodyid[other]).name,
                normal_force=float(abs(f6[0]))))
        return out

    # ------------------------------------------------------------- observation
    def block_pose(self) -> BlockPose:
        q = self.data.qpos[self._bq:self._bq + 7]
        R = quat_to_mat(q[3:7])
        return BlockPose(float(q[0]), float(q[1]), float(q[2]), yaw_from_mat(R), tilt_from_mat(R))

    def block_speed(self) -> float:
        return float(np.linalg.norm(self.data.qvel[self._bv:self._bv + 6]))

    def site_pos(self, name: str) -> np.ndarray:
        return self.data.site(name).xpos.copy()

    def snapshot(self) -> dict:
        return {
            "time": float(self.data.time),
            "qpos": self.data.qpos.tolist(),
            "qvel": self.data.qvel.tolist(),
            "ctrl": self.data.ctrl.tolist(),
        }

    def restore(self, snap: dict) -> None:
        self.data.time = snap["time"]
        self.data.qpos[:] = snap["qpos"]
        self.data.qvel[:] = snap["qvel"]
        self.data.ctrl[:] = snap["ctrl"]
        mujoco.mj_forward(self.model, self.data)

    # --------------------------------------------------------------- rendering
    def render(self, camera: str = "overview", width: int = 640, height: int = 480, view: dict | None = None) -> np.ndarray:
        """view: the user's own adjustment on top of the chosen camera (see apply_view)."""
        if self._renderer is None or self._renderer.width != width or self._renderer.height != height:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height, width)
        self._renderer.update_scene(self.data, camera=self.camera(camera, view))
        return self._renderer.render()

    def camera(self, camera: str = "overview", view: dict | None = None):
        """The camera to draw with: a scene camera's name, or a free camera (the tracking one, or any camera
        the user has turned)."""
        if camera == "track":  # free camera framing the block, the goal marker and (optionally) the robot
            cam = mujoco.MjvCamera()
            pts = [self.data.qpos[self._bq:self._bq + 2].copy(), self.data.mocap_pos[self._goal_mocap][:2].copy()]
            if self.track_extra is not None:
                pts.append(np.asarray(self.track_extra())[:2])
            pts = np.array(pts)
            c = pts.mean(0)
            spread = float(np.max(np.linalg.norm(pts - c, axis=1)))
            cam.lookat[:] = [c[0] - 0.05, c[1], 0.12]
            cam.distance = max(1.1, 1.0 + 2.2 * spread)
            cam.azimuth, cam.elevation = self.track_azimuth, -28
            apply_view(cam, view)
            remember(cam)
            return cam
        if view_adjusted(view):
            cam = camera_as_free(self.model, self.data, camera)
            apply_view(cam, view)
            remember(cam)
            return cam
        if view is not None:  # so a first drag from this camera knows which way it faces
            remember(camera_as_free(self.model, self.data, camera))
        return camera

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


DEFAULT_VIEW = {"az": 0.0, "el": 0.0, "zoom": 1.0, "pan": [0.0, 0.0, 0.0]}
LAST_CAM = {"az": 150.0, "el": -28.0, "dist": 1.5}  # the last camera shown (for turning a pan into world units)


def remember(cam: mujoco.MjvCamera) -> None:
    LAST_CAM.update(az=float(cam.azimuth), el=float(cam.elevation), dist=float(cam.distance),
                    lookat=[float(v) for v in cam.lookat])


def nudge_view(view: dict, d: dict) -> dict:
    """Add one drag / wheel step to the user's view. d: daz, del (degrees), zoom (factor), pan (dx, dy as
    fractions of the viewer: the scene follows the mouse)."""
    v = {"az": view["az"], "el": view["el"], "zoom": view["zoom"], "pan": list(view["pan"])}
    v["az"] = (v["az"] + float(d.get("daz", 0)) + 180) % 360 - 180
    base_el = LAST_CAM["el"] - view["el"]  # the chosen camera's own tilt
    v["el"] = float(np.clip(v["el"] + float(d.get("del", 0)), -89 - base_el, -2 - base_el))
    v["zoom"] = float(np.clip(v["zoom"] * float(d.get("zoom", 1)), 0.08, 8.0))
    dx, dy = (float(x) for x in d.get("pan", (0, 0)))
    if dx or dy:
        az, el = np.radians(LAST_CAM["az"]), np.radians(LAST_CAM["el"])
        f = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        r = np.array([np.sin(az), -np.cos(az), 0.0])
        u = np.cross(r, f)
        k = LAST_CAM["dist"] * 0.83  # the view's width at the look-at point (45 degree field of view)
        v["pan"] = [float(x) for x in np.asarray(v["pan"]) - dx * k * r + dy * k * u]
    return v


def view_adjusted(view: dict | None) -> bool:
    return bool(view) and (abs(view["az"]) > 1e-6 or abs(view["el"]) > 1e-6 or abs(view["zoom"] - 1) > 1e-6
                           or any(abs(v) > 1e-6 for v in view["pan"]))


def apply_view(cam: mujoco.MjvCamera, view: dict | None) -> None:
    """The user's drag / zoom / pan on top of a free camera: turn it, tilt it, move closer, shift what
    it looks at. The camera stays above the floor."""
    if not view_adjusted(view):
        return
    cam.azimuth = cam.azimuth + view["az"]
    cam.elevation = float(np.clip(cam.elevation + view["el"], -89.0, -2.0))
    cam.distance = float(np.clip(cam.distance * view["zoom"], 0.15, 20.0))
    cam.lookat[:] = np.asarray(cam.lookat) + np.asarray(view["pan"])


def camera_as_free(m: mujoco.MjModel, d: mujoco.MjData, name: str) -> mujoco.MjvCamera:
    """A scene camera (e.g. "overview", "top") as an orbit camera, so the user can turn it."""
    cam = mujoco.MjvCamera()
    try:
        i = m.camera(name).id
    except KeyError:
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = [0.4, 0.0, 0.1], 2.0, 150.0, -28.0
        return cam
    pos, fwd = d.cam_xpos[i].copy(), -d.cam_xmat[i].reshape(3, 3)[:, 2]
    t = (0.1 - pos[2]) / fwd[2] if fwd[2] < -1e-3 else 1.5  # where it looks at the work height
    cam.lookat[:] = pos + fwd * t
    cam.distance = float(t)
    cam.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
    cam.elevation = float(np.degrees(np.arcsin(np.clip(fwd[2], -1, 1))))
    return cam


def settle(env: SimulationEnvironment, seconds: float, ctrl=None, ctrl_fn=None) -> None:
    for _ in range(max(1, int(math.ceil(seconds / env.control_dt)))):
        env.advance(ctrl, ctrl_fn=ctrl_fn)
