"""A user's own scene: the object to move, plus the static things around it.

The scene is any MJCF file. The object is the body with a free joint (or the one named in the
project). Every other body without joints, and loose world geoms other than the floor, becomes a
fixture: its geoms are copied into Morph's scene in world coordinates, so every robot sees them.

Fixtures are classified from their geometry, not their names:
  container  walls around an open interior (a bin, a tray): things go *inside* it
  surface    a top you can rest something on (a platform, a table, a shelf)
  obstacle   anything else (a wall, a post): things can go *against* it

Limits: the object is modelled as its bounding box (exact for box objects); mesh fixtures are
replaced by boxes of the same extent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .geometry import yaw_from_mat
from .scene import BlockSpec

GEOM_NAMES = {int(mujoco.mjtGeom.mjGEOM_BOX): "box", int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
              int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule", int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
              int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid"}
PLANE, MESH = int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_MESH)


@dataclass
class Fixture:
    name: str
    kind: str  # container | surface | obstacle
    lo: np.ndarray  # world AABB
    hi: np.ndarray
    top_z: float  # highest point
    rest_z: float  # height something placed at the centre would rest on (container floor / surface top)
    inner_lo: np.ndarray | None = None  # container interior (xy), inside the walls
    inner_hi: np.ndarray | None = None
    geoms: list = field(default_factory=list)  # (type, size, pos, quat) in world
    top_lo: np.ndarray | None = None  # the largest flat face at the very top (xy), e.g. a shelf top, a wall's
    top_hi: np.ndarray | None = None  # 1 cm edge, one side of a bin's rim

    @property
    def top_size(self) -> tuple[float, float]:
        d = self.top_hi - self.top_lo
        return float(d[0]), float(d[1])

    def geometry(self) -> dict:
        """Plain measurements for reasoning about requests ("on top of", "in", "next to")."""
        sx, sy, sz = (self.hi - self.lo) * 100
        tw, td = (v * 100 for v in self.top_size)
        g = {"kind": self.kind, "footprint_cm": [round(sx, 1), round(sy, 1)], "height_cm": round(sz, 1),
             "centre_xy_m": [round(float(self.lo[0] + self.hi[0]) / 2, 3), round(float(self.lo[1] + self.hi[1]) / 2, 3)],
             "top_face_cm": [round(tw, 1), round(td, 1)],
             "extent_m": {"x": [round(float(self.lo[0]), 3), round(float(self.hi[0]), 3)],
                          "y": [round(float(self.lo[1]), 3), round(float(self.hi[1]), 3)]}}
        if self.kind == "container":
            ix, iy = (self.inner_hi - self.inner_lo) * 100
            g |= {"opening_cm": [round(ix, 1), round(iy, 1)], "inside_floor_cm": round(self.rest_z * 100, 1)}
        return g

    def describe(self) -> str:
        sx, sy = (self.hi - self.lo)[:2] * 100
        s = f"{self.name}: {self.kind}, {sx:.0f} x {sy:.0f} cm, top at {self.top_z * 100:.1f} cm, centre ({(self.lo[0] + self.hi[0]) / 2:.2f}, {(self.lo[1] + self.hi[1]) / 2:.2f})"
        if self.kind == "container":
            ix, iy = (self.inner_hi - self.inner_lo) * 100
            s += f", opening {ix:.0f} x {iy:.0f} cm, floor at {self.rest_z * 100:.1f} cm"
        return s


@dataclass
class UserScene:
    path: str
    object_name: str
    block: BlockSpec  # the object as Morph simulates it, with the fixtures attached
    start: tuple[float, float, float]  # object x, y, yaw at the start
    fixtures: dict[str, Fixture]
    notes: list[str] = field(default_factory=list)

    def geometry(self) -> dict:
        """Everything a planner needs to reason about a request: the object's size and each fixture's shape."""
        b = self.block
        return {"object": {"name": self.object_name, "size_cm": [round(2 * b.half_x * 100, 1), round(2 * b.half_y * 100, 1),
                                                                  round(2 * b.half_z * 100, 1)],
                           "shape": b.shape, "start_xy_m": [round(self.start[0], 3), round(self.start[1], 3)],
                           "start_heading_deg": round(math.degrees(self.start[2]), 1)},
                "fixtures": {n: f.geometry() for n, f in self.fixtures.items()},
                "frame": "metres; x east, y north, z up; heading 0 = east, 90 = north; the floor is z = 0"}

    def summary(self) -> dict:
        b = self.block
        return {"object": {"name": self.object_name, "size_cm": [round(2 * b.half_x * 100, 1), round(2 * b.half_y * 100, 1),
                                                                  round(2 * b.half_z * 100, 1)],
                           "shape": b.shape, "mass_kg": b.mass, "start_xy": [round(self.start[0], 3), round(self.start[1], 3)],
                           "start_yaw_deg": round(math.degrees(self.start[2]), 1)},
                "fixtures": {n: f.describe() for n, f in self.fixtures.items()}, "notes": self.notes}


def _geom_aabb(m: mujoco.MjModel, d: mujoco.MjData, g: int) -> tuple[np.ndarray, np.ndarray]:
    R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
    if m.geom_type[g] == MESH:
        v = m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:m.mesh_vertadr[m.geom_dataid[g]] + m.mesh_vertnum[m.geom_dataid[g]]]
        w = v @ R.T + p
        return w.min(0), w.max(0)
    s = m.geom_size[g]
    t = m.geom_type[g]
    if t == int(mujoco.mjtGeom.mjGEOM_BOX):
        half = s[:3]
    elif t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        half = np.array([s[0]] * 3)
    elif t == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        half = np.array([s[0], s[0], s[1] + s[0]])
    elif t == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        half = np.array([s[0], s[0], s[1]])
    else:
        half = s[:3]
    ext = np.abs(R) @ half
    return p - ext, p + ext


def _classify(name: str, boxes: list[tuple[np.ndarray, np.ndarray]]) -> Fixture:
    f = _classify_kind(name, boxes)
    # top face: of the geoms reaching the very top, the one with the largest flat area
    tops = [b for b in boxes if b[1][2] >= f.hi[2] - 0.005]
    best = max(tops, key=lambda b: (b[1][0] - b[0][0]) * (b[1][1] - b[0][1]))
    f.top_lo, f.top_hi = best[0][:2].copy(), best[1][:2].copy()
    return f


def _classify_kind(name: str, boxes: list[tuple[np.ndarray, np.ndarray]]) -> Fixture:
    lo = np.min([b[0] for b in boxes], 0)
    hi = np.max([b[1] for b in boxes], 0)
    c = (lo + hi) / 2
    covers = [b[0][0] <= c[0] <= b[1][0] and b[0][1] <= c[1] <= b[1][1] for b in boxes]
    under = [b for b, k in zip(boxes, covers) if k]
    walls = [b for b, k in zip(boxes, covers) if not k]
    centre_top = max((b[1][2] for b in under), default=0.0)
    if walls:
        wall_top = max(b[1][2] for b in walls)
        # interior: the xy region bounded by the walls' inner faces on each side of the centre
        inner_lo, inner_hi = lo[:2].copy(), hi[:2].copy()
        sides = 0
        for b in walls:
            if b[1][0] <= c[0]:
                inner_lo[0] = max(inner_lo[0], b[1][0]); sides |= 1
            elif b[0][0] >= c[0]:
                inner_hi[0] = min(inner_hi[0], b[0][0]); sides |= 2
            if b[1][1] <= c[1]:
                inner_lo[1] = max(inner_lo[1], b[1][1]); sides |= 4
            elif b[0][1] >= c[1]:
                inner_hi[1] = min(inner_hi[1], b[0][1]); sides |= 8
        if bin(sides).count("1") >= 3 and wall_top > centre_top + 0.01:
            return Fixture(name, "container", lo, hi, float(hi[2]), float(centre_top), inner_lo, inner_hi)
    if under and centre_top > 0.005 and (hi[0] - lo[0]) > 0.04 and (hi[1] - lo[1]) > 0.04:
        return Fixture(name, "surface", lo, hi, float(hi[2]), float(centre_top))
    return Fixture(name, "obstacle", lo, hi, float(hi[2]), 0.0)


def load_scene(path: str | Path, object_name: str | None = None) -> UserScene:
    path = Path(path)
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    notes = []
    free_bodies = [m.jnt_bodyid[j] for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if object_name:
        ob = m.body(object_name).id
    elif len(free_bodies) == 1:
        ob = free_bodies[0]
    else:
        names = [m.body(b).name for b in free_bodies]
        raise ValueError(f"which object should move? the scene has {len(free_bodies)} free bodies: {names}; "
                         "set `object` in morph.toml")
    obj_name = m.body(ob).name

    # object: its collision geoms' extent in its own frame
    og = [g for g in range(m.ngeom) if m.geom_bodyid[g] == ob and (m.geom_contype[g] or m.geom_conaffinity[g])]
    if not og:
        raise ValueError(f"object {obj_name} has no collision geoms")
    Rb, pb = d.xmat[ob].reshape(3, 3), d.xpos[ob]
    pts = []
    for g in og:
        glo, ghi = _geom_aabb(m, d, g)
        corners = np.array([[x, y, z] for x in (glo[0], ghi[0]) for y in (glo[1], ghi[1]) for z in (glo[2], ghi[2])])
        pts.append((corners - pb) @ Rb)
    pts = np.vstack(pts)
    half = (pts.max(0) - pts.min(0)) / 2
    shape = "box"
    if len(og) == 1:  # a single sphere, upright cylinder or capsule is simulated as itself
        t, sz = int(m.geom_type[og[0]]), m.geom_size[og[0]]
        upright = abs((d.geom_xmat[og[0]].reshape(3, 3))[2, 2]) > 0.99
        if t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            shape, half = "sphere", np.array([sz[0]] * 3)
        elif t == int(mujoco.mjtGeom.mjGEOM_CYLINDER) and upright:
            shape, half = "cylinder", np.array([sz[0], sz[0], sz[1]])
        elif t == int(mujoco.mjtGeom.mjGEOM_CAPSULE) and upright:
            shape, half = "capsule", np.array([sz[0], sz[0], sz[1] + sz[0]])
    if shape == "box" and not (len(og) == 1 and m.geom_type[og[0]] == int(mujoco.mjtGeom.mjGEOM_BOX)):
        notes.append(f"object {obj_name} is modelled as its {2 * half[0] * 100:.1f} x {2 * half[1] * 100:.1f} x "
                     f"{2 * half[2] * 100:.1f} cm bounding box")
    block_mass = float(m.body_subtreemass[ob])
    friction = float(m.geom_friction[og[0]][0])
    start = (float(pb[0]), float(pb[1]), yaw_from_mat(Rb))

    # fixtures: static bodies (no joints anywhere up their chain) and loose world geoms
    def static(b):
        while b != 0:
            if m.body_jntnum[b] > 0:
                return False
            b = m.body_parentid[b]
        return True

    groups: dict[str, list[int]] = {}
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        if m.geom_type[g] == PLANE or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        if b == 0:
            groups.setdefault(m.geom(g).name or f"fixture{g}", []).append(g)
        elif static(b):
            top = b
            while m.body_parentid[top] != 0:
                top = m.body_parentid[top]
            groups.setdefault(m.body(top).name or f"fixture{top}", []).append(g)

    fixtures, xml = {}, []
    for name, gs in groups.items():
        boxes = [_geom_aabb(m, d, g) for g in gs]
        f = _classify(name, boxes)
        body = [f'\n    <body name="{name}">']
        for g, (glo, ghi) in zip(gs, boxes):
            t = int(m.geom_type[g])
            rgba = " ".join(f"{v:.3f}" for v in m.geom_rgba[g])
            fr = " ".join(f"{v:g}" for v in m.geom_friction[g])
            if t in GEOM_NAMES:
                size = " ".join(f"{v:.5f}" for v in m.geom_size[g][:3])
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, d.geom_xmat[g])
                pos = " ".join(f"{v:.5f}" for v in d.geom_xpos[g])
                q = " ".join(f"{v:.6f}" for v in quat)
                body.append(f'      <geom type="{GEOM_NAMES[t]}" size="{size}" pos="{pos}" quat="{q}" rgba="{rgba}" friction="{fr}"/>')
            else:
                c, h = (glo + ghi) / 2, (ghi - glo) / 2
                body.append(f'      <geom type="box" size="{h[0]:.5f} {h[1]:.5f} {h[2]:.5f}" pos="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}" '
                            f'rgba="{rgba}" friction="{fr}"/>')
                notes.append(f"{name}: a mesh geom was replaced by its bounding box")
            f.geoms.append((t, m.geom_size[g].copy(), d.geom_xpos[g].copy()))
        body.append("    </body>")
        xml.append("\n".join(body))
        fixtures[name] = f

    block = BlockSpec(half_x=float(half[0]), half_y=float(half[1]), half_z=float(half[2]), mass=block_mass,
                      friction=friction, fixtures_xml="".join(xml), shape=shape)
    return UserScene(str(path), obj_name, block, start, fixtures, notes)
