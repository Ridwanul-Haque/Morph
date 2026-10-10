"""Bring an object in from its 3D model (STL / OBJ, as CAD tools export it): Morph works out its units, lays it in its
most stable resting pose (longest side along x, shortest up), centres it, gives it a mass from its volume, and writes
a scene with it in the work cell (the example's bin, shelf and wall), ready to use like any other scene.

    morph object path/to/Pen.STL            # -> runs/app_uploads/scenes/pen/scene.xml
    (or Upload in the app's scene picker, with an .stl / .obj)

The physics collides with the object's convex hull (its outline with any hollows filled in: a mug's inside and the gap
under its handle are solid), which keeps it fast enough for the GPU learner.
"""

from __future__ import annotations

import math
import re
import shutil
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UNITS = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254}


def guess_units(extent: float) -> str:
    """The largest side, in the file's own numbers -> the units it was most likely saved in. CAD tools (SolidWorks,
    Fusion, Onshape) export millimetres by default; an object Morph can handle is a few cm to about a metre."""
    if extent > 30:  # a 30 m object is unlikely: millimetres (a 105 mm pen, a 300 mm box)
        return "mm"
    if extent > 2.5:  # 2.5..30: centimetres (or inches; say so with units=)
        return "cm"
    return "m"


def _mesh_model(path: Path, scale: float) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_string(f"""<mujoco><asset><mesh name="m" file="{path}" scale="{scale} {scale} {scale}"/></asset>
<worldbody><body><freejoint/><geom type="mesh" mesh="m" density="1000"/></body></worldbody></mujoco>""")
    return spec.compile()


def rest_pose(m: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where the mesh goes in the object's frame so it rests flat: (geom pos, geom quat, half extents). MuJoCo places a
    mesh by its original file coordinates relative to the geom frame; its principal axes (mesh_quat) are turned so the
    longest side lies along x and the shortest stands up (the most stable way to lie), and the box around it is centred."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.mesh_quat[0])
    R = R.reshape(3, 3)  # principal (mesh-local) -> file frame
    v = m.mesh_vert[: m.mesh_vertnum[0]]  # in the principal frame, about the centre of mass
    ext = v.max(0) - v.min(0)
    order = np.argsort(-ext)  # longest, middle, shortest -> x, y, z
    S = np.zeros((3, 3))
    for row, col in enumerate(order):
        S[row, col] = 1.0
    if np.linalg.det(S) < 0:  # keep it a rotation (not a mirror image)
        S[1] *= -1
    A = S @ R.T  # file frame -> object frame
    vb = v @ S.T
    c = (vb.max(0) + vb.min(0)) / 2  # the box around it, centred on the object's origin
    pos = -(A @ m.mesh_pos[0]) - c
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, A.reshape(-1))
    half = (vb.max(0) - vb.min(0)) / 2
    return pos, q, half


def object_scene(mesh: str | Path, out: Path | None = None, name: str | None = None, units: str | None = None,
                 mass: float | None = None, friction: float = 0.6, base: str | Path | None = None) -> dict:
    """A scene (scene.xml + morph.toml) with this object in the work cell. Returns what was decided: units, size,
    mass, and the folder."""
    mesh = Path(mesh)
    if mesh.suffix.lower() not in (".stl", ".obj"):
        raise ValueError(f"{mesh.suffix} isn't a mesh Morph can read: export it as STL or OBJ (in SolidWorks: File > Save "
                         "As > STL)")
    name = re.sub(r"[^a-z0-9_]+", "_", (name or mesh.stem).lower()).strip("_") or "object"
    raw = _mesh_model(mesh, 1.0)
    rv = raw.mesh_vert[: raw.mesh_vertnum[0]]
    extent = float((rv.max(0) - rv.min(0)).max())
    units = units or guess_units(extent)
    if units not in UNITS:
        raise ValueError(f"units must be one of {', '.join(UNITS)}")
    k = UNITS[units]
    m = _mesh_model(mesh, k)
    pos, q, half = rest_pose(m)
    volume_mass = float(m.body_mass[1])  # at 1000 kg/m3 (water; most plastics are 900..1400)
    mass = float(mass) if mass else max(volume_mass, 0.005)
    out = Path(out or ROOT / "runs" / "app_uploads" / "scenes" / name)
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    dst = out / "meshes" / mesh.name
    if dst.resolve() != mesh.resolve():
        shutil.copy(mesh, dst)
    base = Path(base or ROOT / "examples" / "parcel" / "scene.xml")
    text = base.read_text()
    body = (f'<body name="{name}" pos="0.45 -0.18 {half[2]:.5f}">\n      <freejoint/>\n'
            f'      <geom type="mesh" mesh="{name}_mesh" pos="{pos[0]:.6g} {pos[1]:.6g} {pos[2]:.6g}" '
            f'quat="{q[0]:.6g} {q[1]:.6g} {q[2]:.6g} {q[3]:.6g}" mass="{mass:.5g}" friction="{friction} 0.005 0.0001" '
            f'rgba="0.9 0.55 0.2 1"/>\n    </body>')
    text, n = re.subn(r'<body name="parcel".*?</body>', body, text, count=1, flags=re.S)
    if not n:
        raise ValueError("the base scene has no parcel to replace")
    text = text.replace("<worldbody>", f'<asset>\n    <mesh name="{name}_mesh" file="meshes/{mesh.name}" scale="{k} {k} {k}"/>\n'
                                       "  </asset>\n  <worldbody>", 1)
    text = text.replace("a parcel on the floor", f"a {name} on the floor (from {mesh.name})")
    (out / "scene.xml").write_text(text)
    (out / "morph.toml").write_text(f'[project]\nname = "{name}"\nscene = "scene.xml"\nobject = "{name}"\n')
    size = [round(2 * float(h) * 100, 1) for h in half]
    return {"path": str(out), "object": name, "units": units, "size_cm": size, "mass_kg": round(mass, 4),
            "mass_from": "given" if k and mass and volume_mass != mass else "its volume, at the density of water",
            "note": (f"read {mesh.name} as {dict(mm='millimetres', cm='centimetres', m='metres', **{'in': 'inches'})[units]}: "
                     f"{size[0]} × {size[1]} × {size[2]} cm lying down, {mass * 1000:.0f} g (from its volume at the "
                     f"density of water; change it if it's heavier or lighter)")}
