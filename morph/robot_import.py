"""Bring robots in from the formats developers actually have.

MJCF (.xml) is used as is. Xacro (.xacro / .urdf.xacro, how most ROS packages ship their robot) is
expanded to URDF first, with `$(find pkg)` resolved from the package.xml files around it. URDF (.urdf, what ROS robots and most manufacturers publish) is converted:
  * meshes: `package://pkg/...` and relative paths are found on disk (next to the URDF or in the
    folders above it) and gathered into one folder, since MuJoCo looks them up by file name
  * motors: a URDF has none that MuJoCo reads, so every moving joint gets a position servo whose
    force limit is the URDF's <limit effort=...> (stiffness scaled from it)
  * base: URDFs have no free joint. If the robot stands on several feet (two or more chains ending at
    its lowest point, well below its root) it gets a floating base, like a legged robot needs;
    otherwise it stays bolted down, like an arm.
The result is written next to Morph's run files as an ordinary MJCF that the rest of Morph uses.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONVERTED = ROOT / "runs" / "converted"


def _folder(src: Path) -> str:
    """One output folder per source file (two robots may both be called robot.urdf)."""
    return f"{re.sub(r'[^A-Za-z0-9_-]+', '_', src.name)}_{hashlib.sha1(str(src).encode()).hexdigest()[:6]}"


def _find_mesh(ref: str, urdf: Path, pkgs: dict | None = None) -> Path | None:
    """package://pkg/meshes/x.stl or meshes/x.stl -> a file on disk."""
    m = re.match(r"^package://([^/]+)/(.*)$", ref)
    if m and pkgs and m.group(1) in pkgs and (pkgs[m.group(1)] / m.group(2)).exists():
        return pkgs[m.group(1)] / m.group(2)
    rel = re.sub(r"^package://[^/]+/", "", ref)
    rel = re.sub(r"^file://", "", rel)
    if Path(rel).is_absolute() and Path(rel).exists():
        return Path(rel)
    for base in [urdf.parent, *urdf.parents][:5]:
        for cand in (base / rel, base / Path(rel).name):
            if cand.exists():
                return cand
        hits = list(base.rglob(Path(rel).name))
        if hits:
            return hits[0]
    return None


def convert_urdf(urdf_path: str | Path, floating: bool | None = None,
                 origin: str | Path | None = None) -> tuple[Path, list[str]]:
    """origin: where the robot really came from (the .xacro), for finding its meshes and packages."""
    urdf = Path(urdf_path).resolve()
    home = Path(origin).resolve() if origin else urdf
    pkgs = _packages_near(home)
    name = urdf.stem
    out = CONVERTED / _folder(home)
    meshes = out / "meshes"
    meshes.mkdir(parents=True, exist_ok=True)
    notes = []
    tree = ET.parse(urdf)
    robot = tree.getroot()
    missing = []
    unsupported = []
    for m in robot.iter("mesh"):
        ref = m.get("filename", "")
        f = _find_mesh(ref, home, pkgs)
        if f is None:
            missing.append(ref)
            continue
        if f.suffix.lower() not in (".stl", ".obj", ".msh"):  # MuJoCo reads STL / OBJ (not DAE, the ROS visual mesh)
            unsupported.append(f.name)
            m.set("filename", f.name)
            continue
        dst = meshes / f.name
        if not dst.exists():
            shutil.copy(f, dst)
        m.set("filename", f.name)
    if unsupported:
        notes.append(f"{len(unsupported)} visual mesh(es) in a format MuJoCo can't read (e.g. {unsupported[0]}): "
                     "those are left out; collision shapes are kept")
    if missing:
        notes.append(f"{len(missing)} mesh file(s) not found, those parts are left out: {', '.join(missing[:3])}")
    if missing or unsupported:
        for link in robot.iter("link"):  # drop visuals/collisions whose mesh is missing
            for tag in ("visual", "collision"):
                for el in list(link.findall(tag)):
                    mm = el.find("geometry/mesh")
                    if mm is not None and not (meshes / mm.get("filename", "")).exists():
                        link.remove(el)
    ext = robot.find("mujoco")
    if ext is None:
        ext = ET.SubElement(robot, "mujoco")
    comp = ext.find("compiler")
    if comp is None:
        comp = ET.SubElement(ext, "compiler")
    comp.set("meshdir", str(meshes))
    comp.set("balanceinertia", "true")
    comp.set("discardvisual", "false")
    fixed = out / f"{name}_fixed.urdf"
    tree.write(fixed)
    m0 = mujoco.MjSpec.from_file(str(fixed)).compile()
    if floating is None:
        floating = _looks_legged(m0)
    if floating:
        # MuJoCo folds a URDF's root link into the world. Give it a parent and a URDF "floating" joint instead,
        # raised so the feet start on the floor.
        d0 = mujoco.MjData(m0)
        mujoco.mj_forward(m0, d0)
        lowest = min((d0.geom_xpos[g][2] - m0.geom_rbound[g] for g in range(m0.ngeom)), default=0.0)
        children = {j.find("child").get("link") for j in robot.iter("joint")}
        root = next(link.get("name") for link in robot.iter("link") if link.get("name") not in children)
        ET.SubElement(robot, "link", name="morph_world")
        j = ET.SubElement(robot, "joint", name="morph_floating_base", type="floating")
        ET.SubElement(j, "parent", link="morph_world")
        ET.SubElement(j, "child", link=root)
        ET.SubElement(j, "origin", xyz=f"0 0 {max(0.0, -lowest) + 0.005:.4f}", rpy="0 0 0")
        tree.write(fixed)
        notes.append("stands on several feet: given a floating base")
    spec = mujoco.MjSpec.from_file(str(fixed))
    m0 = spec.compile()
    effort = {}
    for j in robot.iter("joint"):
        lim = j.find("limit")
        if j.get("type") in ("revolute", "prismatic", "continuous") and lim is not None and lim.get("effort"):
            effort[j.get("name")] = float(lim.get("effort"))

    for jn in [m0.joint(i).name for i in range(m0.njnt)]:
        j = spec.joint(jn)
        if j.type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        tau = effort.get(jn, 30.0)
        kp = max(5.0, 4.0 * tau) if j.type == mujoco.mjtJoint.mjJNT_HINGE else max(50.0, 40.0 * tau)
        act = spec.add_actuator(name=jn, target=jn, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.gaintype, act.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = kp
        act.biasprm[0], act.biasprm[1], act.biasprm[2] = 0.0, -kp, -0.1 * kp ** 0.5
        if j.range[1] > j.range[0]:
            act.ctrllimited, act.ctrlrange = True, list(j.range)
        act.forcelimited, act.forcerange = True, [-tau, tau]
    if not effort:
        notes.append("the URDF gives no joint effort limits: motors assume 30 Nm")
    spec.meshdir = str(meshes)
    xml_path = out / f"{name}.xml"
    xml_path.write_text(spec.to_xml())
    mujoco.MjModel.from_xml_path(str(xml_path))  # it must load on its own
    notes.append(f"converted from URDF: {sum(1 for _ in spec.actuators)} position motors added")
    return xml_path, notes


def _scan(base: Path, depth: int, found: dict) -> None:
    for root, dirs, files in os.walk(base):
        rel = len(Path(root).relative_to(base).parts)
        dirs[:] = [] if rel >= depth else [d for d in dirs if not d.startswith(".")]
        if "package.xml" in files:
            m = re.search(r"<name>\s*([^<\s]+)\s*</name>", (Path(root) / "package.xml").read_text(errors="ignore"))
            if m:
                found.setdefault(m.group(1), Path(root))


def _packages_near(path: Path) -> dict[str, Path]:
    """ROS packages near a robot file: the package it's in and its neighbours in the same workspace
    (a package's parent folder, e.g. ws/src), or any packages just below the file's folder."""
    found = {}
    for up in [path.parent, *path.parents][:5]:
        if (up / "package.xml").exists():
            _scan(up.parent, 3, found)
            break
    _scan(path.parent, 3, found)
    return found


def expand_xacro(path: str | Path, args: dict | None = None) -> tuple[Path, list[str]]:
    """.xacro -> plain URDF. Uses the ROS `xacro` tool if it's installed, else the pip package.
    `$(find pkg)` needs the package's location: Morph builds a tiny ament index pointing at every
    package.xml near the file, added to whatever ROS environment is already set up."""
    src = Path(path).resolve()
    name = re.sub(r"(\.urdf)?\.xacro$", "", src.name)
    out = CONVERTED / _folder(src)
    out.mkdir(parents=True, exist_ok=True)
    index = out / "ament_index"
    shutil.rmtree(index, ignore_errors=True)
    (index / "share" / "ament_index" / "resource_index" / "packages").mkdir(parents=True)
    pkgs = _packages_near(src)
    for pkg, d in pkgs.items():
        (index / "share" / "ament_index" / "resource_index" / "packages" / pkg).write_text("")
        (index / "share" / pkg).symlink_to(d, target_is_directory=True)
    env = dict(os.environ)
    env["AMENT_PREFIX_PATH"] = os.pathsep.join(filter(None, [str(index), env.get("AMENT_PREFIX_PATH")]))
    env["ROS_PACKAGE_PATH"] = os.pathsep.join(filter(None, [*map(str, {d.parent for d in pkgs.values()}),
                                                            env.get("ROS_PACKAGE_PATH")]))
    urdf = out / f"{name}.urdf"
    extra = [f"{k}:={v}" for k, v in (args or {}).items()]
    tool = shutil.which("xacro") or next((p for p in ("/opt/ros/jazzy/bin/xacro", "/opt/ros/humble/bin/xacro")
                                          if Path(p).exists()), None)
    cmds = ([[tool, str(src), "-o", str(urdf), *extra]] if tool else []) + \
           [[sys.executable, "-m", "xacro", str(src), "-o", str(urdf), *extra]]
    err = ""
    for cmd in cmds:
        if cmd[0] == tool and "/opt/ros/" in tool and "PYTHONPATH" not in env:
            ros = Path(tool).parents[1]  # the ROS tool runs on the ROS Python packages
            env["PYTHONPATH"] = str(next(ros.glob("lib/python3*/site-packages"), ""))
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(src.parent), timeout=120)
        if r.returncode == 0 and urdf.exists():
            notes = [f"expanded xacro ({len(pkgs)} ROS package{'s' if len(pkgs) != 1 else ''} found nearby)"]
            return urdf, notes
        err = (r.stderr or r.stdout).strip().splitlines()[-1:] or ["unknown error"]
        err = err[0]
    if "No module named xacro" in err and not tool:
        raise ValueError("this robot is a .xacro file: install xacro (pip install xacro, or source your ROS setup) "
                         "or give Morph the expanded URDF")
    raise ValueError(f"couldn't expand the xacro file: {err}")


def _looks_legged(m: mujoco.MjModel) -> bool:
    """Several branch ends (feet) at the robot's lowest point, well below its root: it stands on legs."""
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    low = {}
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        low[b] = min(low.get(b, np.inf), d.geom_xpos[g][2] - m.geom_rbound[g])
    if not low:
        return False
    parents = {int(m.body_parentid[b]) for b in range(1, m.nbody)}
    ends = [b for b in range(1, m.nbody) if b not in parents]
    # a foot may be a geom-less end body (a site holder): use its nearest ancestor with geoms
    def lowest(b):
        while b and b not in low:
            b = int(m.body_parentid[b])
        return low.get(b, np.inf)
    lo = min(low.values())
    feet = {b for b in ends if lowest(b) < lo + 0.03}
    return len(feet) >= 2 and d.xpos[1][2] - lo > 0.1


def import_robot(path: str | Path) -> tuple[Path, list[str]]:
    """Any supported robot file -> an MJCF path Morph can use, plus notes on what was done."""
    p = Path(path)
    if p.suffix.lower() == ".xacro":
        urdf, notes = expand_xacro(p)
        xml, more = convert_urdf(urdf, origin=p)
        return xml, notes + more
    if p.suffix.lower() == ".urdf":
        return convert_urdf(p)
    if p.suffix.lower() == ".xml":
        return p, []
    raise ValueError(f"{p.suffix} robots aren't supported yet (use MuJoCo .xml, URDF .urdf or xacro .xacro)")
