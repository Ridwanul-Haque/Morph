import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

from morph.goal_sources import goal_from_description_rules
from morph.learn_task import export, make_task, try_policy
from morph.project import load_project
from morph.task_model import build_task_model, indices, load_task, obs_size

ROOT = Path(__file__).resolve().parents[1]


def _random_policy(folder: Path, n_obs: int, n_act: int, seed=0):
    rng = np.random.default_rng(seed)
    sizes = [n_obs, 64, 64, 2 * n_act]
    arrays = {"obs_mean": np.zeros(n_obs), "obs_std": np.ones(n_obs)}
    for i in range(3):
        arrays[f"W{i}"] = rng.normal(0, 0.1, (sizes[i], sizes[i + 1]))
        arrays[f"b{i}"] = np.zeros(sizes[i + 1])
    np.savez(folder / "policy.npz", **arrays)


def test_learn_try_and_export_run_end_to_end(tmp_path, monkeypatch):
    import morph.learn_task as lt
    monkeypatch.setattr(lt, "LEARNED", tmp_path)
    pj = load_project(ROOT / "examples" / "parcel")
    goal = goal_from_description_rules("push the parcel against the wall", pj.scene)
    task_path = make_task(pj, goal, str(ROOT / "tests" / "data" / "tiny_arm.urdf"))
    task = load_task(task_path)
    assert not task["floating"] and task["target"]["lift"] is False
    m = build_task_model(task)
    ix = indices(m, task)
    _random_policy(task_path.parent, obs_size(ix), m.nu)
    res = try_policy(pj, goal, task_path, n=1, log=lambda s: None)
    assert res["n"] == 1  # a random policy won't succeed; the point is that it runs and is checked
    zp = export(task_path)
    out = tmp_path / "unzipped"
    zipfile.ZipFile(zp).extractall(out)
    assert {"policy.npz", "policy.json", "model.xml", "run_policy.py", "README.txt"} <= {p.name for p in out.iterdir()}
    r = subprocess.run([sys.executable, "run_policy.py"], cwd=out, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "object ended at" in r.stdout, r.stderr[-800:]


import importlib.util  # noqa: E402

import mujoco  # noqa: E402
import pytest  # noqa: E402

from morph.task_model import observe  # noqa: E402


@pytest.mark.parametrize("robot, origin", [("tiny_arm.urdf", (0.0, 0.0, 0.0)), ("tiny_dog.urdf", (1.0, -2.0, 0.7))])
def test_ros2_package_rebuilds_the_training_observation(tmp_path, monkeypatch, robot, origin):
    """The ROS node sees only topics (poses in the robot's base frame, or in an odom frame for a mobile
    robot); from those it must rebuild exactly what the policy saw in training."""
    import morph.learn_task as lt
    monkeypatch.setattr(lt, "LEARNED", tmp_path)
    pj = load_project(ROOT / "examples" / "parcel")
    goal = goal_from_description_rules("push the parcel against the wall", pj.scene)
    task_path = make_task(pj, goal, str(ROOT / "tests" / "data" / robot))
    task = load_task(task_path)
    m0 = build_task_model(task)
    _random_policy(task_path.parent, obs_size(indices(m0, task)), m0.nu)
    export(task_path)
    pkg = next((task_path.parent / "export" / "ros2").iterdir())
    for f in ("package.xml", "setup.py", "launch/policy.launch.py", "config/params.yaml", f"{pkg.name}/policy_node.py",
              f"{pkg.name}/sim_bridge.py", "policy/model.xml", "policy/policy.npz"):
        assert (pkg / f).exists(), f
    spec = importlib.util.spec_from_file_location("policy_core", pkg / pkg.name / "policy_core.py")
    core_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core_mod)
    core = core_mod.MorphPolicy(pkg / "policy", scene_origin=origin)
    assert core.floating == task["floating"]
    out = core.scene_to_input()  # scene frame -> what the robot's topics would carry
    m = mujoco.MjModel.from_xml_path(str(pkg / "policy" / "model.xml"))
    d = mujoco.MjData(m)
    d.qpos[:] = core.spec["initial_qpos"]
    ix = indices(m, task)
    rng = np.random.default_rng(1)
    for _ in range(30):  # move things around a bit so velocities aren't zero
        d.ctrl[:] = np.clip(ix.q0 + rng.normal(0, 0.3, m.nu), ix.act_lo, ix.act_hi)
        for _ in range(10):
            mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)  # after mj_step, xpos and actuator lengths still describe the state before the step
    b, bv = ix.block, ix.block_v
    kw = {}
    if core.floating:
        Rb = d.xmat[ix.base].reshape(3, 3)
        kw = dict(base_pos=out.point(d.xpos[ix.base]), base_quat=out.quat(d.xquat[ix.base]),
                  base_lin_vel=Rb.T @ d.qvel[ix.base_v:ix.base_v + 3], base_ang_vel=d.qvel[ix.base_v + 3:ix.base_v + 6])
    core.set_state(d.qpos[ix.home_adr], d.qvel[ix.joint_vadr], out.point(d.xpos[b]), out.quat(d.xquat[b]),
                   out.vector(d.qvel[bv:bv + 3]), **kw)
    t = task["target"]
    want = observe(np, d.qvel, d.actuator_length, d.actuator_velocity, d.xpos, d.xmat, np.zeros(m.nu), ix, (t["x"], t["y"], t["z"], t["yaw"]))
    np.testing.assert_allclose(core.observation(), want, atol=1e-6)
