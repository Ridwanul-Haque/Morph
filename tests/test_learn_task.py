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


def test_several_step_plans_are_tried_and_the_best_is_kept(tmp_path, monkeypatch):
    """Eureka-style search: Nemotron proposes plans, each practises briefly, the best on one yardstick wins."""
    import json as _json
    import morph.learn_task as lt
    from morph.reasoning import StepCandidates
    monkeypatch.setattr(lt, "LEARNED", tmp_path)
    pj = load_project(ROOT / "examples" / "parcel")
    goal = goal_from_description_rules("push the parcel against the wall", pj.scene)
    task_path = make_task(pj, goal, str(ROOT / "tests" / "data" / "tiny_arm.urdf"))
    m = build_task_model(load_task(task_path))
    n_obs, n_act = obs_size(indices(m, load_task(task_path))), m.nu
    _random_policy(task_path.parent, n_obs, n_act)
    (task_path.parent / "params.pkl").write_bytes(b"x")

    class FakeLLM:
        available = True

        def step_candidates(self, *a, **k):
            mk = lambda name, f: {"name": name, "done_when": f}
            return StepCandidates(candidates=[{"idea": "touch it first", "steps": [mk("touch", "touch_dist < 0.05")]},
                                              {"idea": "get behind it", "steps": [mk("behind", "hand_x < x - 0.03")]}],
                                  rationale="stuck at step 1"), None

    scores = {"keep the current steps": 1.0, "touch it first": 9.0, "get behind it": 4.0}

    def fake_train(tp, n, restore):  # a policy, and a training evaluation that scores each plan differently
        t = _json.loads(tp.read_text())
        idea = {"touch_dist < 0.05": "touch it first", "hand_x < x - 0.03": "get behind it"}.get(
            (t.get("steps") or [{}])[0].get("done_when"), "keep the current steps")
        _random_policy(tp.parent, n_obs, n_act)
        (tp.parent / "train_meta.json").write_text(_json.dumps({"curve": [{"success": scores[idea], "dist": 1.0}]}))
        (tp.parent / "params.pkl").write_bytes(idea.encode())
        return True

    chosen = lt.search_step_plans(pj, goal, task_path, fake_train, FakeLLM(), {"stuck_at": {}}, [{"round": 1}], 1000,
                                  tries=1, k=2, log=lambda s: None)
    assert chosen["idea"] == "touch it first" and chosen["tried"] == 3
    assert load_task(task_path)["steps"][0]["done_when"] == "touch_dist < 0.05"
    assert (task_path.parent / "params.pkl").read_bytes() == b"touch it first"  # the winner's policy continues
    assert not list(task_path.parent.glob("candidate_*"))


def test_corrections_on_a_recipe_replay_it_and_export_the_same(tmp_path, monkeypatch):
    """Residual learning: a policy that outputs nothing replays the recipe exactly; the ROS 2 core and Morph's own
    loop give the same commands step by step (recipe + scale x correction, with the recipe's phase observed)."""
    import morph.learn_task as lt
    from morph.learn_task import NumpyPolicy, demo_to_task
    from morph.task_model import CTRL_DT, Residual, next_ctrl
    monkeypatch.setattr(lt, "LEARNED", tmp_path)
    pj = load_project(ROOT / "examples" / "parcel")
    goal = goal_from_description_rules("push the parcel against the wall", pj.scene)
    task_path = make_task(pj, goal, str(ROOT / "tests" / "data" / "tiny_arm.urdf"), plan=False)
    task = load_task(task_path)
    m = build_task_model(task)
    ix = indices(m, task)
    d = mujoco.MjData(m)
    d.qpos[ix.home_adr] = ix.home_val
    mujoco.mj_forward(m, d)
    rng = np.random.default_rng(0)
    demo, c = [], np.clip(ix.q0, ix.act_lo, ix.act_hi)
    for k in range(60):  # a "recipe" recorded on the task's own model: a smooth wander of the joints
        c = np.clip(c + rng.normal(0, 0.02, m.nu), ix.act_lo, ix.act_hi)
        d.ctrl[:] = c
        for _ in range(round(CTRL_DT / m.opt.timestep)):
            mujoco.mj_step(m, d)
        demo.append((d.qpos.copy(), d.qvel.copy(), c.copy()))
    info = demo_to_task(task_path, None, [demo])
    assert info["residual"]
    task = load_task(task_path)
    res = Residual.of(task, task_path.parent)
    assert res is not None and res.T == 60
    z = np.load(task_path.parent / "demo.npz")
    assert np.allclose(z["act"], 0.0)  # along the recipe itself there's nothing to correct
    _random_policy(task_path.parent, obs_size(indices(m, task)) + 1, m.nu)
    export(task_path)
    pkg = next((task_path.parent / "export" / "ros2").iterdir())
    assert (pkg / "policy" / "recipe.npy").exists()
    spec = importlib.util.spec_from_file_location("policy_core", pkg / pkg.name / "policy_core.py")
    core_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core_mod)
    core = core_mod.MorphPolicy(pkg / "policy")
    out = core.scene_to_input()
    me = mujoco.MjModel.from_xml_path(str(pkg / "policy" / "model.xml"))
    de = mujoco.MjData(me)
    de.qpos[:] = core.spec["initial_qpos"]
    mujoco.mj_forward(me, de)
    ix = indices(me, task)
    pol = NumpyPolicy(task_path.parent / "policy.npz")
    t = task["target"]
    tgt = (t["x"], t["y"], t["z"], t["yaw"])
    last, ctrl = np.zeros(me.nu), np.clip(ix.q0, ix.act_lo, ix.act_hi)
    core.ctrl = ctrl.copy()  # the same first command on both sides
    for k in range(25):
        b, bv = ix.block, ix.block_v
        core.set_state(de.qpos[ix.home_adr], de.qvel[ix.joint_vadr], out.point(de.xpos[b]), out.quat(de.xquat[b]),
                       out.vector(de.qvel[bv:bv + 3]))
        theirs = core.step()
        o = np.concatenate([observe(np, de.qvel, de.actuator_length, de.actuator_velocity, de.xpos, de.xmat, last, ix, tgt),
                            res.phase(np, k)])
        last = pol(o)
        ctrl = next_ctrl(np, ctrl, res.action(np, k, last), ix.q0, ix.act_scale, ix.act_lo, ix.act_hi, ix.act_step)
        np.testing.assert_allclose(theirs, ctrl, atol=1e-6)
        de.ctrl[:] = ctrl
        for _ in range(round(CTRL_DT / me.opt.timestep)):
            mujoco.mj_step(me, de)
        mujoco.mj_forward(me, de)
    zp = task_path.parent / "export.zip"
    unz = tmp_path / "unz"
    zipfile.ZipFile(zp).extractall(unz)
    r = subprocess.run([sys.executable, "run_policy.py"], cwd=unz, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "object ended at" in r.stdout, r.stderr[-800:]


def test_what_the_person_watching_says_reaches_nemotron_and_changes_the_plan(tmp_path, monkeypatch):
    """A suggestion typed during a round goes to Nemotron with the measurements at the end of it; the revised steps
    are what the next round practises, and the round's record says what was said."""
    import json as _json
    import morph.learn_task as lt
    from morph.reasoning import SkillStepsProposal
    monkeypatch.setattr(lt, "LEARNED", tmp_path)
    pj = load_project(ROOT / "examples" / "parcel")
    goal = goal_from_description_rules("push the parcel against the wall", pj.scene)
    task_path = make_task(pj, goal, str(ROOT / "tests" / "data" / "tiny_arm.urdf"))
    m = build_task_model(load_task(task_path))
    n_obs, n_act = obs_size(indices(m, load_task(task_path))), m.nu
    heard = []

    class FakeLLM:
        available = True

        def revise_steps(self, *a, advice=None, **k):
            heard.append(advice)
            return SkillStepsProposal(steps=[{"name": "get the tip low behind it", "done_when": "touch_dist < 0.03"}],
                                      reward_terms=[{"name": "low", "formula": "-abs(hand_z - z)", "weight": 1}],
                                      rationale="You said: push it lower. Added a step that gets the tip low first."), None

    def fake_train(tp, n, restore):
        _random_policy(tp.parent, n_obs, n_act)
        (tp.parent / "train_meta.json").write_text(_json.dumps({"curve": [{"success": 0.0, "dist": 1.0}]}))
        return True

    said = [["it pushes the top and tips it, push it lower"]]
    events = []
    res = lt.learn_in_rounds(pj, goal, task_path, fake_train, llm=FakeLLM(), log=lambda s: None,
                             emit=lambda **e: events.append(e), total_steps=10_000_000, rounds=2, tries=1, candidates=0,
                             advice=lambda: said.pop() if said else [])
    assert heard == [["it pushes the top and tips it, push it lower"]]
    t = load_task(task_path)
    assert t["steps"][0]["done_when"] == "touch_dist < 0.03" and t["reward_terms"][0]["name"] == "low"
    assert res["rounds"][0]["person_said"] == ["it pushes the top and tips it, push it lower"]
    assert any(e.get("stage") == "advice" and e["rationale"].startswith("You said") for e in events)


def test_only_a_finished_skill_that_works_is_used(tmp_path):
    """A policy on disk isn't a learned skill: learning must have finished and succeeded in at least half its tries."""
    import json as _json
    from morph.learn_task import learning_status
    d = tmp_path / "robot__goal"
    d.mkdir()
    assert learning_status(d)["state"] == "none"
    (d / "task.json").write_text("{}")
    (d / "policy.npz").write_bytes(b"x")
    (d / "status.json").write_text(_json.dumps({"state": "learning"}))
    assert not learning_status(d)["good"]  # stopped part-way
    (d / "status.json").write_text(_json.dumps({"state": "done", "success": 2, "n": 10}))
    assert not learning_status(d)["good"]  # finished, but it doesn't work
    (d / "status.json").write_text(_json.dumps({"state": "done", "success": 6, "n": 10}))
    assert learning_status(d)["good"]
    (d / "status.json").unlink()  # older folders: finished, and its last evaluation reached the goal
    (d / "train_meta.json").write_text(_json.dumps({"curve": [{"success": 0.0}]}))
    assert not learning_status(d)["good"]
    (d / "train_meta.json").write_text(_json.dumps({"curve": [{"success": 120.0}]}))
    assert learning_status(d)["good"]
    (d / "candidate_0").mkdir()  # an interrupted plan search
    assert not learning_status(d)["good"]
