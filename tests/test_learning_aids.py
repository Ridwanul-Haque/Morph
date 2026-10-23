"""What helps the learner when there's no working program to copy: acting in hand space, practice starts near success
from Morph's own route, and learning settings meant for arms."""
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

import morph.learn_task as lt
from morph.goal_sources import goal_from_description_rules
from morph.project import _params, load_project
from morph.task_model import apply_action, build_task_model, indices, load_task

ROOT = Path(__file__).resolve().parents[1]
MEN = ROOT / "third_party" / "mujoco_menagerie"


@pytest.fixture(scope="module")
def parcel_bin(tmp_path_factory):
    pj = load_project(ROOT / "examples" / "parcel")
    return pj, goal_from_description_rules("put the parcel in the bin", pj.scene), tmp_path_factory.mktemp("aids")


def test_an_arm_can_act_in_hand_space(parcel_bin, monkeypatch):
    pj, goal, tmp = parcel_bin
    monkeypatch.setattr(lt, "LEARNED", tmp)
    tp = lt.make_task(pj, goal, str(MEN / "franka_emika_panda" / "panda.xml"), plan=False, folder=tmp / "panda")
    t = json.loads(tp.read_text())
    assert len(t["hand_space"]["joints"]) == 7 and "action_space" not in t and "ppo" not in t  # opt-in, not the default
    assert lt.use_hand_space(tp, log=lambda s: None)
    task = load_task(tp)
    from morph.sim import SimulationEnvironment
    m = build_task_model(task, accurate=True)
    ix = indices(m, task)
    env = SimulationEnvironment(m)
    h = ix.hand

    def moved(vec, n=30):
        env.reset(0, pj.scene.start, _params(0))
        d = env.data
        d.qpos[ix.home_adr] = ix.home_val
        ctrl0 = np.clip(ix.q0, ix.act_lo, ix.act_hi)
        ctrl = ctrl0.copy()
        mujoco.mj_forward(m, d)
        hand = lambda: d.xpos[h["body"]] + d.xmat[h["body"]].reshape(3, 3) @ h["offset"]  # noqa: E731
        p0 = hand().copy()
        a = np.zeros(m.nu)
        a[h["slots"]] = vec
        for _ in range(n):
            ctrl = apply_action(np, ctrl, a, ctrl0, ix, d)
            env.advance(ctrl)
        return hand() - p0

    dp = moved([0, -1, 0, 0, 0, 0])
    assert dp[1] < -0.08 and abs(dp[0]) < 0.02 and abs(dp[2]) < 0.02  # where it was told, and nowhere else
    assert np.linalg.norm(moved([0] * 6)) < 0.002  # told nothing: it holds still (no sag)


def test_practice_starts_near_success_from_morphs_own_route(parcel_bin, monkeypatch):
    import morph.contact_plan as cp
    from morph.scaffold import scaffold_starts
    pj, goal, tmp = parcel_bin
    monkeypatch.setattr(lt, "LEARNED", tmp)
    monkeypatch.setattr(cp, "GRASP_CACHE", tmp / "grasps")
    tp = lt.make_task(pj, goal, str(MEN / "franka_emika_panda" / "panda.xml"), plan=False, folder=tmp / "route")
    info = scaffold_starts(tp, pj, goal, log=lambda s: None)
    assert info["holding"] >= 10  # real states with the parcel in its grip
    t = json.loads(tp.read_text())
    z = np.load(tp.parent / t["starts"]["file"])
    assert t["starts"]["scaffold"] and len(z["qpos"]) == info["starts"] and np.all(np.diff(z["t"]) >= 0)


def test_a_wheeled_base_is_set_down_where_it_can_grip_the_object(parcel_bin, monkeypatch):
    """The Stretch never got within 50 cm of the parcel; then Morph could drive it up but not work its gripper (only
    bolted-down arms had theirs recognised), so practice never started with the parcel held. Now the base is set down
    where its hand can reach the parcel from one side, Morph grips it there itself, and practice also starts holding
    it."""
    import morph.contact_plan as cp
    from morph.scaffold import scaffold_starts
    from morph.task_model import contact_held
    pj, goal, tmp = parcel_bin
    monkeypatch.setattr(lt, "LEARNED", tmp)
    monkeypatch.setattr(cp, "GRASP_CACHE", tmp / "grasps_stretch")
    tp = lt.make_task(pj, goal, str(MEN / "hello_robot_stretch" / "stretch.xml"), plan=False, folder=tmp / "stretch")
    t = json.loads(tp.read_text())
    assert "ppo" not in t  # a mobile base keeps the walking-robot settings
    assert t["gripper"]["actuator"] == "grip"  # the jaws' motor, not the wrist's
    info = scaffold_starts(tp, pj, goal, log=lambda s: None)
    assert info and info["drive"] >= 4 and info["holding"] >= 10
    task = load_task(tp)
    m = build_task_model(task, accurate=True)
    ix = indices(m, task)
    d = mujoco.MjData(m)
    z = np.load(tp.parent / "starts.npz")
    held = 0
    for q in z["qpos"]:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        held += contact_held(m, d, ix) > 0.5
    assert held >= 10  # real states with the parcel in its grip
    # and the moments just before Morph's grip, gripper open, to practise closing from; the learner aims for that grip
    assert info["ready"] >= 5 and json.loads(tp.read_text())["grasp"]["fingers"]
    ready = z["ready"]
    gi = ix.grip_act
    assert all((z["ctrl"][i][gi] - ix.grip_open) / (ix.grip_close - ix.grip_open) < 0.2 for i in np.flatnonzero(ready))
