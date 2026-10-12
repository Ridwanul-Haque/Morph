"""The app's long jobs (learning, runs) keep the scene and goal they started with: the page may load another scene or goal
meanwhile (loading a scene clears the goal), which once crashed learning at its very end, after all the practice."""
import json
from pathlib import Path

import morph.app.server as srv
import morph.learn_task as lt
from morph.goal_sources import goal_from_description_rules
from morph.project import load_project

ROOT = Path(__file__).resolve().parents[1]


def test_learning_finishes_with_its_own_goal_after_the_page_loads_another_scene(tmp_path, monkeypatch):
    app = srv.App.__new__(srv.App)  # without the live viewer's renderer
    app.history, app.subs, app.lock = [], [], srv.threading.Lock()
    app.planned, app.placements, app.suggestions, app.suggest_lock = {}, {}, [], srv.threading.Lock()
    app.compute, app.demo, app.llm, app.training_task, app.latest_ckpt = {"mode": "local"}, None, None, None, None
    app.live = type("L", (), {"pace": False, "cancel": False})()
    app.project = load_project(ROOT / "examples" / "parcel")
    app.goal = goal_from_description_rules("push the parcel against the wall", app.project.scene)
    app.resolve = lambda p: p
    started = app.goal
    task = tmp_path / "skill" / "task.json"
    task.parent.mkdir()
    task.write_text(json.dumps({"robot": "tiny", "steps": [], "conditions": []}))
    (task.parent / "policy.npz").write_bytes(b"")
    monkeypatch.setattr(lt, "prepare_learning", lambda *a, **k: (task, {}))
    seen = {}

    def learn_in_rounds(pj, goal, *a, **k):
        app.goal = None  # what loading another scene does to the page's goal, an hour into learning
        seen["goal"] = goal
        return {"success": 3, "n": 10, "rounds": []}

    monkeypatch.setattr(lt, "learn_in_rounds", learn_in_rounds)
    monkeypatch.setattr(srv, "ROOT", tmp_path)  # .venv-rl/bin/python must "exist"
    (tmp_path / ".venv-rl" / "bin").mkdir(parents=True)
    (tmp_path / ".venv-rl" / "bin" / "python").write_text("")
    srv._learn(app, "tiny.urdf", use_llm=False)
    assert seen["goal"] is started
    assert app.result["goal"] == started.model_dump() and app.result["robots"][0]["success"] == 3
    assert any(e["type"] == "learn_done" for e in app.history)
