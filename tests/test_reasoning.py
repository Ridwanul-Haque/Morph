"""Tests for the Nemotron client using a fake HTTP layer (no real API calls).
The live path is covered by `morph check-llm` and `morph transfer`.
"""

import json

import httpx
import pytest

from morph import reasoning as R


def test_extract_json_handles_thinking_and_fences():
    assert R._extract_json('<think>{"no": 1}</think> answer: {"a": 1}') == {"a": 1}
    assert R._extract_json('text ```json\n{"b": [1, 2]}\n``` trailing') == {"b": [1, 2]}
    with pytest.raises(ValueError):
        R._extract_json("no json here")


def test_validate_revision_rejects_out_of_space():
    space = {"strategy_id": {"choices": ["a", "b"]}, "push_dist": {"range": [0.02, 0.2]}}
    assert R.validate_revision({"strategy_id": "a", "push_dist": 0.1}, space) == []
    bad = R.validate_revision({"strategy_id": "z", "push_dist": 5.0, "teleport": True}, space)
    assert len(bad) == 3


def test_missing_key_is_reported_not_faked(monkeypatch, tmp_path):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    svc = R.ReasoningService()
    assert not svc.available
    with pytest.raises(R.ReasoningUnavailable):
        svc.interpret_task("x", {}, {})


class _Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.headers = {}
        self.text = json.dumps(body)

    def json(self):
        return self._body


def _ok(content):
    return _Resp(200, {"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 42}})


def test_retry_then_success_and_log(monkeypatch, tmp_path):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key-not-real")
    calls = []
    good = {"strategy_id": "goal_replan", "rationale": "r", "evidence_cited": [], "expected_risks": []}
    seq = [_Resp(503, {"error": "busy"}), _ok("<think>hmm</think>" + json.dumps(good))]

    def fake_post(url, headers, json, timeout):
        calls.append((url, headers["Authorization"]))
        return seq.pop(0)

    monkeypatch.setattr(R.httpx, "post", fake_post)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    svc = R.ReasoningService(log_path=tmp_path / "inf.jsonl")
    choice, rec = svc.select_strategy({}, {}, [{"id": "goal_replan"}, {"id": "joint_replay"}])
    assert choice.strategy_id == "goal_replan" and rec.attempts == 2 and rec.usage["total_tokens"] == 42
    log = (tmp_path / "inf.jsonl").read_text()
    assert "test-key-not-real" not in log  # the key never reaches the log


def test_invented_strategy_is_rejected(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "k")
    bad = {"strategy_id": "use_a_magic_gripper", "rationale": "r"}
    monkeypatch.setattr(R.httpx, "post", lambda *a, **k: _ok(json.dumps(bad)))
    with pytest.raises(R.ReasoningError):
        R.ReasoningService().select_strategy({}, {}, [{"id": "goal_replan"}])


def test_http_error_falls_back_to_rule_in_pipeline(monkeypatch, tmp_path):
    monkeypatch.setenv("NEBIUS_API_KEY", "k")
    monkeypatch.setattr(R.httpx, "post", lambda *a, **k: _Resp(401, {"error": "unauthorized"}))
    from morph.pipeline import run_arm_transfer
    r = run_arm_transfer(tmp_path / "run", "Place the block in the green zone pointing north.", verify_n=0, render=False,
                         log=lambda s: None)
    assert r["status"] == "success"
    assert r["inference"]["method"] == "rule_measured_final_pose" and "401" in r["inference"]["llm_error"]
    assert all(c["error"] for c in r["inference_calls"])  # failed calls still show up in the result
