"""The learner's reward, part by part: shown in the app, changed through suggestions within limits, and watched for
loopholes (the score rising while the checklist doesn't)."""
from morph.reward_parts import REWARD_PARTS, check_weights, loophole, view, weights


def test_suggested_weights_stay_within_limits():
    task = {"conditions": [], "reward_weights": {"reach": 0.0, "near": 0.0, "fling": 9}}
    w = weights(task)
    assert w["reach"] == 0.0  # a hint can be switched off ...
    assert w["near"] == 0.5  # ... but not the parts that measure the goal itself
    assert w["fling"] == 3.0 and w["toward"] == 1.0
    kept, problems = check_weights({"reach": 0.2, "wobble": 1, "copy": "lots"}, task)
    assert kept["reach"] == 0.2 and len(problems) == 2


def test_a_throw_isnt_penalised_for_moving_the_object_fast():
    throw = {"conditions": [{"kind": "expression", "expr": "speed > 1", "when": "release"}]}
    assert weights(throw)["fling"] == 0.0 and weights({"conditions": []})["fling"] == 1.0
    v = view(throw)
    assert set(v["reward_parts"]) == set(REWARD_PARTS) and v["reward_defaults"]["fling"] == 0.0


def _curve(reward, reach, success=0.0):
    return [{"step": k * 1_000_000, "reward": r, "success": s, "steps_done": 0.0, "parts": {"reach": p, "near": 0.1}}
            for k, (r, p, s) in enumerate(zip(reward, reach, success if isinstance(success, list) else [success] * len(reward)))]


def test_a_loophole_is_the_score_rising_late_while_the_checklist_doesnt():
    # the second half of the round: the score climbs, all of it from reaching, and the checklist never passes
    hole = loophole(_curve([1, 2, 3, 6, 9], [1, 2, 3, 6, 9]), {})
    assert hole and hole["part"] == "reach" and hole["label"] in hole["text"]
    # rising early in the round only (learning to reach the object): progress, not a loophole
    assert loophole(_curve([1, 6, 9, 9, 9], [1, 6, 9, 9, 9]), {}) is None
    # rising, and the checklist with it: fine
    assert loophole(_curve([1, 2, 3, 6, 9], [1, 2, 3, 6, 9], [0, 0, 0, 0.2, 0.5]), {}) is None
