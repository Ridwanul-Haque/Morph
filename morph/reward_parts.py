"""The parts of the learner's reward (scripts/train_task.py), in plain words, and how far each may be turned up or down.

The reward is what the learner is scored on every moment while it practises; the goal checklist is what it's judged on.
Each part is logged on its own (its total per practice run), shown in the app's Learning tab, and can be weighted by a
person's suggestion through Nemotron (task["reward_weights"]: part -> multiplier). The parts that measure the goal itself
can't be turned below half, so a suggestion can't switch off what the learner is there to do.
"""

from __future__ import annotations

# key -> (plain words, lowest multiplier, highest)
REWARD_PARTS: dict[str, tuple[str, float, float]] = {
    "toward": ("moving the object towards where it must end up", 0.5, 2.0),
    "near": ("the object being close to where it must end up, with the checklist met", 0.5, 2.0),
    "success": ("a bonus every moment the whole checklist is met", 0.5, 2.0),
    "reach": ("the hand getting to the object (to Morph's planned grip, when there is one)", 0.0, 2.0),
    "heading": ("the object facing the way the goal asks", 0.0, 2.0),
    "upright": ("keeping the object upright (a penalty for tilting it)", 0.0, 2.0),
    "smooth": ("smooth motion (a penalty for jerky moves)", 0.0, 3.0),
    "fling": ("not flinging the object (a penalty when it moves faster than 0.5 m/s)", 0.0, 3.0),
    "standing": ("staying on its feet", 0.5, 2.0),
    "copy": ("doing what Nemotron's working program did", 0.0, 2.0),
    "steps": ("reaching each step of the plan, in order", 0.0, 2.0),
}


# key -> a short name for the app (the plain words above go under it)
SHORT = {"toward": "Moving it towards the goal", "near": "Being at the goal", "success": "Checklist met",
         "reach": "Reaching the object", "heading": "Facing the right way", "upright": "Keeping it upright",
         "smooth": "Smooth motion", "fling": "Not flinging it", "standing": "Staying on its feet",
         "copy": "Copying the program", "steps": "Following the plan's steps"}


def default_weights(task: dict) -> dict[str, float]:
    """Morph's own starting weights: 1 for every part, except the flinging penalty on a throw (the goal itself asks for
    the object to leave the hand fast)."""
    w = {k: 1.0 for k in REWARD_PARTS}
    if any(c.get("when") == "release" for c in task.get("conditions", [])):
        w["fling"] = 0.0
    return w


def weights(task: dict) -> dict[str, float]:
    """The weights the learner uses: Morph's defaults, then any a suggestion changed, each kept within its range."""
    w = default_weights(task)
    for k, v in (task.get("reward_weights") or {}).items():
        if k in REWARD_PARTS:
            _, lo, hi = REWARD_PARTS[k]
            w[k] = min(max(float(v), lo), hi)
    return w


def check_weights(proposed, task: dict) -> tuple[dict[str, float], list[str]]:
    """Nemotron's proposed weights -> (the ones kept, each within its range; problems in words)."""
    if not proposed:
        return dict(task.get("reward_weights") or {}), []
    if not isinstance(proposed, dict):
        return dict(task.get("reward_weights") or {}), ["reward_weights must be an object: {part: multiplier}"]
    out, problems = dict(task.get("reward_weights") or {}), []
    for k, v in proposed.items():
        if k not in REWARD_PARTS:
            problems.append(f"reward_weights: unknown part {k!r} (known: {', '.join(REWARD_PARTS)})")
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            problems.append(f"reward_weights: {k} must be a number")
            continue
        _, lo, hi = REWARD_PARTS[k]
        out[k] = round(min(max(v, lo), hi), 3)
    return out, problems


def doc() -> str:
    """For Nemotron: the parts, and how to change them."""
    return ("The reward also has Morph's built-in parts (reward_parts in measured: each one's total per practice run at "
            "the start and end of the last round). You may change how much each counts with \"reward_weights\": "
            "{part: multiplier} (1 = as now; " + "; ".join(f"{k}: {t} ({lo:g}..{hi:g})" for k, (t, lo, hi) in
                                                        REWARD_PARTS.items()) + "). Use it when a part pays for "
            "something that isn't the goal (a loophole: e.g. reach paying for poking the object rather than gripping it), "
            "or when the person asks for a part to count more or less. ")


def part_totals(rec: dict, task: dict) -> dict[str, float]:
    """One evaluation's per-part totals, Nemotron's own terms under their names."""
    out = dict(rec.get("parts") or {})
    for t, v in zip(task.get("reward_terms") or [], rec.get("term_totals") or []):
        out[f"term: {t['name']}"] = v
    return out


def loophole(curve: list[dict], task: dict) -> dict | None:
    """The reward going up while the checklist doesn't: which part is paying for it. Only in the second half of a round
    (early on, the reward rises as the hand learns to reach the object: that's progress, not a loophole): the first
    evaluation from half-way against the latest."""
    pts = [c for c in curve if c.get("parts")]
    if not pts:
        return None
    pts = [c for c in pts if c.get("step", 0) >= pts[-1].get("step", 0) / 2]
    if len(pts) < 2:
        return None
    a, b = pts[0], pts[-1]
    ra, rb = a.get("reward", 0.0), b.get("reward", 0.0)
    if not (rb - ra > max(1.0, 0.2 * abs(ra))):  # not clearly rising
        return None
    if (b.get("success") or 0.0) - (a.get("success") or 0.0) > 0.02 or (b.get("steps_done") or 0.0) - (a.get("steps_done") or 0.0) > 0.1:
        return None  # the checklist (or the plan's steps) is improving with it: fine
    pa, pb = part_totals(a, task), part_totals(b, task)
    k = max(pb, key=lambda x: pb[x] - pa.get(x, 0.0))
    if pb[k] - pa.get(k, 0.0) <= 0:
        return None
    label = REWARD_PARTS[k][0] if k in REWARD_PARTS else k
    return {"part": k, "label": label, "from": round(pa.get(k, 0.0), 2), "to": round(pb[k], 2),
            "reward_from": round(ra, 1), "reward_to": round(rb, 1),
            "text": f"its score rose from {ra:.0f} to {rb:.0f} but it isn't meeting the checklist more often; most of the "
                    f"rise comes from \"{label}\" ({pa.get(k, 0.0):.1f} → {pb[k]:.1f}). It may be earning that without "
                    "getting closer to the goal"}


def view(task: dict) -> dict:
    """For the app: every part's plain words and weight, and Nemotron's own terms."""
    return {"reward_parts": {k: t for k, (t, _, _) in REWARD_PARTS.items()}, "reward_short": SHORT,
            "reward_weights": weights(task),
            "reward_defaults": default_weights(task),
            "reward_terms": [{"name": t["name"], "weight": t["weight"]} for t in task.get("reward_terms") or []]}
