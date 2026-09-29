"""Intent benchmark: does the goal inferred from (demo + instruction) match what the
operator wanted? Compares Nemotron against the rule-based baseline.

Each case runs a real source demo towards `demo_goal` (which can be sloppy, i.e. not
what the operator meant), infers a TaskSpec, runs the target robot on it, and scores
the final block pose against `intended`, not against the inferred spec. Some cases
are ones the rule baseline already gets right.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .arm_skills import expert_action, run_episode
from .capability import assess
from .geometry import yaw_error
from .orchestrator import RunContext, infer
from .reasoning import ReasoningService
from .recorder import record_demo
from .robots.arm import ArmAdapter
from .scene import ARM_A, ARM_B
from .sim import PhysicalParams

D = math.radians
SOURCE, TARGET = "franka_panda", "ufactory_xarm7"


@dataclass
class Case:
    name: str
    instruction: str
    demo_block: tuple
    demo_goal: tuple  # where the demonstrator actually put it
    intended: tuple | None  # (x, y, yaw or None if heading irrelevant); None = should be unsupported
    kind: str  # clean | sloppy | variant | unsupported | free_heading


CASES = [
    Case("clean_north", "Place the arrow-marked block inside the green zone with its arrow pointing north. Do not tip it over.",
         (0.46, -0.20, 0.4), (0.45, 0.20, D(90)), (0.45, 0.20, D(90)), "clean"),
    Case("clean_west_here", "Put the block down where I put it, with the arrow pointing west.",
         (0.50, -0.15, -1.0), (0.40, 0.25, D(180)), (0.40, 0.25, D(180)), "clean"),
    Case("clean_south", "Leave the block in the green zone with the arrow pointing south.",
         (0.40, -0.25, 2.0), (0.45, 0.20, D(-90)), (0.45, 0.20, D(-90)), "clean"),
    Case("sloppy_heading", "Place the block in the green zone with the arrow pointing north.",
         (0.46, -0.20, 0.4), (0.45, 0.20, D(68)), (0.45, 0.20, D(90)), "sloppy"),
    Case("sloppy_position", "Place the block inside the green zone, arrow pointing north.",
         (0.42, -0.22, -0.6), (0.52, 0.24, D(90)), (0.45, 0.20, D(90)), "sloppy"),
    Case("variant_east", "Same place as my demonstration, but this time the arrow should point east.",
         (0.46, -0.20, 0.4), (0.45, 0.20, D(90)), (0.45, 0.20, D(0)), "variant"),
    Case("free_heading", "Move the block into the green zone. Its orientation does not matter.",
         (0.48, -0.18, 1.2), (0.45, 0.20, D(37)), (0.45, 0.20, None), "free_heading"),
    Case("flip_request", "Put the block in the green zone and turn it over so the arrow faces the floor.",
         (0.46, -0.20, 0.4), (0.45, 0.20, D(90)), None, "unsupported"),
]


def score_case(case: Case, method: str, llm: ReasoningService | None, out: Path, seed: int = 31) -> dict:
    from .robots.registry import make_arm
    arm_a, arm_b = make_arm(SOURCE), make_arm(TARGET)
    demo = record_demo(arm_a, seed, case.demo_block, case.demo_goal, case.instruction)
    ctx = RunContext(out, llm if method == "nemotron" else None, log=lambda s: None)
    spec, info = infer(ctx, demo, case.instruction)
    spec = spec.confirm()
    cap = assess(spec, arm_b.capabilities(), (0.47, -0.18, 0.0))
    row = {"case": case.name, "kind": case.kind, "method": method, "inference": info, "verdict": cap.verdict,
           "inferred_goal": {"x": round(spec.gx, 3), "y": round(spec.gy, 3), "yaw_deg": round(math.degrees(spec.gyaw), 1),
                             "pos_tol_cm": round(float(spec.goal.pos_tol_m.value) * 100, 1),
                             "yaw_tol_deg": round(math.degrees(float(spec.goal.yaw_tol_rad.value)), 1),
                             "mode": spec.goal.orientation_mode.value},
           "demo_final": demo.block_end}
    if case.intended is None:
        row["matches_intent"] = cap.verdict == "unsupported"
        row["note"] = "expected outcome: 'unsupported'"
        return row
    if cap.verdict == "unsupported":
        row["matches_intent"] = False
        row["note"] = "wrongly declared unsupported"
        return row
    arm_b.reset(seed + 1, (0.47, -0.18, 0.0), PhysicalParams(block_mass=0.25, block_friction=0.55))
    r = run_episode(arm_b, spec, expert_action)
    p = arm_b.env.block_pose()
    pos_err = math.hypot(p.x - case.intended[0], p.y - case.intended[1])
    yerr = None if case.intended[2] is None else yaw_error(p.yaw, case.intended[2])
    ok = r.outcome.released and pos_err <= 0.05 and (yerr is None or yerr <= D(15)) and p.tilt < D(10)
    row |= {"target_final": p.as_dict(), "err_vs_intent_cm": round(pos_err * 100, 2),
            "yaw_err_vs_intent_deg": None if yerr is None else round(math.degrees(yerr), 1), "matches_intent": bool(ok)}
    return row


def run_bench(methods=("rule", "nemotron"), out: Path = Path("artifacts/evidence/intent_bench"), log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    llm = ReasoningService(log_path=out / "inference.jsonl")
    res = {"cases": [asdict(c) for c in CASES], "rows": [], "started": time.time()}
    for m in methods:
        if m == "nemotron" and not llm.available:
            log("nemotron: skipped (NEBIUS_API_KEY not set)")
            continue
        for c in CASES:
            row = score_case(c, m, llm, out)
            res["rows"].append(row)
            log(f"{m:9s} {c.name:16s} verdict={row['verdict']:25s} inferred={row['inferred_goal']} "
                f"matches_intent={row['matches_intent']}")
    summary = {}
    for m in methods:
        rows = [r for r in res["rows"] if r["method"] == m]
        if rows:
            summary[m] = {"matches_intent": sum(r["matches_intent"] for r in rows), "n": len(rows),
                          "by_kind": {k: f"{sum(r['matches_intent'] for r in rows if r['kind'] == k)}/"
                                         f"{sum(1 for r in rows if r['kind'] == k)}" for k in sorted({r['kind'] for r in rows})}}
    res["summary"] = summary
    res["inference_calls"] = [{"call": r.call, "latency_s": r.latency_s, "usage": r.usage, "accepted": r.accepted,
                               "error": r.error} for r in llm.records]
    (out / "intent_bench.json").write_text(json.dumps(res, indent=2, default=str))
    log(json.dumps(summary, indent=2))
    return res
