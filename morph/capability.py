"""Capability check: a quick filter before trying anything, not a guarantee.

Verdicts: "plausible", "unknown_needs_experiment", "unsupported".
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .robots.base import Capabilities
from .task import TaskSpec

Verdict = str


@dataclass
class Check:
    name: str
    status: Verdict
    evidence: str


@dataclass
class CapabilityReport:
    robot: str
    verdict: Verdict
    checks: list[Check] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"robot": self.robot, "verdict": self.verdict, "checks": [asdict(c) for c in self.checks]}


def _in_annulus(ws: dict, x: float, y: float) -> tuple[bool, float]:
    r = math.hypot(x - ws["center"][0], y - ws["center"][1])
    return ws["r_min"] <= r <= ws["r_max"], r


def assess(spec: TaskSpec, caps: Capabilities, object_start: tuple[float, float, float] | None = None) -> CapabilityReport:
    checks: list[Check] = []
    g = spec.goal
    face = str(g.target_top_face.value)
    mode = str(g.orientation_mode.value)

    # 1. Orientation class
    if mode != "planar_yaw" or face != "+z":
        if caps.can_flip_object:
            checks.append(Check("orientation", "unknown_needs_experiment", f"flip to top face {face} requested; robot lists a flip primitive"))
        else:
            reason = ("no grasp: the controller can only push; pushing a box on flat ground rotates it about the vertical "
                      "axis, and tipping it needs a fixture or lever that is not in the scene"
                      if not caps.prehensile else
                      "top-down grasp only (wrist pitch fixed vertical); a flip needs a side grasp or regrasp fixture, not implemented")
            checks.append(Check("orientation", "unsupported", f"requested top face {face} (mode {mode}): {reason}"))
    else:
        checks.append(Check("orientation", "plausible" if caps.can_set_planar_yaw else "unsupported",
                            "planar yaw change only" + ("" if caps.can_set_planar_yaw else "; robot cannot change yaw")))

    # 2. Reach / workspace
    ws = caps.object_workspace
    if ws.get("type") == "annulus":
        pts = [("goal", spec.gx, spec.gy)] + ([("start", object_start[0], object_start[1])] if object_start else [])
        for label, x, y in pts:
            ok, r = _in_annulus(ws, x, y)
            checks.append(Check(f"reach_{label}", "plausible" if ok else "unsupported",
                                f"{label} at r={r:.3f} m; fixed-base reachable annulus [{ws['r_min']}, {ws['r_max']}] m"))
    else:
        checks.append(Check("reach", "plausible", "mobile base: can reposition to any floor location in the scene"))

    # 3. Object size / mass vs end effector
    m = spec.object.mass_kg
    if caps.prehensile:
        width = min(spec.object.size_m[:2])
        opening = caps.modeled_limits.get("gripper_max_opening_m", 0)
        checks.append(Check("grasp_width", "plausible" if width < opening else "unsupported",
                            f"object {width:.3f} m vs gripper opening {opening:.3f} m"))
        checks.append(Check("payload", "plausible" if m <= caps.max_payload_kg else "unsupported",
                            f"mass {m:.2f} kg vs friction-limited payload {caps.max_payload_kg:.2f} kg"))
    else:
        need = m * 9.81 * spec.object.friction
        ok = need < 0.5 * caps.max_contact_force_n
        checks.append(Check("push_force", "plausible" if ok else "unknown_needs_experiment",
                            f"sliding needs ~{need:.2f} N (m*g*mu) vs ~{caps.max_contact_force_n:.0f} N foot force budget"))
        h = spec.object.size_m[2]
        lo, hi = ws.get("object_height_range_m", [0, 1])
        checks.append(Check("contact_height", "plausible" if lo <= h <= hi else "unsupported",
                            f"object height {h:.3f} m; foot can push objects {lo}-{hi} m tall"))
        checks.append(Check("yaw_by_pushing", "unknown_needs_experiment",
                            "yaw must be produced by off-centre pushes; controllability depends on friction -> needs rollouts"))

    # 4. Uprightness
    if g.keep_upright.value and not caps.prehensile:
        checks.append(Check("keep_upright", "unknown_needs_experiment",
                            "push height below the block's centre reduces tipping; verify in simulation"))

    statuses = [c.status for c in checks]
    verdict = ("unsupported" if "unsupported" in statuses else
               "unknown_needs_experiment" if "unknown_needs_experiment" in statuses else "plausible")
    return CapabilityReport(robot=caps.robot, verdict=verdict, checks=checks)
