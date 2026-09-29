"""Go1 foot-contact feasibility check.

Can the Go1 stand, shift onto three legs, bring its front-right foot into contact
with the block and push it, without falling?

Writes runs/spike_quadruped/{log.json,summary.json,spike.mp4}.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from morph.robots.quadruped import LEGS, QuadrupedAdapter, home_feet
from morph.sim import PhysicalParams

OUT = Path("runs/spike_quadruped")


def smooth(a: float) -> float:
    a = min(1.0, max(0.0, a))
    return a * a * (3 - 2 * a)


def main(video: bool = True, block_x: float = 0.34, seed: int = 0) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    qa = QuadrupedAdapter()
    params = PhysicalParams(block_mass=0.2, block_friction=0.6)
    block0 = (block_x, -0.13, 0.0)
    qa.reset(seed, block0, params)
    env = qa.env
    env.set_goal_marker(block_x + 0.08, -0.13, 0.0)

    feet0 = home_feet()
    stance = ["FL", "RR", "RL"]
    frames, log = [], []
    shift = np.zeros(2)
    swing_start = None
    contact_force_max, contact_ticks, first_contact_t = 0.0, 0, None
    min_margin = 1e9
    fell_at = None

    T = 8.0
    n = int(T / env.control_dt)
    for k in range(n):
        t = k * env.control_dt
        b = qa.base()
        if t < 1.0:  # settle on four legs
            phase = "stand"
        elif t < 2.0:  # shift the CoM over the three-leg support triangle
            phase = "shift"
            if k == int(round(1.0 / env.control_dt)):
                tri = np.array([qa.world_to_trunk(qa.foot_world(l))[:2] for l in stance])
                com_t = qa.world_to_trunk(qa.com())[:2]
                target = tri.mean(0)
                shift_goal = (com_t - target) * 1.15  # small overshoot for margin
        elif t < 7.0:
            phase = "swing"
        else:
            phase = "return"

        s = shift_goal * smooth(t - 1.0) if t >= 1.0 else np.zeros(2)
        if phase == "return":
            s = shift_goal * (1 - smooth((t - 7.0) / 0.8))
        feet = {l: feet0[l] + np.array([s[0], s[1], 0.0]) for l in LEGS}

        if phase == "swing":
            # Swing-foot target defined in a yaw-only frame from the floor, so the
            # push line is fixed in the world while the trunk wobbles.
            if swing_start is None:
                swing_start = qa.foot_world("FR")
            ts = t - 2.0
            fx0, fy0 = swing_start[0], swing_start[1]
            face_x = block0[0] - qa.block.half_x - 0.023  # foot centre at block rear face
            if ts < 0.8:  # lift and move forward above the floor
                a = smooth(ts / 0.8)
                w = np.array([fx0 + (face_x - 0.04 - fx0) * a, fy0, 0.023 + 0.07 * math.sin(math.pi * a / 2)])
            elif ts < 1.4:  # lower to push height
                a = smooth((ts - 0.8) / 0.6)
                w = np.array([face_x - 0.04, fy0, 0.093 - (0.093 - 0.035) * a])
            elif ts < 3.4:  # push forward 12 cm through the block face
                a = smooth((ts - 1.4) / 2.0)
                w = np.array([face_x - 0.04 + 0.16 * a, fy0, 0.035])
            elif ts < 4.2:  # retract and lift
                a = smooth((ts - 3.4) / 0.8)
                w = np.array([face_x + 0.12 - 0.12 * a, fy0, 0.035 + 0.05 * a])
            else:  # put the foot back down
                a = smooth((ts - 4.2) / 0.8)
                w = np.array([face_x + (swing_start[0] - face_x) * a, fy0, 0.085 - 0.062 * a])
            feet["FR"] = qa.heading_frame_to_trunk(w, 0.0) - np.array([0, 0, b.pos[2]])
            feet["FR"][2] = w[2] - b.pos[2]

        ctrl = qa.ctrl_from_feet(feet)
        if ctrl is None:
            print(f"IK failure at t={t:.2f} ({phase})", file=sys.stderr)
            ctrl = env.data.ctrl.copy()
        step = env.advance(ctrl)
        fr_contacts = [c for c in step.robot_contacts if c.geom == "FR"]
        other_contacts = sorted({c.body for c in step.robot_contacts if c.geom != "FR"})
        if fr_contacts:
            contact_ticks += 1
            contact_force_max = max(contact_force_max, max(c.normal_force for c in fr_contacts))
            first_contact_t = first_contact_t if first_contact_t is not None else t
        if phase in ("shift", "swing") and t > 1.8:
            min_margin = min(min_margin, qa.support_margin(stance))
        if fell_at is None and qa.fallen():
            fell_at = t
        bp = env.block_pose()
        b = qa.base()
        log.append({"t": round(t, 3), "phase": phase, "trunk": [round(float(v), 4) for v in b.pos],
                    "roll_deg": round(math.degrees(b.roll), 2), "pitch_deg": round(math.degrees(b.pitch), 2),
                    "margin_m": round(qa.support_margin(stance), 4), "fr_foot": [round(float(v), 4) for v in qa.foot_world("FR")],
                    "fr_block_force_n": round(max([c.normal_force for c in fr_contacts], default=0.0), 3),
                    "other_block_contacts": other_contacts, "block": bp.as_dict(),
                    "max_block_step_disp_mm": round(step.max_step_disp * 1000, 3)})
        if video and k % 2 == 0:
            frames.append(env.render("overview", 640, 480))

    bp = env.block_pose()
    b = qa.base()
    summary = {
        "question": "Go1 deliberate front-foot contact and push, three-leg stance",
        "block_start": block0,
        "block_end": bp.as_dict(),
        "block_displacement_m": round(math.hypot(bp.x - block0[0], bp.y - block0[1]), 4),
        "block_yaw_change_deg": round(math.degrees(bp.yaw - block0[2]), 2),
        "fr_foot_block_contact_ticks": contact_ticks,
        "fr_first_contact_t": first_contact_t,
        "fr_max_contact_force_n": round(contact_force_max, 2),
        "non_foot_block_contacts": sorted({c for r in log for c in r["other_block_contacts"]}),
        "min_support_margin_m_during_swing": round(min_margin, 4),
        "max_abs_roll_deg": max(abs(r["roll_deg"]) for r in log),
        "max_abs_pitch_deg": max(abs(r["pitch_deg"]) for r in log),
        "trunk_height_min_m": min(r["trunk"][2] for r in log),
        "fell_at": fell_at,
        "final_trunk": [round(float(v), 4) for v in b.pos],
        "max_block_step_disp_mm": max(r["max_block_step_disp_mm"] for r in log),
    }
    summary["verdict"] = (
        "PASS" if fell_at is None and contact_ticks > 5 and summary["block_displacement_m"] > 0.03
        and not summary["non_foot_block_contacts"] else "FAIL")
    (OUT / "log.json").write_text(json.dumps(log))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    if video and frames:
        import imageio.v2 as imageio
        imageio.mimsave(OUT / "spike.mp4", frames, fps=25)
    env.close()
    return summary


if __name__ == "__main__":
    print(json.dumps(main(video="--no-video" not in sys.argv), indent=2))
