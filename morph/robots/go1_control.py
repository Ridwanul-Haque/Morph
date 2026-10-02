"""Hand-written Go1 motions: crawl gait and front-foot push.

* crawl: one leg in the air at a time, shifting the CoM inside the three-leg support
  triangle before each swing,
* push: three-leg stance, swing the front-right foot along a world-frame line.

The controller keeps a desired trunk pose (x, y, yaw, fixed height). Stance feet stay
at their measured touchdown points, expressed in that frame, so the position servos
move the body over the planted feet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..geometry import rot2, wrap_angle
from .quadruped import QuadrupedAdapter

CRAWL_ORDER = ("RL", "FL", "RR", "FR")
RIGHT_TURN_ORDER = ("RR", "FR", "RL", "FL")


def smooth(a: float) -> float:
    a = min(1.0, max(0.0, a))
    return a * a * (3 - 2 * a)


@dataclass
class MotionLog:
    ticks: int = 0
    min_margin: float = 1e9
    max_tilt_deg: float = 0.0
    fell: bool = False
    ik_failures: int = 0
    block_contacts: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    touchdowns: list = field(default_factory=list)
    aborted_steps: int = 0


class Go1Controller:
    def __init__(self, qa: QuadrupedAdapter, height: float | None = None, render_every: int = 0,
                 camera: str = "overview", on_tick=None):
        """`qa.legs` (robots/legs.py) supplies the leg geometry: hand-written for the Go1, measured for
        other quadrupeds. Go1 FOOT_Z: the softened foot contact rests with the sphere centre ~1 cm above
        the floor (measured), not at its 2.3 cm radius; stance/landing targets use that."""
        self.qa = qa
        self.g = g = qa.legs
        self.env = qa.env
        self.h = g.stand_height if height is None else height
        b = qa.base()
        self.des = np.array([b.pos[0], b.pos[1], b.yaw])
        self.feet_w = {l: qa.foot_world(l)[:2].copy() for l in g.legs}  # planted xy
        self.feet_z = {l: g.foot_z for l in g.legs}
        # Chosen by a parameter sweep over margin, stance width, shift time and lead.
        self.max_lead = 0.04
        self.max_yaw_lead = 0.08
        self.margin = 0.045
        self.shift_s = 0.6
        self.swing_s = 0.5
        self.width = 0.04  # extra lateral foot offset (m) beyond the hip line
        self.swing_clear = 0.12  # foot height while moving to the pre-contact point (block top 0.06 + foot 0.023)
        self.render_every = render_every
        self.camera = camera
        self.on_tick = on_tick
        self.log = MotionLog()

    # ----------------------------------------------------------- low level
    def _command_frame(self) -> tuple[np.ndarray, float]:
        """Measured trunk pose plus a clipped lead towards the desired pose.

        All feet are expressed in this single rigid frame, so stance legs never
        fight each other (which made feet slide when the body lagged the plan).
        """
        b = self.qa.base()
        lead = self.des[:2] - b.pos[:2]
        n = np.linalg.norm(lead)
        if n > self.max_lead:
            lead *= self.max_lead / n
        ylead = float(np.clip(wrap_angle(self.des[2] - b.yaw), -self.max_yaw_lead, self.max_yaw_lead))
        return b.pos[:2] + lead, b.yaw + ylead

    def _trunk_frame(self, pw: np.ndarray, z: float, frame=None) -> np.ndarray:
        pos, yaw = frame if frame is not None else self._command_frame()
        d = rot2(-yaw) @ (pw - pos)
        return np.array([d[0], d[1], z - self.h])

    def _tick(self, swing: dict[str, np.ndarray] | None = None, stance_legs: list[str] | None = None) -> None:
        feet = {}
        frame = self._command_frame()
        for l in self.g.legs:
            if swing and l in swing:
                p = swing[l]
                feet[l] = self._trunk_frame(p[:2], p[2], frame)
            else:
                feet[l] = self._trunk_frame(self.feet_w[l], self.feet_z[l], frame)
        ctrl = self.env.data.ctrl.copy()
        for l in self.g.legs:  # per-leg fallback: an unreachable target freezes only that leg
            q = self.g.ik(l, feet[l])
            if q is None:
                self.log.ik_failures += 1
            else:
                ctrl[self.g.ctrl_idx[l]] = q
        step = self.env.advance(ctrl)
        L = self.log
        L.ticks += 1
        b = self.qa.base()
        L.max_tilt_deg = max(L.max_tilt_deg, math.degrees(b.tilt))
        if stance_legs:
            L.min_margin = min(L.min_margin, self.qa.support_margin(stance_legs))
        if self.qa.fallen():
            L.fell = True
        for c in step.robot_contacts:
            L.block_contacts.append({"t": round(c.time, 3), "geom": c.geom, "body": c.body, "force_n": round(c.normal_force, 3)})
        if self.render_every and L.ticks % self.render_every == 0:
            L.frames.append(self.env.render(self.camera))
        if self.on_tick:
            self.on_tick(self)

    def _foot_loaded(self, leg: str) -> bool:
        m, d = self.qa.model, self.env.data
        g = self.qa.foot_geom[leg]
        floor = m.geom("floor").id
        return any({d.contact[i].geom1, d.contact[i].geom2} == {g, floor} for i in range(d.ncon))

    def hold(self, seconds: float) -> None:
        for _ in range(int(seconds / self.env.control_dt)):
            self._tick()

    # ----------------------------------------------------------- body shift
    def _shift_to(self, target_xy: np.ndarray, target_yaw: float, seconds: float, stance: list[str] | None = None) -> None:
        start = self.des.copy()
        goal = np.array([target_xy[0], target_xy[1], start[2] + wrap_angle(target_yaw - start[2])])
        n = max(1, int(seconds / self.env.control_dt))
        for k in range(1, n + 1):
            a = smooth(k / n)
            self.des = start + (goal - start) * a
            self._tick(stance_legs=stance)
        self.des[2] = wrap_angle(self.des[2])

    def _com_offset(self) -> np.ndarray:
        """Measured horizontal CoM offset from the trunk origin, in world axes."""
        return self.qa.com()[:2] - self.qa.base().pos[:2]

    def _support_target(self, stance: list[str], margin: float = 0.04) -> np.ndarray:
        """Smallest trunk shift that puts the CoM at least `margin` inside the stance polygon."""
        pts = np.array([self.feet_w[l] for l in stance])
        c0 = pts.mean(0)
        pts = pts[np.argsort(np.arctan2(pts[:, 1] - c0[1], pts[:, 0] - c0[0]))]  # CCW
        com_off = self._com_offset()
        com = self.des[:2] + com_off
        for _ in range(3):
            for i in range(len(pts)):
                a, b = pts[i], pts[(i + 1) % len(pts)]
                e = b - a
                n = np.array([-e[1], e[0]]) / (np.linalg.norm(e) + 1e-9)
                d = float(n @ (com - a))
                if d < margin:
                    com = com + n * (margin - d)
        return com - com_off

    # ----------------------------------------------------------- swing
    def _swing(self, leg: str, target_xy: np.ndarray, seconds: float = 0.5, height: float = 0.06,
               stance: list[str] | None = None) -> None:
        p0 = np.array([*self.feet_w[leg], self.feet_z[leg]])
        p1 = np.array([target_xy[0], target_xy[1], self.g.foot_z])
        n = max(1, int(seconds / self.env.control_dt))
        for k in range(1, n + 1):
            a = smooth(k / n)
            p = p0 + (p1 - p0) * a
            p[2] = p0[2] + (p1[2] - p0[2]) * a + height * math.sin(math.pi * k / n)
            self._tick(swing={leg: p}, stance_legs=stance)
        # hold briefly so the foot is loaded before it is trusted as support
        for _ in range(4):
            self._tick(swing={leg: p1}, stance_legs=stance)
        self.feet_w[leg] = self.qa.foot_world(leg)[:2].copy()  # measured touchdown point
        self.feet_z[leg] = self.g.foot_z
        self.log.touchdowns.append({"leg": leg, "loaded": self._foot_loaded(leg)})

    # ----------------------------------------------------------- walking
    def nominal_foot(self, leg: str, base: np.ndarray) -> np.ndarray:
        return base[:2] + rot2(base[2]) @ (self.g.nominal[leg] + np.array([0.0, self.g.side[leg] * self.width]))

    def _shift_until_margin(self, stance: list[str], margin: float, timeout: float = 2.0, speed: float = 0.08) -> bool:
        """Move the desired trunk towards the support target until the measured CoM margin holds."""
        good = 0
        for _ in range(int(timeout / self.env.control_dt)):
            m = self.qa.support_margin(stance)
            good = good + 1 if m >= margin else 0
            if good >= 5:
                return True
            tgt = self._support_target(stance, margin + 0.01)
            step = tgt - self.des[:2]
            n = np.linalg.norm(step)
            lim = speed * self.env.control_dt
            if n > lim:
                step *= lim / n
            self.des[:2] += step
            self._tick(stance_legs=stance)
        return False

    def _step_leg(self, leg: str, target_xy: np.ndarray) -> bool:
        stance = [l for l in self.g.legs if l != leg]
        for margin in (self.margin, self.margin * 0.7):
            if self._shift_until_margin(stance, margin):
                self._swing(leg, target_xy, seconds=self.swing_s, stance=stance)
                return True
            self.log.aborted_steps += 1
            centre = np.mean([self.feet_w[l] for l in self.g.legs], axis=0) - rot2(self.des[2]) @ np.mean(list(self.g.nominal.values()), axis=0)
            self._shift_to(centre, self.des[2], 0.4, None)  # back to four-foot centre, then retry
        return False

    @staticmethod
    def crawl_order(forward: float, dyaw: float) -> tuple[str, ...]:
        """Leg sequence matched to the motion: forward/left RL,FL,RR,FR; backward reversed; right turn mirrored."""
        if abs(dyaw) > 0.03:  # turning dominates: order by turn direction
            return CRAWL_ORDER if dyaw > 0 else RIGHT_TURN_ORDER
        if forward < -0.005:
            return ("FR", "RR", "FL", "RL")
        return CRAWL_ORDER

    def walk_to(self, x: float, y: float, yaw: float, max_step: float = 0.06, max_turn: float = 0.2,
                tol: float = 0.02, yaw_tol: float = 0.05, max_cycles: int = 60) -> bool:
        """Body-first static crawl towards a base pose. Returns True if reached within tolerance.

        Each cycle: (1) move/rotate the body a bounded step with all four feet planted,
        (2) step each leg, one at a time, to its nominal spot under the current body after
        shifting until the measured CoM margin holds, (3) re-centre.
        """
        lateral_tol = max(tol, 0.05)  # side-stepping is this gait's weak mode; accept lateral error
        for _ in range(max_cycles):
            b = self.qa.base()
            err = np.array([x - b.pos[0], y - b.pos[1]])
            yerr = wrap_angle(yaw - b.yaw)
            h = np.array([math.cos(b.yaw), math.sin(b.yaw)])
            fwd_err, lat_err = float(err @ h), float(err @ np.array([-h[1], h[0]]))
            if abs(fwd_err) < tol and abs(lat_err) < lateral_tol and abs(yerr) < yaw_tol:
                return True
            # Step from the measured error (closed loop), with anti-wind-up: the desired
            # pose may not run more than a small margin past the target.
            step = err * min(1.0, max_step / (np.linalg.norm(err) + 1e-9))
            dyaw = float(np.clip(yerr, -max_turn, max_turn))
            to_go = wrap_angle(yaw - self.des[2])
            if dyaw * to_go <= 0:
                dyaw = float(np.clip(to_go, -max_turn, max_turn)) if abs(to_go) > 0.02 else 0.0
            elif abs(dyaw) > abs(to_go) + 0.03:
                dyaw = math.copysign(abs(to_go) + 0.03, dyaw)
            dgo = np.array([x - self.des[0], y - self.des[1]])
            if np.linalg.norm(step) > np.linalg.norm(dgo) + 0.02:
                step = step * (np.linalg.norm(dgo) + 0.02) / (np.linalg.norm(step) + 1e-9)
            nxt = np.array([self.des[0] + step[0], self.des[1] + step[1], self.des[2] + dyaw])
            self._shift_to(nxt[:2], nxt[2], self.shift_s, None)  # four feet planted
            fwd = float(step @ np.array([math.cos(self.des[2]), math.sin(self.des[2])]))
            for leg in self.crawl_order(fwd, dyaw):
                tgt = self.nominal_foot(leg, nxt)
                if np.linalg.norm(tgt - self.feet_w[leg]) < 0.012:
                    continue
                self._step_leg(leg, tgt)
                if self.log.fell:
                    return False
            self._shift_to(nxt[:2], nxt[2], 0.3, None)
        return False

    def balance_shift(self, leg: str) -> float:
        """How far (m) the body must move to balance on the other three feet as they are planted now."""
        stance = [l for l in self.g.legs if l != leg]
        return float(np.linalg.norm(self._support_target(stance, margin=0.05) - self.des[:2]))

    def square_up(self, tol: float = 0.04) -> int:
        """Step any foot more than `tol` from its standing spot back under the body, one at a time.
        A learned gait can stop mid-stride (the Go2 stopped with a front foot 13 cm back and the rear feet
        splayed; balancing on three of those feet carried the body so far back that the pushing foot
        couldn't reach). Only worth it when the push is otherwise impossible: re-stepping a stance that
        was fine cost the Go1 balance for nothing (2 falls in 16)."""
        moved = 0
        for leg in self.g.legs:
            tgt = self.nominal_foot(leg, self.des)
            if np.linalg.norm(tgt - self.feet_w[leg]) > tol:
                self._step_leg(leg, tgt)
                moved += 1
                if self.log.fell:
                    break
        return moved

    # ----------------------------------------------------------- pushing
    def push(self, leg: str, contact_xy: np.ndarray, direction: float, distance: float, push_height: float = 0.035,
             speed: float = 0.06) -> dict:
        """Swing `leg` behind contact_xy (along -direction), lower, and push `distance` along `direction`.

        Returns contact evidence. The contact must happen through collision: nothing is attached.
        """
        stance = [l for l in self.g.legs if l != leg]
        u = np.array([math.cos(direction), math.sin(direction)])
        start_contacts = len(self.log.block_contacts)
        # The contact line must be in the leg's reach from where the body stands now AND from where it will
        # stand after shifting over the three stance feet: a foot target beyond reach freezes the leg against
        # the object and then jumps when it becomes reachable again (this flipped narrow blocks; the Go2 hit
        # it when a big balance shift carried the body back). Checking only after the shift let through
        # pushes that tipped the Go1 (3 falls in 16).
        tgt = self._support_target(stance, margin=0.05)
        frames = [self._command_frame(), (tgt, float(self.des[2]))]
        reach = lambda xy: all(self.g.ik(leg, self._trunk_frame(np.asarray(xy), push_height, f)) is not None  # noqa: E731
                               for f in frames)
        pre_ok = reach(np.asarray(contact_xy) - u * 0.05)
        full = distance
        while distance > 0.015 and not reach(np.asarray(contact_xy) + u * distance):
            distance -= 0.01
        if pre_ok and reach(np.asarray(contact_xy) + u * distance):
            self._shift_to(tgt, self.des[2], 0.8, stance)
        else:
            return {"foot_contact_ticks": 0, "max_force_n": 0.0, "other_contacts": [], "skipped": "contact line out of reach",
                    "planned_distance": full, "executed_distance": 0.0}
        p0 = np.array([*self.feet_w[leg], self.feet_z[leg]])
        pre = contact_xy - u * 0.05
        seq = []
        n1 = int(0.5 / self.env.control_dt)
        for k in range(1, n1 + 1):  # lift + move to pre-contact above the floor
            a = smooth(k / n1)
            p = p0 + (np.array([pre[0], pre[1], self.swing_clear]) - p0) * a
            seq.append(p)
        n2 = int(0.3 / self.env.control_dt)
        for k in range(1, n2 + 1):  # lower to push height
            a = smooth(k / n2)
            seq.append(np.array([pre[0], pre[1], self.swing_clear + (push_height - self.swing_clear) * a]))
        total = 0.05 + distance
        n3 = max(1, int(total / speed / self.env.control_dt))
        for k in range(1, n3 + 1):  # push along the line, easing out at the end (an abrupt stop flicked narrow blocks)
            a = k / n3 if k < n3 * 0.7 else 0.7 + 0.3 * smooth((k / n3 - 0.7) / 0.3)
            q = pre + u * total * a
            seq.append(np.array([q[0], q[1], push_height]))
        end = seq[-1]
        n4 = int(0.3 / self.env.control_dt)
        n_back = int(0.5 / self.env.control_dt)
        for k in range(1, n_back + 1):  # pull straight back, slightly downward, before lifting (lifting while still
            a = smooth(k / n_back)       # pressed against the face levered narrow blocks over)
            q = end[:2] - u * 0.05 * a
            seq.append(np.array([q[0], q[1], push_height - 0.004 * a]))
        back = seq[-1][:2]
        for k in range(1, n4 + 1):
            a = smooth(k / n4)
            q = back - u * 0.02 * a
            seq.append(np.array([q[0], q[1], push_height + 0.06 * a]))
        for p in seq:
            self._tick(swing={leg: p}, stance_legs=stance)
            if self.log.fell:
                break
        # put the foot back at its nominal place
        self._swing(leg, self.nominal_foot(leg, self.des), stance=stance)
        centre = np.mean([self.feet_w[l] for l in self.g.legs], axis=0) - rot2(self.des[2]) @ np.mean(list(self.g.nominal.values()), axis=0)
        self._shift_to(centre, self.des[2], 0.3, None)
        cs = self.log.block_contacts[start_contacts:]
        fg = self.g.foot_geom[leg]
        foot = [c for c in cs if c["geom"] == fg]
        return {"foot_contact_ticks": len({c["t"] for c in foot}), "max_force_n": max([c["force_n"] for c in foot], default=0.0),
                "other_contacts": sorted({c["body"] for c in cs if c["geom"] != fg}),
                "planned_distance": full, "executed_distance": round(distance, 3)}
