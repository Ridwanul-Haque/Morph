"""Any goal as numbers: how far the world is from meeting each condition (0 = met).

The placing conditions (on top of, inside, at a spot, ...) give the ready-made skills a target. Everything
else a person might ask for is measured here instead, the same way when checking a result (NumPy) and
while learning it (JAX, as the reward), so what is learned is exactly what is checked:

  orientation(up_face)     right way up / upside down / on its side
  height(min_z, hold)      lifted at least this high (bottom above the floor); hold: the robot may keep holding it
  away_from(target, tol)   at least tol metres from a fixture
  moved(tol, side)         at least tol metres from where it started (optionally towards north/south/east/west)
  not <relation>           not on / not inside / not touching / not near a fixture ("take it out of the bin")
  expression(expr)         anything else, as a formula over measured quantities (Nemotron writes it), e.g.
                           "z > 0.3 and tilt_deg < 20", "dist_to('wall') > 0.4 or held"

Quantities an expression can use (metres, degrees):
  x, y, z (object centre), bottom, top, tilt_deg, yaw_deg, speed, vx, vy, vz (its velocity, m/s), travelled,
  start_x, start_y,
  held (1 if the robot is holding it), dist_to('fixture') (edge to edge), robot_x, robot_y,
  hand_dist (the robot's nearest end part to the object's centre), hand_x, hand_y, hand_z (that end
  part's position), base_z, base_tilt_deg (the robot body's height and tilt: 0 = level), grip (0..1: how close
  the robot's tips are to holding it from opposite sides, and lifting it), touch_dist (m from the nearest tip to
  its surface), lifted (m above whatever is under it), closure (0..1: how far the gripper is commanded shut)
functions: abs, min, max, sqrt, hypot; operators: + - * / < <= > >= == and or not.
"""

from __future__ import annotations

import ast
import math

SIDES = {"east": (1, 0), "west": (-1, 0), "north": (0, 1), "south": (0, -1)}
FACES = {"top": (2, 1), "bottom": (2, -1), "+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1)}
VARIABLES = {"x", "y", "z", "bottom", "top", "tilt_deg", "yaw_deg", "speed", "vx", "vy", "vz", "travelled", "start_x", "start_y",
             "held", "robot_x", "robot_y", "hand_dist", "hand_x", "hand_y", "hand_z", "base_z", "base_tilt_deg",
             "grip", "touch_dist", "lifted", "closure", "spin_dps", "spin_z_dps"}
ROBOT_VARIABLES = {"robot_x", "robot_y", "hand_dist", "hand_x", "hand_y", "hand_z", "base_z", "base_tilt_deg", "grip",
                   "touch_dist"}
FUNCS = {"abs", "min", "max", "sqrt", "hypot", "dist_to"}
GENERIC = ("orientation", "height", "away_from", "moved", "expression")
HOLD_TICKS = 10  # a skill step only counts once it has held for 0.2 s (10 control ticks): no flukes from a bounce
ANGLE_UNIT, LENGTH_UNIT = 90.0, 0.3  # a violation of 90 degrees or 30 cm counts as "1" in the reward


class ExprError(ValueError):
    pass


# ------------------------------------------------------------------ expressions
def normalize(expr: str, object_name: str = "") -> str:
    """Common ways of writing the same thing: && || ! true false, object_x / parcel_x for x, ..."""
    import re
    e = expr.replace("&&", " and ").replace("||", " or ")
    e = re.sub(r"!(?!=)", " not ", e)
    e = re.sub(r"\btrue\b", "1", e, flags=re.I)
    e = re.sub(r"\bfalse\b", "0", e, flags=re.I)
    names = {"object", "obj", "parcel", "box", "block", "item"} | ({object_name.lower()} if object_name else set())
    for n in names:
        for a in ("x", "y", "z", "tilt_deg", "yaw_deg", "speed", "bottom", "top"):
            e = re.sub(rf"\b{re.escape(n)}_{a}\b", a, e)
    return e


_PARSED: dict = {}


def parse(expr: str, fixtures, object_name: str = "") -> ast.Expression:
    """Parse and check an expression: only the quantities, functions and operators above (cached)."""
    key = (expr, tuple(sorted(fixtures)), object_name)
    if key not in _PARSED:
        if len(_PARSED) > 2000:
            _PARSED.clear()
        _PARSED[key] = _parse(expr, fixtures, object_name)
    return _PARSED[key]


def _parse(expr: str, fixtures, object_name: str = "") -> ast.Expression:
    expr = normalize(expr, object_name)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ExprError(f"can't read {expr!r}: {e.msg}") from None
    ok_nodes = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd, ast.BinOp,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Compare, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
                ast.Call, ast.Name, ast.Constant, ast.Load)
    for n in ast.walk(tree):
        if not isinstance(n, ok_nodes):
            raise ExprError(f"{type(n).__name__} isn't allowed in a goal formula")
        if isinstance(n, ast.Name) and n.id not in VARIABLES and n.id not in FUNCS and n.id not in fixtures:
            raise ExprError(f"unknown quantity {n.id!r} (known: {', '.join(sorted(VARIABLES))})")
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name) or n.func.id not in FUNCS:
                raise ExprError("only abs, min, max, sqrt, hypot and dist_to can be called")
            if n.func.id == "dist_to":
                a = n.args[0] if n.args else None
                name = a.value if isinstance(a, ast.Constant) else a.id if isinstance(a, ast.Name) else None
                if name not in fixtures:
                    raise ExprError(f"dist_to needs a fixture of the scene ({', '.join(fixtures)}), got {name!r}")
        if isinstance(n, ast.Constant) and not isinstance(n.value, (int, float, str)):
            raise ExprError(f"{n.value!r} isn't a number")
    return tree


def uses(expr: str) -> set[str]:
    return {n.id for n in ast.walk(ast.parse(expr, mode="eval")) if isinstance(n, ast.Name)}


def _unit(node) -> float:
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    return ANGLE_UNIT if any(v.endswith("_deg") for v in names) else LENGTH_UNIT


def violation(xp, tree: ast.Expression, v: dict, numeric: bool = False):
    """How far from true the formula is (0 when true), in rough units (90 deg ~ 30 cm ~ 1)."""

    def val(n):
        if isinstance(n, ast.Constant):
            return float(n.value)
        if isinstance(n, ast.Name):
            return v[n.id]
        if isinstance(n, ast.UnaryOp):
            return -val(n.operand) if isinstance(n.op, ast.USub) else val(n.operand)
        if isinstance(n, ast.BinOp):
            a, b = val(n.left), val(n.right)
            return {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
                    ast.Div: lambda: a / (b + 1e-9)}[type(n.op)]()
        if isinstance(n, ast.Call):
            f = n.func.id
            if f == "dist_to":
                a = n.args[0]
                return v["dist"][a.value if isinstance(a, ast.Constant) else a.id]
            args = [val(a) for a in n.args]
            if f == "abs":
                return xp.abs(args[0])
            if f == "sqrt":
                return xp.sqrt(xp.maximum(args[0], 0.0))
            if f == "hypot":
                return xp.sqrt(args[0] ** 2 + args[1] ** 2)
            out = args[0]
            for a in args[1:]:
                out = xp.minimum(out, a) if f == "min" else xp.maximum(out, a)
            return out
        if isinstance(n, (ast.Compare, ast.BoolOp)) or (isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not)):
            # a condition inside arithmetic (a reward term like "closure * (touch_dist < 0.03)"): 1 when it holds
            return xp.where(viol(n, False) <= 1e-9, 1.0, 0.0)
        raise ExprError("a comparison can't be used as a number")

    def viol(n, neg):
        if isinstance(n, ast.Expression):
            return viol(n.body, neg)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return viol(n.operand, not neg)
        if isinstance(n, ast.BoolOp):
            parts = [viol(x, neg) for x in n.values]
            conj = isinstance(n.op, ast.And) != neg  # De Morgan
            out = parts[0]
            for p in parts[1:]:
                out = out + p if conj else xp.minimum(out, p)
            return out
        if isinstance(n, ast.Compare):
            total, left = 0.0, n.left
            for op, right in zip(n.ops, n.comparators):
                a, b, u = val(left), val(right), _unit(n)
                if isinstance(op, (ast.Eq, ast.NotEq)):
                    tol = xp.maximum(0.01 if u == LENGTH_UNIT else 2.0, 0.02 * xp.abs(b))
                    d = xp.maximum(xp.abs(a - b) - tol, 0.0)
                    flip = neg != isinstance(op, ast.NotEq)
                    total = total + (xp.maximum(tol - xp.abs(a - b), 0.0) if flip else d) / u
                else:
                    less = isinstance(op, (ast.Lt, ast.LtE)) != neg
                    total = total + xp.maximum(a - b if less else b - a, 0.0) / u
                left = right
            return total
        # a bare quantity used as a condition ("held"): true when > 0.5
        x = val(n)
        return xp.maximum(x - 0.5, 0.0) * 2 if neg else xp.maximum(0.5 - x, 0.0) * 2

    if numeric:
        body = tree.body if isinstance(tree, ast.Expression) else tree
        if isinstance(body, (ast.Compare, ast.BoolOp)) or (isinstance(body, ast.UnaryOp) and isinstance(body.op, ast.Not)):
            return -viol(tree, False)  # a condition: its graded distance from true, as a penalty (0 when it holds)
        return val(body)
    return viol(tree, False)


def term_value(xp, tree: ast.Expression, v: dict):
    """A reward term's value: the number a formula gives, or for a condition minus how far from true it is."""
    return violation(xp, tree, v, numeric=True)


# ------------------------------------------------------------------ the condition kinds
def box_distance(xp, p, lo, hi):
    """Point to axis-aligned box (0 inside)."""
    d = xp.maximum(xp.maximum(lo - p, p - hi), 0.0)
    return xp.sqrt(xp.sum(d * d))


def depth_inside(xp, x, y, lo, hi):
    """How far inside a box's footprint a point is (0 outside)."""
    return xp.maximum(xp.minimum(xp.minimum(x - lo[0], hi[0] - x), xp.minimum(y - lo[1], hi[1] - y)), 0.0)


def condition_violation(xp, c: dict, v: dict, fixtures: dict, half: tuple):
    """One generic condition (a dict, as in task.json) -> violation, 0 when met."""
    k = c["kind"]
    if k == "upright":  # the classic one, measured the same way as "the right way up"
        c, k = {**c, "kind": "orientation", "up_face": "top", "tol": c.get("tol") or 10}, "orientation"
    if k == "orientation":
        axis, sign = FACES.get(c.get("up_face") or "top", (2, 1))
        tol = math.radians(c.get("tol") or 20)
        if c.get("up_face") == "side":  # any side face up: the top axis lies flat
            ang = xp.abs(xp.arcsin(xp.clip(v["R"][2, 2], -1.0, 1.0)))
        else:
            ang = xp.arccos(xp.clip(sign * v["R"][2, axis], -1.0, 1.0))
        return xp.maximum(ang - tol, 0.0) / math.radians(ANGLE_UNIT)
    if k == "height":
        return xp.maximum(float(c.get("min_z") or 0.1) - v["bottom"], 0.0) / LENGTH_UNIT
    if k == "away_from":
        return xp.maximum(float(c.get("tol") or 0.3) - v["dist"][c["target"]], 0.0) / LENGTH_UNIT
    if k == "moved":
        dx, dy = v["x"] - v["start_x"], v["y"] - v["start_y"]
        if c.get("side") in SIDES:
            sx, sy = SIDES[c["side"]]
            along = dx * sx + dy * sy
            return (xp.maximum(float(c.get("tol") or 0.2) - along, 0.0) + 0.5 * xp.abs(dx * sy - dy * sx)
                    * (1.0 if c.get("strict") else 0.0)) / LENGTH_UNIT
        return xp.maximum(float(c.get("tol") or 0.2) - xp.sqrt(dx * dx + dy * dy), 0.0) / LENGTH_UNIT
    if k == "expression":
        return violation(xp, parse(c["expr"], fixtures), v)
    if not c.get("negate") and k in PLACES:  # where it is, as a number (so places can be "at some point" too)
        return place_violation(xp, c, v, fixtures)
    if c.get("negate"):
        f = fixtures[c["target"]]
        if k in ("inside", "on_top_of"):  # leave its footprint
            lo, hi = (f["inner_lo"], f["inner_hi"]) if k == "inside" else (f["top_lo"], f["top_hi"])
            return (depth_inside(xp, v["x"], v["y"], lo, hi) + 0.02 * (depth_inside(xp, v["x"], v["y"], lo, hi) > 0)) / LENGTH_UNIT
        if k == "touching":
            return xp.maximum(0.02 - v["dist"][c["target"]], 0.0) / LENGTH_UNIT
        if k in ("near", "beside"):
            return xp.maximum(float(c.get("tol") or 0.1) - v["dist"][c["target"]], 0.0) / LENGTH_UNIT
    raise ValueError(f"{k} isn't a generic condition")


PLACES = ("inside", "on_top_of", "touching", "near", "beside", "at_position", "heading")


def place_violation(xp, c: dict, v: dict, fixtures: dict):
    """The placing conditions measured like everything else (0 = met), for checking them at a moment during
    the task ("first put it on the shelf, then in the bin"). Same thresholds as goals.check."""
    k, x, y = c["kind"], v["x"], v["y"]
    if k == "at_position":
        return xp.maximum(xp.sqrt((x - c["x"]) ** 2 + (y - c["y"]) ** 2) - (c.get("tol") or 0.05), 0.0) / LENGTH_UNIT
    if k == "heading":
        e = xp.abs((v["yaw_deg"] - c["yaw_deg"] + 180.0) % 360.0 - 180.0)
        return xp.maximum(e - (c.get("tol") or 15), 0.0) / ANGLE_UNIT
    f = fixtures[c["target"]]
    if k in ("inside", "on_top_of"):
        lo, hi = (f["inner_lo"], f["inner_hi"]) if k == "inside" else (f["top_lo"], f["top_hi"])
        surface = f.get("rest_z", 0.0) if k == "inside" else f["top_z"]
        out = xp.sqrt(xp.maximum(xp.maximum(lo[0] - x, x - hi[0]), 0.0) ** 2 + xp.maximum(xp.maximum(lo[1] - y, y - hi[1]), 0.0) ** 2)
        return (out + xp.maximum(xp.abs(v["bottom"] - surface) - 0.012, 0.0)) / LENGTH_UNIT
    d = v["dist"][c["target"]]
    if k == "touching":
        return xp.maximum(d - 0.005, 0.0) / LENGTH_UNIT
    if k == "near":
        return (xp.maximum(d - (c.get("tol") or 0.1), 0.0) + depth_inside(xp, x, y, f["lo"], f["hi"])) / LENGTH_UNIT
    sx, sy = SIDES.get(c.get("side") or "", (0, 0))  # beside
    off = (xp.maximum(f["hi"][0] - x, 0.0) if sx > 0 else xp.maximum(x - f["lo"][0], 0.0) if sx < 0 else 0.0) + \
          (xp.maximum(f["hi"][1] - y, 0.0) if sy > 0 else xp.maximum(y - f["lo"][1], 0.0) if sy < 0 else 0.0)
    return (off + xp.maximum(d - 0.2, 0.0)) / LENGTH_UNIT


def is_generic(c: dict) -> bool:
    """Measured here (not one of the places the ready-made skills aim for)."""
    return c["kind"] in GENERIC or bool(c.get("negate")) or c.get("when", "end") != "end"


def is_temporal(c: dict) -> bool:
    return c.get("when", "end") != "end"


def snapshot(v: dict) -> dict:
    out = {k: float(x) for k, x in v.items() if k not in ("dist", "R")}
    out["dist"] = {n: float(x) for n, x in v["dist"].items()}
    return out


class History:
    """Follows the temporal conditions through a run (NumPy): the value at the moment the robot lets go,
    the worst moment (always), the best moment (ever). check() reads the result."""

    def __init__(self, conditions: list[dict], fixtures: dict, half):
        import numpy as np
        self.np, self.fx, self.half = np, fixtures, half
        self.items = [(i, c) for i, c in enumerate(conditions) if is_temporal(c)]
        self.value: dict[int, float] = {}
        self.snap: dict[int, dict] = {}  # what things looked like at that moment (for the checklist)
        self.was_held = False
        self.released = False
        self.window: dict[int, list] = {}

    def update(self, v: dict) -> None:
        held = float(v["held"]) > 0.5
        let_go = self.was_held and not held
        self.was_held = held
        for i, c in self.items:
            x = float(condition_violation(self.np, {**c, "when": "end"}, v, self.fx, self.half))
            w = c["when"]
            keep = False
            if w == "release":
                keep = let_go  # the last release counts
                self.released |= let_go
            elif w == "always":
                keep = x >= self.value.get(i, -1.0)  # the worst moment
            elif w == "ever":
                keep = x < self.value.get(i, math.inf)  # the best moment
            elif w == "last":  # the worst moment of the last `seconds` (it must keep it up until the end)
                win = self.window.setdefault(i, [])
                win.append(x)
                del win[:-max(1, int(round(float(c.get("seconds") or 2.0) / 0.02)))]
                worst = max(win)
                if worst != self.value.get(i):
                    self.value[i] = worst
                    self.snap[i] = snapshot(v)
            if keep:
                self.value[i] = x
                self.snap[i] = snapshot(v)

    def result(self) -> dict[int, float]:
        """Violation per temporal condition (0 = met); a release condition with no release fails."""
        out = dict(self.value)
        for i, c in self.items:
            out.setdefault(i, math.inf)
        return out


def quantities(xp, *, pos, R, vel, start, half, fixtures: dict, held, robot_xy=None, hand_dist=None,
               hand_pos=None, base_z=None, base_up=None, ang=None) -> dict:
    """ang: the object's angular velocity in its own frame (a free joint's qvel[3:6]), for spin."""
    """Everything a condition or formula can read, from the object's state (NumPy or JAX arrays)."""
    x, y, z = pos[0], pos[1], pos[2]
    extent_z = xp.abs(R[2, 0]) * half[0] + xp.abs(R[2, 1]) * half[1] + xp.abs(R[2, 2]) * half[2]  # half height as it lies
    r_obj = min(half[0], half[1])
    dist = {n: xp.maximum(box_distance(xp, pos, xp.asarray(f["lo"]), xp.asarray(f["hi"])) - r_obj, 0.0)
            for n, f in fixtures.items()}
    v = {"x": x, "y": y, "z": z, "bottom": z - extent_z, "top": z + extent_z,
         "tilt_deg": xp.degrees(xp.arccos(xp.clip(R[2, 2], -1.0, 1.0))),
         "yaw_deg": xp.degrees(xp.arctan2(R[1, 0], R[0, 0])), "speed": xp.sqrt(xp.sum(vel * vel)),
         "vx": vel[0], "vy": vel[1], "vz": vel[2],
         "start_x": start[0], "start_y": start[1],
         "travelled": xp.sqrt((x - start[0]) ** 2 + (y - start[1]) ** 2), "held": held, "dist": dist, "R": R}
    if ang is not None:  # how fast it spins (deg/s), and about the vertical (+ = anticlockwise seen from above)
        w = R @ ang
        v["spin_dps"] = xp.degrees(xp.sqrt(xp.sum(w * w)))
        v["spin_z_dps"] = xp.degrees(w[2])
    else:
        v["spin_dps"] = v["spin_z_dps"] = xp.asarray(0.0)
    if robot_xy is not None:
        v["robot_x"], v["robot_y"] = robot_xy[0], robot_xy[1]
    if hand_dist is not None:
        v["hand_dist"] = hand_dist
    if hand_pos is not None:
        v["hand_x"], v["hand_y"], v["hand_z"] = hand_pos[0], hand_pos[1], hand_pos[2]
    if base_z is not None:
        v["base_z"] = base_z
    if base_up is not None:  # the body's z axis in the world: level when it points straight up
        v["base_tilt_deg"] = xp.degrees(xp.arccos(xp.clip(base_up, -1.0, 1.0)))
    return v


def fixtures_geometry(scene) -> dict:
    """The fixtures as plain numbers (what task.json stores for the learner)."""
    out = {}
    for n, f in scene.fixtures.items():
        out[n] = {"kind": f.kind, "lo": [float(a) for a in f.lo], "hi": [float(a) for a in f.hi],
                  "top_lo": [float(a) for a in f.top_lo], "top_hi": [float(a) for a in f.top_hi],
                  "inner_lo": [float(a) for a in (f.inner_lo if f.inner_lo is not None else f.lo)],
                  "inner_hi": [float(a) for a in (f.inner_hi if f.inner_hi is not None else f.hi)],
                  "top_z": float(f.top_z), "rest_z": float(f.rest_z)}
    return out
