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
  roll_deg, pitch_deg (the object's tilt about its own long and short axes), goal_x, goal_y, goal_z (where the goal
  puts it, when it has a place; target_x ... mean the same), <fixture>_x, <fixture>_y, <fixture>_top (a fixture's
  centre and top, e.g. bin_x)
functions: abs, min, max, sqrt, hypot, exp, log, tanh, cos, sin, atan2 (radians), cosd, sind, atan2d (degrees),
  clip(x, lo, hi), pow(x, n); constant pi; operators: + - * / ** (or ^, a power with a number as the exponent)
  < <= > >= == and or not.
"""

from __future__ import annotations

import ast
import math

SIDES = {"east": (1, 0), "west": (-1, 0), "north": (0, 1), "south": (0, -1)}
FACES = {"top": (2, 1), "bottom": (2, -1), "+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1)}
VARIABLES = {"x", "y", "z", "bottom", "top", "tilt_deg", "yaw_deg", "speed", "vx", "vy", "vz", "travelled", "start_x", "start_y",
             "held", "robot_x", "robot_y", "hand_dist", "hand_x", "hand_y", "hand_z", "base_z", "base_tilt_deg",
             "grip", "touch_dist", "lifted", "closure", "spin_dps", "spin_z_dps", "roll_deg", "pitch_deg",
             "goal_x", "goal_y", "goal_z"}
GOAL_VARIABLES = {"goal_x", "goal_y", "goal_z"}  # only when the goal has a place (else not a number: the check says so)
FIXTURE_ATTRS = ("_x", "_y", "_top")  # bin_x, bin_y, bin_top: a fixture's centre and top
ROBOT_VARIABLES = {"robot_x", "robot_y", "hand_dist", "hand_x", "hand_y", "hand_z", "base_z", "base_tilt_deg", "grip",
                   "touch_dist"}
# the robot's own body (goals about the robot, not the object: "make a fist", "raise the arm", "lift a foot"): each takes
# a name pattern ("pip", "index|middle", "foot") matched against the robot's joint / end-part names. joint_frac: how far
# the matching joints are through their range, 0 at the lower limit, 1 at the upper (mean; _min / _max: the least / most).
# part_x / part_y / part_z: where the matching end parts (fingertips, hand, feet) are (their mean), in metres.
BODY_FUNCS = {"joint_frac", "joint_frac_min", "joint_frac_max", "part_x", "part_y", "part_z"}
# the maths a shaping term needs (Eureka-style terms are often exp(-d / scale), tanh, a squared error, an angle)
MATH_FUNCS = {"exp", "log", "tanh", "cos", "sin", "atan2", "cosd", "sind", "atan2d", "clip", "pow"}
FUNCS = {"abs", "min", "max", "sqrt", "hypot", "dist_to"} | MATH_FUNCS | BODY_FUNCS
CONSTANTS = {"pi": math.pi}
MAX_POWER = 4.0  # x ** n with a number n, |n| <= 4 (a squared or cubed error; not an overflowing x ** 100)
GENERIC = ("orientation", "height", "away_from", "moved", "expression")
HOLD_TICKS = 10  # a skill step only counts once it has held for 0.2 s (10 control ticks): no flukes from a bounce
ANGLE_UNIT, LENGTH_UNIT = 90.0, 0.3  # a violation of 90 degrees or 30 cm counts as "1" in the reward


class ExprError(ValueError):
    pass


# ------------------------------------------------------------------ expressions
def normalize(expr: str, object_name: str = "") -> str:
    """Common ways of writing the same thing: && || ! true false, object_x / parcel_x for x, target_x for goal_x,
    clamp / clipped for clip, ..."""
    import re
    if isinstance(expr, bool):  # a step written as "done_when": true
        return "1" if expr else "0"
    e = str(expr).replace("&&", " and ").replace("||", " or ")
    e = re.sub(r"!(?!=)", " not ", e)
    e = re.sub(r"\btrue\b", "1", e, flags=re.I)
    e = re.sub(r"\bfalse\b", "0", e, flags=re.I)
    names = {"object", "obj", "parcel", "box", "block", "item"} | ({object_name.lower()} if object_name else set())
    for n in names:
        for a in ("x", "y", "z", "tilt_deg", "yaw_deg", "speed", "bottom", "top"):
            e = re.sub(rf"\b{re.escape(n)}_{a}\b", a, e)
    e = re.sub(r"\b(?:target|goal|place)_(x|y|z)\b", r"goal_\1", e)
    e = re.sub(r"\b(?:clamp|clipped|clamped)\s*\(", "clip(", e)
    e = re.sub(r"\b(?:math|np|numpy|jnp)\.", "", e)  # math.exp(...) -> exp(...)
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
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.BitXor, ast.Compare, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
                ast.Eq, ast.NotEq, ast.Call, ast.Name, ast.Constant, ast.Load)
    for n in ast.walk(tree):
        if not isinstance(n, ok_nodes):
            raise ExprError(f"{type(n).__name__} isn't allowed in a goal formula")
        if isinstance(n, ast.Name) and n.id not in VARIABLES and n.id not in FUNCS and n.id not in fixtures \
                and n.id not in CONSTANTS and not _fixture_attr(n.id, fixtures):
            fx = [f"{f}{a}" for f in fixtures for a in FIXTURE_ATTRS][:9]
            raise ExprError(f"unknown quantity {n.id!r} (known: {', '.join(sorted(VARIABLES) + fx)})")
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Pow, ast.BitXor)):  # x ** 2 (x ^ 2 means the same here)
            _power(n.right)
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name) or n.func.id not in FUNCS:
                raise ExprError(f"only these can be called: {', '.join(sorted(FUNCS - BODY_FUNCS))} (and "
                                f"{', '.join(sorted(BODY_FUNCS))} for the robot's own body)")
            arity = {"atan2": 2, "atan2d": 2, "clip": 3, "pow": 2}.get(n.func.id, 1 if n.func.id in MATH_FUNCS else None)
            if arity is not None and len(n.args) != arity:
                raise ExprError(f"{n.func.id} takes {arity} value{'s' if arity > 1 else ''}")
            if n.func.id == "pow":
                _power(n.args[1])
            if n.func.id in BODY_FUNCS:
                a = n.args[0] if len(n.args) == 1 else None
                if not (isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.strip()):
                    raise ExprError(f"{n.func.id} needs one name pattern in quotes, e.g. {n.func.id}('index')")
            if n.func.id == "dist_to":
                a = n.args[0] if n.args else None
                name = a.value if isinstance(a, ast.Constant) else a.id if isinstance(a, ast.Name) else None
                if name not in fixtures:
                    raise ExprError(f"dist_to needs a fixture of the scene ({', '.join(fixtures)}), got {name!r}")
        if isinstance(n, ast.Constant) and not isinstance(n.value, (int, float, str)):
            raise ExprError(f"{n.value!r} isn't a number")
    return tree


def _fixture_attr(name: str, fixtures) -> bool:
    return any(name == f"{f}{a}" for f in fixtures for a in FIXTURE_ATTRS)


def _power(node) -> float:
    """The exponent of a power: a plain number, at most MAX_POWER either way."""
    k = node.operand if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) else node
    if not (isinstance(k, ast.Constant) and isinstance(k.value, (int, float)) and not isinstance(k.value, bool)):
        raise ExprError("a power's exponent must be a number (x ** 2), not a quantity")
    n = -float(k.value) if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) else float(k.value)
    if abs(n) > MAX_POWER:
        raise ExprError(f"a power's exponent must be between -{MAX_POWER:g} and {MAX_POWER:g}")
    return n


def _pow(xp, a, n: float):
    """a ** n without NaN or infinity: a fractional power of |a|, a negative power of a kept off zero."""
    if n != int(n):
        a = xp.abs(a)
    if n < 0:
        a = xp.where(xp.abs(a) < 1e-6, 1e-6, a)
    return a ** n


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
            if n.id in CONSTANTS:
                return CONSTANTS[n.id]
            if n.id not in v and (n.id in GOAL_VARIABLES or n.id.endswith(FIXTURE_ATTRS)):
                # a goal with no place (goal_x) or a fixture this model doesn't know: not a number, so not "met"
                return xp.asarray(float("nan") if xp.__name__ == "numpy" else 0.0)
            return v[n.id]
        if isinstance(n, ast.UnaryOp):
            return -val(n.operand) if isinstance(n.op, ast.USub) else val(n.operand)
        if isinstance(n, ast.BinOp):
            if isinstance(n.op, (ast.Pow, ast.BitXor)):
                return _pow(xp, val(n.left), _power(n.right))
            a, b = val(n.left), val(n.right)
            return {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
                    ast.Div: lambda: a / (b + 1e-9)}[type(n.op)]()
        if isinstance(n, ast.Call):
            f = n.func.id
            if f in BODY_FUNCS:
                return body_value(xp, f, n.args[0].value, v)
            if f == "dist_to":
                a = n.args[0]
                return v["dist"][a.value if isinstance(a, ast.Constant) else a.id]
            if f == "pow":
                return _pow(xp, val(n.args[0]), _power(n.args[1]))
            args = [val(a) for a in n.args]
            if f in MATH_FUNCS:
                a = args[0]
                return {"exp": lambda: xp.exp(xp.clip(a, -50.0, 50.0)), "log": lambda: xp.log(xp.maximum(a, 1e-9)),
                        "tanh": lambda: xp.tanh(a), "cos": lambda: xp.cos(a), "sin": lambda: xp.sin(a),
                        "cosd": lambda: xp.cos(xp.radians(a)), "sind": lambda: xp.sin(xp.radians(a)),
                        "atan2": lambda: xp.arctan2(a, args[-1]), "atan2d": lambda: xp.degrees(xp.arctan2(a, args[-1])),
                        "clip": lambda: xp.clip(a, args[1], args[2])}[f]()
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


def body_names(pattern: str, names) -> list[str]:
    """The names a pattern picks out: any of its |-separated words appearing in the name (case-insensitive)."""
    import fnmatch
    import re
    pat = pattern.strip().lower()
    if any(ch in pat for ch in "()[]^$+"):  # written as a regular expression: "(index|middle)_(pip|dip)|thumb_ip"
        try:
            rx = re.compile(pat)
        except re.error:
            rx = None
        if rx is not None:
            full = [n for n in names if rx.fullmatch(n.lower())]
            return full or [n for n in names if rx.search(n.lower())]
    alts = [a.strip() for a in pat.split("|") if a.strip()]

    def hit(name: str, a: str) -> bool:  # "pip" anywhere in the name, or a wildcard pattern ("*_pip", "index_*")
        return fnmatch.fnmatch(name, a if a.startswith("*") or a.endswith("*") else f"*{a}*") if "*" in a or "?" in a \
            else a in name
    return [n for n in names if any(hit(n.lower(), a) for a in alts)]


def body_value(xp, f: str, pattern: str, v: dict):
    """joint_frac / part_x ... for this robot (see BODY_FUNCS). NaN if it has no joint or part of that name (NumPy;
    the checklist then says so), 0 under JAX (Morph doesn't train for a body that can't be measured)."""
    if f.startswith("joint_frac"):
        src = v.get("joints") or {}
        vals = [src[k] for k in body_names(pattern, src)]
        if not vals:
            return xp.asarray(float("nan") if xp is not None and xp.__name__ == "numpy" else 0.0)
        arr = xp.stack([xp.asarray(x) for x in vals])
        return {"joint_frac": xp.mean, "joint_frac_min": xp.min, "joint_frac_max": xp.max}[f](arr)
    src = v.get("parts") or {}
    vals = [src[k] for k in body_names(pattern, src)]
    if not vals:
        return xp.asarray(float("nan") if xp.__name__ == "numpy" else 0.0)
    return xp.mean(xp.stack([xp.asarray(x) for x in vals]), axis=0)["xyz".index(f[-1])]


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
    """The measurements at one moment, as plain numbers (for the checklist's "at that moment: ...")."""
    out = {}
    for k, x in v.items():
        if k in ("dist", "R", "parts"):
            continue
        if k == "joints":
            out[k] = {n: float(j) for n, j in x.items()}
        elif isinstance(x, dict) or getattr(x, "ndim", 0):
            continue  # not a single number
        else:
            out[k] = float(x)
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
               hand_pos=None, base_z=None, base_up=None, ang=None, goal=None) -> dict:
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
         "travelled": xp.sqrt((x - start[0]) ** 2 + (y - start[1]) ** 2), "held": held, "dist": dist, "R": R,
         # tilt about its own x axis (roll) and y axis (pitch); tilt_deg is the total
         "roll_deg": xp.degrees(xp.arctan2(R[2, 1], R[2, 2])),
         "pitch_deg": xp.degrees(xp.arcsin(xp.clip(-R[2, 0], -1.0, 1.0)))}
    for n, f in fixtures.items():  # each fixture's centre and top (bin_x, bin_y, bin_top)
        v[f"{n}_x"], v[f"{n}_y"] = (f["lo"][0] + f["hi"][0]) / 2, (f["lo"][1] + f["hi"][1]) / 2
        v[f"{n}_top"] = f.get("top_z", f["hi"][2] if len(f["hi"]) > 2 else 0.0)
    if goal is not None:  # where the goal puts it, when it has a place
        v["goal_x"], v["goal_y"], v["goal_z"] = goal[0], goal[1], goal[2]
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


# ------------------------------------------------------------------ formulas in plain words (for the app)
# quantity -> (words, unit): "m" is shown in cm, "deg" in degrees, "frac" 0..1, "flag" yes / no, "" as is
WORDS = {
    "x": ("the object's x", "m"), "y": ("the object's y", "m"), "z": ("the object's centre height", "m"),
    "bottom": ("the object's bottom", "m"), "top": ("the object's top", "m"),
    "tilt_deg": ("the object's tilt", "deg"), "yaw_deg": ("the object's heading", "deg"),
    "roll_deg": ("the object's roll", "deg"), "pitch_deg": ("the object's pitch", "deg"),
    "speed": ("the object's speed", "m/s"), "vx": ("its speed east", "m/s"), "vy": ("its speed north", "m/s"),
    "vz": ("its speed up", "m/s"), "travelled": ("how far it has moved", "m"),
    "held": ("the robot holding it", "flag"), "grip": ("the grip score (0 to 1)", "frac"),
    "closure": ("the gripper closed", "pct"), "touch_dist": ("the nearest fingertip's gap to it", "m"),
    "lifted": ("it lifted", "m"), "hand_dist": ("the hand's distance to it", "m"),
    "hand_x": ("the hand's x", "m"), "hand_y": ("the hand's y", "m"), "hand_z": ("the hand's height", "m"),
    "robot_x": ("the robot's x", "m"), "robot_y": ("the robot's y", "m"), "base_z": ("the robot body's height", "m"),
    "base_tilt_deg": ("the robot body's tilt", "deg"), "spin_dps": ("how fast it spins", "deg/s"),
    "goal_x": ("the goal's x", "m"), "goal_y": ("the goal's y", "m"), "goal_z": ("the goal's height", "m"),
}
_OPS = {ast.Lt: "under", ast.LtE: "at most", ast.Gt: "over", ast.GtE: "at least", ast.Eq: "about", ast.NotEq: "not"}


def describe_formula(expr: str, object_name: str = "the object") -> str:
    """A step's done_when / keep in plain words: "hand_dist < 0.3 and base_tilt_deg < 10" -> "the hand's distance to the
    parcel under 30 cm and the robot body's tilt under 10°". Parts it can't put simply are shown as written."""
    try:
        tree = ast.parse(normalize(expr, object_name if object_name != "the object" else ""), mode="eval").body
    except SyntaxError:
        return str(expr)
    obj = object_name if object_name.startswith("the ") else f"the {object_name}"

    def word(n) -> tuple[str, str] | None:
        if isinstance(n, ast.Name) and n.id in WORDS:
            w, u = WORDS[n.id]
            w = w.replace("the object", obj)
            return (obj + w[2:] if w.startswith("it ") else w.replace(" it", f" {obj}")), u
        if isinstance(n, ast.Name) and _fixture_attr(n.id, {n.id.rsplit("_", 1)[0]}):
            f, a = n.id.rsplit("_", 1)
            return f"the {f}'s {'top' if a == 'top' else a}", "m"
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "hypot" and len(n.args) == 2
                and all(isinstance(a, ast.BinOp) and isinstance(a.op, ast.Sub) and isinstance(a.left, ast.Name)
                        and isinstance(a.right, ast.Name) for a in n.args)):  # hypot(x - goal_x, y - goal_y)
            def thing(name: str) -> str:
                base = name[:-2] if name.endswith(("_x", "_y")) else ""
                return {"": obj, "hand": "the hand", "robot": "the robot", "goal": "the goal",
                        "start": "where it started"}.get(base, f"the {base}")
            return f"{thing(n.args[0].left.id)}'s distance to {thing(n.args[0].right.id)}", "m"
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "dist_to" and n.args:
            a = n.args[0]
            return f"{obj}'s distance to the {a.value if isinstance(a, ast.Constant) else getattr(a, 'id', '?')}", "m"
        return None

    def number(v: float, unit: str) -> str:
        if unit == "m":
            return f"{v * 100:.3g} cm"
        if unit == "deg":
            return f"{v:.3g}°"
        if unit == "frac":
            return f"{v:.2g}"
        if unit == "pct":
            return f"{v * 100:.0f}%"
        return f"{v:.3g}" + (f" {unit}" if unit not in ("", "flag") else "")

    def value(n, unit: str) -> str | None:
        """A side of a comparison: a quantity, a number (in the other side's unit), or one plus or minus a number."""
        w = word(n)
        if w:
            return w[0]
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return number(float(n.value), unit)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) and isinstance(n.operand, ast.Constant):
            return number(-float(n.operand.value), unit)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)):
            a, b = value(n.left, unit), value(n.right, unit)
            if a and b:
                return f"{a} {'+' if isinstance(n.op, ast.Add) else '−'} {b}"
        return None

    def unit_of(n) -> str:
        for x in ast.walk(n):
            w = word(x)
            if w:
                return w[1]
        return ""

    def text(n) -> str:
        if isinstance(n, ast.BoolOp):
            return (" and " if isinstance(n.op, ast.And) else " or ").join(text(v) for v in n.values)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            w = word(n.operand)
            if w and w[1] == "flag":  # "not held"
                return w[0].replace("the robot holding", "the robot not holding")
            return f"not ({text(n.operand)})"
        w = word(n)
        if w and w[1] == "flag":  # a bare "held"
            return w[0]
        if isinstance(n, ast.Compare) and len(n.ops) == 1:
            left, right, op = n.left, n.comparators[0], type(n.ops[0])
            const = right.value if isinstance(right, ast.Constant) and isinstance(right.value, (int, float)) else None
            w = word(left)
            if w and const is not None:
                if w[1] == "flag":
                    on = (op in (ast.Eq, ast.GtE, ast.Gt) and const >= 0.5) or (op in (ast.Lt, ast.LtE, ast.NotEq) and const <= 0.5)
                    return w[0] if on else w[0].replace("the robot holding", "the robot not holding")
                return f"{w[0]} {_OPS[op]} {number(float(const), w[1])}"
            # abs(a - b) < c: a within c of b
            if (isinstance(left, ast.Call) and isinstance(left.func, ast.Name) and left.func.id == "abs" and left.args
                    and isinstance(left.args[0], ast.BinOp) and isinstance(left.args[0].op, ast.Sub)
                    and op in (ast.Lt, ast.LtE) and const is not None):
                u = unit_of(left)
                a, b = value(left.args[0].left, u), value(left.args[0].right, u)
                if a and b:
                    return f"{a} within {number(float(const), u)} of {b}"
            u = unit_of(n)
            a, b = value(left, u), value(right, u)
            if a and b:
                return f"{a} {_OPS[op]} {b}"
        return ast.unparse(n)

    return text(tree)
