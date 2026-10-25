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
  held (1 if the robot is holding it, pinched from opposite sides), in_hand (1 while it's off the floor and anything
  else it could rest on: kept up by the robot, even rolling or sliding between the fingers), dist_to('fixture') (edge
  to edge), robot_x, robot_y,
  hand_dist (the robot's nearest end part to the object's centre), hand_x, hand_y, hand_z (that end
  part's position), base_z, base_tilt_deg (the robot body's height and tilt: 0 = level), grip (0..1: how close
  the robot's tips are to holding it from opposite sides, and lifting it), touch_dist (m from the nearest tip to
  its surface), lifted (m above whatever is under it), closure (0..1: how far the gripper is commanded shut),
  grip_gap (m from the hand's point to where Morph's own planned grip puts it on the object: near 0 means the open
  hand is where closing it grips; hand_dist when there's no planned grip)
  fingers_touching (how many of the robot's fingers touch it; with no hand, how many of its parts),
  roll_deg, pitch_deg (the object's tilt about its own long and short axes), goal_x, goal_y, goal_z (where the goal
  puts it, when it has a place; target_x ... mean the same), <fixture>_x, <fixture>_y, <fixture>_top (a fixture's
  centre and top, e.g. bin_x)
functions: abs, min, max, sqrt, hypot, exp, log, tanh, cos, sin, atan2 (radians), cosd, sind, atan2d (degrees),
  clip(x, lo, hi), pow(x, n); constant pi; operators: + - * / ** (or ^, a power with a number as the exponent)
  < <= > >= == and or not.
over time (what one moment can't show; followed through the run by the judge and the learner alike):
  change(q, s)   how much q changed over the last s seconds (an angle counted through whole turns: two turns of yaw_deg
                 are 720, and turning back and forth adds up to about 0); change(q): since the try began
  mean(q, s)     q's average over the last s seconds (a condition counts 1 while it holds: mean(touching('thumb'), 2) is
                 the share of the time the thumb touched it); mean(q): over the whole try
"""

from __future__ import annotations

import ast
import math

SIDES = {"east": (1, 0), "west": (-1, 0), "north": (0, 1), "south": (0, -1)}
FACES = {"top": (2, 1), "bottom": (2, -1), "+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1)}
VARIABLES = {"x", "y", "z", "bottom", "top", "tilt_deg", "yaw_deg", "speed", "vx", "vy", "vz", "travelled", "start_x", "start_y",
             "held", "robot_x", "robot_y", "hand_dist", "hand_x", "hand_y", "hand_z", "base_z", "base_tilt_deg",
             "grip", "touch_dist", "lifted", "closure", "spin_dps", "spin_z_dps", "roll_deg", "pitch_deg",
             "goal_x", "goal_y", "goal_z", "fingers_touching", "grip_gap", "in_hand"}
GOAL_VARIABLES = {"goal_x", "goal_y", "goal_z"}  # only when the goal has a place (else not a number: the check says so)
FIXTURE_ATTRS = ("_x", "_y", "_top")  # bin_x, bin_y, bin_top: a fixture's centre and top
ROBOT_VARIABLES = {"robot_x", "robot_y", "hand_dist", "hand_x", "hand_y", "hand_z", "base_z", "base_tilt_deg", "grip",
                   "touch_dist", "grip_gap"}
# the robot's own body (goals about the robot, not the object: "make a fist", "raise the arm", "lift a foot"): each takes
# a name pattern ("pip", "index|middle", "foot") matched against the robot's joint / end-part names. joint_frac: how far
# the matching joints are through their range, 0 at the lower limit, 1 at the upper (mean; _min / _max: the least / most).
# part_x / part_y / part_z: where the matching end parts (fingertips, hand, feet) are (their mean), in metres.
# touching: 1 if any matching finger ("thumb", "index|middle") or body touches the object. Judged on the contacts; the
# learner on the GPU only sees fingertips, counted as touching within 6 mm of the surface.
BODY_FUNCS = {"joint_frac", "joint_frac_min", "joint_frac_max", "part_x", "part_y", "part_z", "touching"}
# the maths a shaping term needs (Eureka-style terms are often exp(-d / scale), tanh, a squared error, an angle)
MATH_FUNCS = {"exp", "log", "tanh", "cos", "sin", "atan2", "cosd", "sind", "atan2d", "clip", "pow"}
# over time (Windows): change over / average over the last s seconds, or since the try began
WINDOW_FUNCS = {"change", "mean"}
WINDOW_DT, MAX_WINDOW_S = 0.02, 60.0  # s per control tick; the longest window
TURN_UNIT = 360.0  # a whole turn short counts as "1" in the reward (turning is counted in hundreds of degrees)
FUNCS = {"abs", "min", "max", "sqrt", "hypot", "dist_to"} | MATH_FUNCS | BODY_FUNCS | WINDOW_FUNCS
CONSTANTS = {"pi": math.pi}
MAX_POWER = 4.0  # x ** n with a number n, |n| <= 4 (a squared or cubed error; not an overflowing x ** 100)
GENERIC = ("orientation", "height", "away_from", "moved", "expression")
IN_HAND_LIFT = 0.01  # m: in_hand once its bottom is this far above whatever it would rest on
HOLD_TICKS = 10  # a skill step only counts once it has held for 0.2 s (10 control ticks): no flukes from a bounce
ANGLE_UNIT, LENGTH_UNIT = 90.0, 0.3  # a violation of 90 degrees or 30 cm counts as "1" in the reward
SPIN_UNIT, FLAG_UNIT = 90.0, 1.0  # ... and so does 90 deg/s, or a yes/no (touching, held) that is the wrong way round
FLAGS = {"held", "grip", "closure", "fingers_touching", "in_hand"}


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
            if n.func.id in WINDOW_FUNCS:
                _window_args(n)
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


def _window_args(n: ast.Call) -> None:
    """change(q, s) / mean(q, s): a quantity (not itself over time) and a number of seconds, or just the quantity."""
    f = n.func.id
    if not 1 <= len(n.args) <= 2 or n.keywords:
        raise ExprError(f"{f} takes a quantity and a number of seconds ({f}(yaw_deg, 4)), or just the quantity")
    if any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id in WINDOW_FUNCS
           for x in ast.walk(n.args[0])):
        raise ExprError(f"{f}(...) can't contain another change(...) or mean(...)")
    if len(n.args) == 2:
        s = n.args[1]
        if not (isinstance(s, ast.Constant) and isinstance(s.value, (int, float)) and not isinstance(s.value, bool)
                and 0 < s.value <= MAX_WINDOW_S):
            raise ExprError(f"{f}'s seconds must be a number above 0 and at most {MAX_WINDOW_S:g}")


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
    return {n.id for n in ast.walk(ast.parse(normalize(expr), mode="eval")) if isinstance(n, ast.Name)}


def _unit(node) -> float:
    """What counts as 1 for a comparison: by what it compares. Spin in deg/s once shared the 30 cm unit, so missing
    60 deg/s weighed as much as being 18 m off and swamped every other part of the reward."""
    shares = [x for x in ast.walk(node) if isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
              and x.func.id == "mean" and x.args and _is_condition(x.args[0])]
    inner = {id(y) for x in shares for y in ast.walk(x)}
    calls = {n.func.id for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and id(n) not in inner}
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and id(n) not in inner} - calls
    if shares and not names and calls <= WINDOW_FUNCS | {"abs", "min", "max"}:  # a share of the time (0..1)
        return FLAG_UNIT
    turns = [x for x in ast.walk(node) if isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
             and x.func.id == "change" and x.args and _is_angle(x.args[0])]
    if turns:  # how far it turned: 90 deg short of 720 is nearly there, not as far off as 90 deg of tilt
        return TURN_UNIT
    calls -= WINDOW_FUNCS
    if any(v.endswith("_dps") for v in names):
        return SPIN_UNIT
    if any(v.endswith("_deg") for v in names):
        return ANGLE_UNIT
    if (names or calls) and names <= FLAGS and calls <= {"touching"}:
        return FLAG_UNIT
    return LENGTH_UNIT


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
            if f in WINDOW_FUNCS:  # followed through the run (Windows); without the run, not a number
                w = v.get("window") or {}
                k = ast.dump(n)
                return w[k] if k in w else xp.asarray(float("nan") if xp.__name__ == "numpy" else 0.0)
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
                    tol = xp.maximum(2.0 if u in (ANGLE_UNIT, SPIN_UNIT) else 0.01, 0.02 * xp.abs(b))
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
    if f == "touching":
        src = v.get("touch") or {}
        vals = [src[k] for k in body_names(pattern, [k for k in src if k != "fingers_touching"])]
        if not vals:
            return xp.asarray(float("nan") if xp.__name__ == "numpy" else 0.0)
        return xp.max(xp.stack([xp.asarray(x, dtype=float) for x in vals]))
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


def missing_parts(conditions: list[dict], names: dict[str, list[str]]) -> list[str]:
    """Name patterns in the goal's formulas that match nothing on this robot (fingers.part_names), e.g. 'thumb' in
    touching('thumb') on a parallel gripper."""
    out = []
    for c in conditions:
        if not c.get("expr"):
            continue
        try:
            tree = ast.parse(normalize(c["expr"]), mode="eval")
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in BODY_FUNCS and n.args \
                    and isinstance(n.args[0], ast.Constant):
                pat = str(n.args[0].value)
                src = names["touch"] if n.func.id == "touching" else names["joints"] if n.func.id.startswith("joint") \
                    else names["parts"]
                if not body_names(pat, src):
                    out.append(pat)
    return sorted(set(out))


MOTION = {"speed", "vx", "vy", "vz", "spin_dps", "spin_z_dps"}


def needs_motion_at_rest(c: dict) -> bool:
    """A formula only about how the object moves that fails while it is still ("spin_dps > 60"): it can't hold the
    whole time, since every try starts with the object at rest."""
    if c.get("kind") != "expression" or not c.get("expr"):
        return False
    try:
        names = uses(normalize(c["expr"]))
    except SyntaxError:
        return False
    if not names or not names <= MOTION:
        return False
    return float(violation(__import__("numpy"), parse(c["expr"], {}), {n: 0.0 for n in MOTION})) > 1e-9


def term_value(xp, tree: ast.Expression, v: dict):
    """A reward term's value: the number a formula gives, or for a condition minus how far from true it is."""
    return violation(xp, tree, v, numeric=True)


# ------------------------------------------------------------------ rules that can never hold
INF = float("inf")
RANGES = {"yaw_deg": (-180.0, 180.0), "roll_deg": (-180.0, 180.0), "pitch_deg": (-90.0, 90.0), "tilt_deg": (0.0, 180.0),
          "base_tilt_deg": (0.0, 180.0), "held": (0.0, 1.0), "in_hand": (0.0, 1.0), "grip": (0.0, 1.0),
          "closure": (0.0, 1.0), "speed": (0.0, INF), "spin_dps": (0.0, INF), "travelled": (0.0, INF),
          "hand_dist": (0.0, INF), "touch_dist": (0.0, INF), "lifted": (0.0, INF), "grip_gap": (0.0, INF),
          "fingers_touching": (0.0, INF)}
UNIT_FUNCS = {"touching", "joint_frac", "joint_frac_min", "joint_frac_max"}  # 0..1


def value_range(node) -> tuple[float, float]:
    """The values a formula can take at all (-inf..inf when it can't be told)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value), float(node.value)
    if isinstance(node, ast.Name):
        return RANGES.get(node.id, (-INF, INF))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        lo, hi = value_range(node.operand)
        return -hi, -lo
    if _is_condition(node):
        return 0.0, 1.0
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.args:
        f = node.func.id
        if f in UNIT_FUNCS:
            return 0.0, 1.0
        if f == "dist_to":
            return 0.0, INF
        if f == "abs":
            lo, hi = value_range(node.args[0])
            return (0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi))), max(abs(lo), abs(hi))
        if f == "mean":
            return value_range(node.args[0])
        if f == "change" and not _is_angle(node.args[0]):  # an angle goes round: its change has no limit
            lo, hi = value_range(node.args[0])
            return lo - hi, hi - lo
    return -INF, INF


def never_holds(expr: str, fixtures=()) -> list[str]:
    """Parts of a formula that can never be true, whatever the robot does ("abs(yaw_deg) >= 360": a heading only goes
    to 180). An "or" counts only when every side of it can't."""
    try:
        tree = parse(expr, fixtures)
    except ExprError:
        return []

    def bad(n) -> list[str]:
        if isinstance(n, ast.BoolOp):
            parts = [bad(x) for x in n.values]
            if isinstance(n.op, ast.And):
                return [m for p in parts for m in p]
            return [m for p in parts for m in p] if all(parts) else []
        if not (isinstance(n, ast.Compare) and len(n.ops) == 1):
            return []
        left, right, op = n.left, n.comparators[0], n.ops[0]
        flip = {ast.Lt: ast.Gt, ast.LtE: ast.GtE, ast.Gt: ast.Lt, ast.GtE: ast.LtE, ast.Eq: ast.Eq, ast.NotEq: ast.NotEq}
        if isinstance(left, ast.Constant) and not isinstance(right, ast.Constant):
            left, right, op = right, left, flip[type(op)]()
        if not (isinstance(right, ast.Constant) and isinstance(right.value, (int, float))):
            return []
        c = float(right.value)
        lo, hi = value_range(left)
        never = {ast.Gt: hi <= c, ast.GtE: hi < c, ast.Lt: lo >= c, ast.LtE: lo > c,
                 ast.Eq: not (lo - 1e-6 <= c <= hi + 1e-6), ast.NotEq: lo == hi == c}[type(op)]
        if not never:
            return []
        span = f"from {lo:g} up" if hi == INF else f"down to {hi:g}" if lo == -INF else f"from {lo:g} to {hi:g}"
        why = f"{ast.unparse(n)} can never hold: {ast.unparse(left)} only goes {span}"
        if c > hi and any(isinstance(x, ast.Name) and x.id in ("yaw_deg", "roll_deg", "pitch_deg") for x in ast.walk(left)) \
                and not any(isinstance(x, ast.Call) and getattr(x.func, "id", "") == "change" for x in ast.walk(left)):
            why += " (for how far it turned in total, use change(yaw_deg, s) or change(yaw_deg))"
        return [why]
    return bad(tree.body)


# ------------------------------------------------------------------ over time
def _is_angle(node) -> bool:
    """An angle that goes round (yaw_deg ...): its change is counted through whole turns, not wrapped at 180."""
    return isinstance(node, ast.Name) and node.id.endswith("_deg")


def _is_condition(node) -> bool:
    return isinstance(node, (ast.Compare, ast.BoolOp)) or (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not))


def window_calls(trees) -> list[tuple[str, ast.Call]]:
    """The change(...) / mean(...) calls in these parsed formulas, once each, as (key, call)."""
    out: dict[str, ast.Call] = {}
    for t in trees:
        for n in ast.walk(t):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in WINDOW_FUNCS:
                out.setdefault(ast.dump(n), n)
    return list(out.items())


def formula_trees(task: dict) -> list:
    """Every formula a task's conditions, steps and reward terms use, parsed (the ones that don't parse left out)."""
    fx = task.get("fixtures", {})
    src = [c.get("expr") for c in task.get("conditions", []) if c.get("kind") == "expression"]
    src += [s.get(k) for s in task.get("steps", []) for k in ("done_when", "keep")]
    src += [t.get("formula") for t in task.get("reward_terms", [])]
    out = []
    for e in src:
        if e:
            try:
                out.append(parse(e, fx))
            except ExprError:
                pass
    return out


class Windows:
    """Follows the change(...) / mean(...) calls of some formulas through a run, the same way for the judge (NumPy) and
    the learner (JAX: the state is plain arrays it carries from tick to tick). Each tick adds the quantity's change since
    the last tick (change; an angle's taken the short way round, so whole turns add up) or its value (mean) to a ring of
    the last ticks; a window sums the ticks it covers."""

    def __init__(self, trees):
        import numpy as np
        self.calls = window_calls(trees)
        secs = [float(c.args[1].value) if len(c.args) > 1 else 0.0 for _, c in self.calls]
        self.ticks = np.array([max(1, round(s / WINDOW_DT)) if s else 0 for s in secs], dtype=float)
        self.n = int(max([1.0, *self.ticks]))
        self.whole = self.ticks == 0  # since the try began
        self.is_change = np.array([c.func.id == "change" for _, c in self.calls])
        self.wrap = np.array([_is_angle(c.args[0]) for _, c in self.calls])
        self._st = None  # NumPy: the state, when feed() keeps it

    def __bool__(self) -> bool:
        return bool(self.calls)

    def _now(self, xp, v):
        out = []
        for _, c in self.calls:
            t = ast.Expression(body=c.args[0])
            x = xp.where(violation(xp, t, v) <= 1e-9, 1.0, 0.0) if _is_condition(c.args[0]) else violation(xp, t, v, numeric=True)
            out.append(xp.asarray(x, dtype=float))
        return xp.stack(out)

    def start(self, xp, v) -> dict:
        k = len(self.calls)
        st = {"prev": self._now(xp, v), "buf": xp.zeros((k, self.n)), "acc": xp.zeros(k), "count": xp.zeros(())}
        return self.step(xp, st, v)

    def step(self, xp, st: dict, v) -> dict:
        q = self._now(xp, v)
        d = q - st["prev"]
        d = xp.where(xp.asarray(self.wrap), (d + 180.0) % 360.0 - 180.0, d)
        inc = xp.where(xp.asarray(self.is_change), d, q)
        inc = xp.where(xp.isnan(inc), 0.0, inc)  # a moment it can't be measured adds nothing
        slot = st["count"].astype(int) % self.n
        if xp.__name__ == "numpy":
            buf = st["buf"].copy()
            buf[:, slot] = inc
        else:
            buf = st["buf"].at[:, slot].set(inc)
        return {"prev": xp.where(xp.isnan(q), st["prev"], q), "buf": buf, "acc": st["acc"] + inc,
                "count": st["count"] + 1}

    def values(self, xp, st: dict) -> dict:
        """{key: the call's value now}, for v["window"]."""
        age = (st["count"] - 1 - xp.arange(self.n)) % self.n  # how many ticks ago each slot was written
        ticks = xp.asarray(self.ticks)
        mask = (age[None, :] < ticks[:, None]) & (age[None, :] < st["count"])
        total = xp.where(xp.asarray(self.whole), st["acc"], xp.sum(xp.where(mask, st["buf"], 0.0), axis=1))
        count = xp.where(xp.asarray(self.whole), st["count"], xp.minimum(ticks, st["count"]))
        out = xp.where(xp.asarray(self.is_change), total, total / xp.maximum(count, 1.0))
        return {k: out[i] for i, (k, _) in enumerate(self.calls)}

    def feed(self, v: dict) -> dict:
        """NumPy, once a tick: follow this moment and put the values in v["window"] (with any already there)."""
        import numpy as np
        if not self.calls:
            return v
        self._st = self.start(np, v) if self._st is None else self.step(np, self._st, v)
        v["window"] = {**(v.get("window") or {}), **{k: float(x) for k, x in self.values(np, self._st).items()}}
        return v

    def covers(self, v: dict) -> bool:
        """v already has every value this follows (whoever ran the run fed it)."""
        w = v.get("window") or {}
        return all(k in w for k, _ in self.calls)


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
        if k in ("joints", "touch"):
            out[k] = {n: float(j) for n, j in x.items()}
        elif isinstance(x, dict) or getattr(x, "ndim", 0):
            continue  # not a single number
        else:
            out[k] = float(x)
    out["dist"] = {n: float(x) for n, x in v["dist"].items()}
    if v.get("window"):  # change(...) / mean(...) at that moment
        out["window"] = {k: float(x) for k, x in v["window"].items()}
    return out


FLICKER_TICKS = 5  # contacts come and go for a tick or two even in a steady grip: a break shorter than 0.1 s isn't one
CONTACT_NAMES = {"held", "fingers_touching"}


def contact_based(c: dict) -> bool:
    """A condition judged from the contacts (held, touching(...), fingers_touching): yes/no readings that flicker."""
    e = c.get("expr") or ""
    if c.get("kind") != "expression" or not e:
        return False
    try:
        return bool(uses(e) & CONTACT_NAMES) or "touching(" in e
    except SyntaxError:
        return False


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
        self.streak: dict[int, int] = {}  # ticks in a row a contact-based condition has been broken
        trees = []
        for c in conditions:
            try:
                trees += [parse(c["expr"], fixtures)] if c.get("kind") == "expression" and c.get("expr") else []
            except ExprError:
                pass
        self.over_time = Windows(trees)  # change(...) / mean(...), when the run doesn't already follow them
        self.last_window: dict = {}  # their values at the last moment (for conditions judged at the end)

    def update(self, v: dict) -> None:
        if self.over_time and not self.over_time.covers(v):
            self.over_time.feed(v)
        self.last_window = dict(v.get("window") or {})
        held = float(v["held"]) > 0.5
        let_go = self.was_held and not held
        self.was_held = held
        for i, c in self.items:
            x = float(condition_violation(self.np, {**c, "when": "end"}, v, self.fx, self.half))
            w = c["when"]
            if w in ("always", "last") and contact_based(c):  # a flicker of the contacts isn't a break
                self.streak[i] = self.streak.get(i, 0) + 1 if x > 1e-6 else 0
                x = x if self.streak[i] >= FLICKER_TICKS else 0.0
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
    # in the hand: off the floor and off anything else it could rest on, so only the robot can be keeping it up (it may
    # roll and slide between the fingers; held asks for more, a pinch from opposite sides)
    support = xp.zeros(())
    for f in fixtures.values():
        under = (f["lo"][0] <= x) & (x <= f["hi"][0]) & (f["lo"][1] <= y) & (y <= f["hi"][1])
        top = f.get("top_z", f["hi"][2] if len(f["hi"]) > 2 else 0.0)
        support = xp.maximum(support, xp.where(under, f.get("rest_z", 0.0) if f.get("kind") == "container" else top, 0.0))
    v["in_hand"] = xp.asarray(v["bottom"] > support + IN_HAND_LIFT, dtype=float)
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
    "held": ("the robot holding it", "flag"), "in_hand": ("it in the hand", "flag"), "grip": ("the grip score (0 to 1)", "frac"),
    "closure": ("the gripper closed", "pct"), "touch_dist": ("the nearest fingertip's gap to it", "m"),
    "lifted": ("it lifted", "m"), "hand_dist": ("the hand's distance to it", "m"),
    "grip_gap": ("the hand's distance from Morph's grip", "m"),
    "hand_x": ("the hand's x", "m"), "hand_y": ("the hand's y", "m"), "hand_z": ("the hand's height", "m"),
    "robot_x": ("the robot's x", "m"), "robot_y": ("the robot's y", "m"), "base_z": ("the robot body's height", "m"),
    "base_tilt_deg": ("the robot body's tilt", "deg"), "spin_dps": ("how fast it spins", "deg/s"),
    "spin_z_dps": ("how fast it turns about the vertical", "deg/s"), "fingers_touching": ("the number of fingers touching it", ""),
    "goal_x": ("the goal's x", "m"), "goal_y": ("the goal's y", "m"), "goal_z": ("the goal's height", "m"),
}
def _negate(words: str) -> str:
    return words.replace("the robot holding", "the robot not holding").replace(" touching ", " not touching ")


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
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "touching" and n.args \
                and isinstance(n.args[0], ast.Constant):
            return f"the {str(n.args[0].value).replace('|', ' or ')} touching {obj}", "flag"
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in WINDOW_FUNCS and n.args:
            span = "since the start" if len(n.args) < 2 else f"over the last {ast.unparse(n.args[1])} s"
            q = n.args[0]
            if n.func.id == "mean":
                if _is_condition(q) or (word(q) or ("", ""))[1] == "flag":
                    return f"the share of the time {span} with {text(q)}", "pct"
                w = word(q)
                return (f"the average of {w[0]} {span}", w[1]) if w else None
            turn = {"yaw_deg": "turned about the vertical", "roll_deg": "rolled", "pitch_deg": "pitched"}
            if isinstance(q, ast.Name) and q.id in turn:
                return f"how far {obj} {turn[q.id]} {span.replace('over the', 'in the')}", "deg"
            w = word(q)
            return (f"the change in {w[0]} {span}", w[1]) if w else None
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "abs" and len(n.args) == 1 \
                and isinstance(n.args[0], ast.Call) and getattr(n.args[0].func, "id", "") == "change":
            w = word(n.args[0])
            return (w[0] + " (either way)", w[1]) if w else None
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
                return _negate(w[0])
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
                    return w[0] if on else _negate(w[0])
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
