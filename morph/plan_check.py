"""Checks on a step plan before any practice. A language model's plan goes through outside checkers and what they find
goes back to it (LLM-Modulo, Kambhampati et al. 2024): Nemotron's plans read well and still fail in ways Morph can see
without training. Two of them:

- the last step doesn't get the object where the goal wants it. The Stretch's pen plan ended at "close the gripper and
  lift the pen": nothing took it to the bin, and the learner had the whole carry and drop to find alone (0/10)
- a step's keep rule is already broken where the step starts, so the step is penalised from its first moment. The
  Stretch's first step had to keep its hand within 2 cm of the parcel while driving up to it from 50 cm away. Chained
  steps fail at their handoffs (T-STAR, Lee et al. 2021): each must be able to start where the one before leaves off
- steps out of order. A revision of the Stretch's pen plan began "align above the pen: abs(x - goal_x) <= 0.02 ..." with
  the gripper open: x is the pen's, so the pen had to be at the bin before it was picked up. Before the grasp only the
  robot should move the object towards the goal if it's to be carried there; and where Morph has its own route to the
  goal, the steps should come true along it in their order
- a grasp set up by being near the object. The Stretch's step before closing was "hand_dist <= 0.06": done with its
  hand resting on top of the parcel, so practice learned to rest it there and never had its fingers either side of it.
  With Morph's own grip known (task["grasp"]), the step before the grasp has to have the hand there (grip_gap)
- a measured value asked to be exactly a number. The Stretch's first step had "base_tilt_deg == 0": its body sits tilted
  about 0.2 degrees, so the step could never be done and every plan showed 0% of its steps
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np

from .goal_terms import FLAGS, normalize, parse, violation

# conditions judged once things settle (the ones the last step should lead to); "ever" and "always" ones aren't
END = ("end", "last", "release", None)
WHERE = {"x", "y", "z", "bottom", "goal_x", "goal_y", "goal_z"}
KIND_NAMES = {"inside": WHERE, "on_top_of": WHERE, "at_position": WHERE, "near": WHERE, "beside": WHERE,
              "touching": WHERE, "away_from": WHERE, "moved": WHERE | {"travelled"},
              "height": {"bottom", "z", "top", "lifted"}, "heading": {"yaw_deg"},
              "orientation": {"tilt_deg", "roll_deg", "pitch_deg", "yaw_deg"},
              "upright": {"tilt_deg", "roll_deg", "pitch_deg"}}
NEAR = 0.3  # in goal_terms units (about 9 cm or 27 degrees): a step the route came this close to, it went through
START_TICKS = 10  # the first 0.2 s with the robot holding still: a contact flickering at the very start isn't "broken"


def _names(expr: str) -> set[str]:
    tree = ast.parse(normalize(expr), mode="eval")
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - calls | ({"touching"} & calls)
    return names | ({"x", "y"} if "dist_to" in calls else set())  # dist_to('wall'): where the object is


def _uses_held(expr: str) -> bool:
    try:
        return "held" in _names(expr)
    except SyntaxError:
        return False


def before_the_grasp(steps: list[dict], task: dict) -> list[str]:
    """For an object that has to be carried to its place: steps before the first one that has it held mustn't ask for
    the object itself to be near the goal (its x, y against goal_x, goal_y or a fixture's), only the hand."""
    if not (task.get("target") or {}).get("lift"):
        return []
    goal_names = {"goal_x", "goal_y"} | {f"{f}_{k}" for f in task.get("fixtures", {}) for k in ("x", "y")}
    out = []
    for k, st in enumerate(steps):
        if _uses_held(st["done_when"]) or _uses_held(st.get("keep") or "0"):
            break
        try:
            names = _names(st["done_when"])
        except SyntaxError:
            continue
        if {"x", "y"} & names and goal_names & names:
            out.append(f"step {k + 1}, \"{st['name']}\" ({st['done_when']}), asks for the object itself to be at the goal "
                       "before it has been picked up (x and y are the object's; the hand's are hand_x and hand_y). It has "
                       "to be carried there: before the grasp, only the hand should be measured against the goal")
    return out


GRIP_GAP = 0.03  # m: the hand's point within this of where Morph's own grip puts it


def _needs_held(expr: str) -> bool:
    """Does the formula ask for the object to be held ("held == 1", "held"), not for it not to be ("held == 0")?"""
    try:
        tree = ast.parse(normalize(expr), mode="eval")
    except SyntaxError:
        return False
    for n in ast.walk(tree):
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.left, ast.Name) and n.left.id == "held" \
                and isinstance(n.comparators[0], ast.Constant):
            c, op = float(n.comparators[0].value), n.ops[0]
            if isinstance(op, ast.Eq) and c > 0.5 or isinstance(op, ast.GtE) and c > 0 or isinstance(op, ast.Gt) and c >= 0:
                return True
        elif isinstance(n, ast.BoolOp) and any(isinstance(x, ast.Name) and x.id == "held" for x in n.values):
            return True
    return isinstance(tree.body, ast.Name) and tree.body.id == "held"


def _grasp_step(steps: list[dict]) -> int | None:
    """The first step that has the object held when it's done."""
    return next((k for k, st in enumerate(steps) if _needs_held(st["done_when"])), None)


def grip_first(steps: list[dict], task: dict) -> list[str]:
    """With Morph's grip known, the step that ends just before the grasp must have the hand at it: near the object
    isn't enough (a hand resting on top of it is near)."""
    if not task.get("grasp"):
        return []
    g = _grasp_step(steps)
    if not g:
        return []
    st = steps[g - 1]
    try:
        if "grip_gap" in _names(st["done_when"]) or "grip_gap" in _names(steps[g]["done_when"]):
            return []
    except SyntaxError:
        return []
    return [f"step {g}, \"{st['name']}\" ({st['done_when']}), sets up the grasp in step {g + 1} without the hand being "
            f"where it can grip: near the object isn't enough (a hand resting on top of it is near). Add "
            f"grip_gap < {GRIP_GAP} (the hand's point within {GRIP_GAP * 100:.0f} cm of where Morph's own grip puts it, the "
            "fingers either side of the object)"]


def _closes_on(task: dict) -> bool:
    """Does the robot have parts that close on an object from either side (a gripper, a hand's fingers)?"""
    if task.get("gripper"):
        return True
    try:
        import mujoco
        from .fingers import hands
        return bool(hands(mujoco.MjModel.from_xml_path(task["robot_xml"])))
    except Exception:  # noqa: BLE001  (no model to look at)
        return False


def between_first(steps: list[dict], task: dict) -> list[str]:
    """The step that ends just before the grasp must have the object between the fingers: near it isn't enough (the
    Stretch's fingers came within 2.5 cm of a pen, round after round, and closed beside it). Not when the plan names
    Morph's own grip (grip_gap), which already puts the fingers either side of it."""
    g = _grasp_step(steps)
    if not g or task.get("start_state") or not _closes_on(task):
        return []
    st = steps[g - 1]
    try:
        names = _names(st["done_when"])
    except SyntaxError:
        return []
    if "between" in names or task.get("grasp") and "grip_gap" in names:
        return []
    return [f"step {g}, \"{st['name']}\" ({st['done_when']}), sets up the grasp in step {g + 1} without the object "
            "being between the fingers: a gripper can only grab what is between them, and near it isn't enough. Add "
            "between == 1"]


def exact_values(steps: list[dict]) -> list[str]:
    """No step may ask for a measured value (a position, an angle, a distance) to be exactly a number: physics never
    lands on it. Only the yes/no ones (held, between, ...) can be compared with ==."""
    out = []
    for k, st in enumerate(steps):
        for rule in ("done_when", "keep"):
            try:
                tree = ast.parse(normalize(st[rule]), mode="eval") if st.get(rule) else None
            except SyntaxError:
                continue
            for node in ast.walk(tree) if tree else ():
                if not isinstance(node, ast.Compare):
                    continue
                sides = [node.left, *node.comparators]
                for op, a, b in zip(node.ops, sides, sides[1:]):
                    name = next((x.id for x in (a, b) if isinstance(x, ast.Name)), None)
                    if isinstance(op, (ast.Eq, ast.NotEq)) and name and name not in FLAGS \
                            and any(isinstance(x, ast.Constant) for x in (a, b)):
                        out.append(f"step {k + 1}, \"{st['name']}\": its {rule.replace('_', ' ')} asks for {name} to be "
                                   f"exactly a number ({ast.unparse(node)}): a measured value never lands on it exactly "
                                   "(a robot's body sits tilted a fraction of a degree), so the step can never be done. "
                                   f"Use a range instead, e.g. {name} < a small limit")
    return out


def require_grip(task: dict) -> str | None:
    """What grip_first and between_first ask for, done by Morph if the plan still leaves it out: the step before the
    grasp also needs the hand at Morph's grip, or the object between the fingers. Returns what it changed."""
    steps = task.get("steps") or []
    g = _grasp_step(steps)
    if grip_first(steps, task):
        st = steps[g - 1]
        st["done_when"] = f"({st['done_when']}) and grip_gap < {GRIP_GAP}"
        return (f"step {g}, \"{st['name']}\", now also needs the hand where Morph grips the "
                f"{task.get('object', 'object')} (grip_gap < {GRIP_GAP}): near it isn't enough to grip it")
    if between_first(steps, task):
        st = steps[g - 1]
        st["done_when"] = f"({st['done_when']}) and between == 1"
        return (f"step {g}, \"{st['name']}\", now also needs the {task.get('object', 'object')} between the fingers "
                "(between == 1): a gripper can only grab what is between them")
    return None


HELD_ON = re.compile(r"\bheld\s*(?:==\s*1(?:\.0)?|>=?\s*0?\.5|>\s*0(?:\.0)?)(?![\d.])")
HELD_BARE = re.compile(r"(?<!not )\bheld\b(?!\s*(?:==|!=|<|>))")
GAP_CLAUSE = re.compile(r"\s*\band\s+grip_gap\s*<=?\s*[\d.]+|\bgrip_gap\s*<=?\s*[\d.]+\s*(?:\band\b\s*)?")


def in_hand_steps(task: dict) -> str | None:
    """Started in the hand, and the goal never asks for a pinch: a step's "still has it" is in_hand, not held. The
    LinkerHand's pen plans began "secure grip: held == 1" with the pen lying across the fingers, never pinched, so no
    try got past step 1. grip_gap goes too: with no grip of Morph's planned (in-hand skills have none) it's only the
    hand's distance to the object's centre. Returns what it changed."""
    if not task.get("start_state") or task.get("grasp"):
        return None
    if any(_uses_held(c.get("expr") or "") or (c.get("kind") == "height" and c.get("hold"))
           for c in task.get("conditions", [])):
        return None
    changed = []
    for st in task.get("steps") or []:
        for k in ("done_when", "keep"):
            e = st.get(k) or ""
            if not re.search(r"\b(held|grip_gap)\b", e):
                continue
            new = HELD_BARE.sub("in_hand", HELD_ON.sub("in_hand == 1", normalize(e)))
            new = re.sub(r"\s+", " ", GAP_CLAUSE.sub("", new)).strip()
            new = re.sub(r"^\(\s*\)$", "", new)
            if new != e.strip():
                st[k] = new or ("1" if k == "done_when" else "")
                changed.append(st["name"])
    if not changed:
        return None
    return (f"in the hand, not gripped: {', '.join(dict.fromkeys(changed))} now ask{'s' if len(set(changed)) == 1 else ''} "
            "for in_hand (kept up by the fingers) instead of held (a pinch, which a roll or spin loosens)")


def _describe(c: dict) -> str:
    from .goals import Condition
    try:
        return Condition(**c).describe()
    except Exception:  # noqa: BLE001
        return c.get("text") or c.get("expr") or c.get("kind", "")


def unfinished(steps: list[dict], conditions: list[dict]) -> list[str]:
    """Goal conditions the plan's last step never looks at: after it, the learner has that part of the task with no
    step to guide it."""
    if not steps:
        return []
    last = steps[-1]
    try:
        got = _names(last["done_when"])
    except SyntaxError:
        return []
    missing = []
    for c in conditions:
        if c.get("when") not in END or c.get("negate"):
            continue
        if c["kind"] == "expression":
            try:
                need = _names(c["expr"])
            except (SyntaxError, KeyError):
                continue
        else:
            need = set(KIND_NAMES.get(c["kind"], ()))
            if c.get("target"):
                need |= {f"{c['target']}_x", f"{c['target']}_y", f"{c['target']}_top"}
        if need and not need & got:
            missing.append(_describe(c))
    if not missing:
        return []
    return [f"the plan stops short of the goal: its last step, \"{last['name']}\" ({last['done_when']}), never checks "
            + "; ".join(missing) + ". After it the robot would have that part of the task with no step to guide it: "
            "add the steps that get there (the goal itself is checked once your last step is done)"]


class KeepCheck:
    """Can each step's keep rule hold where the step starts? Step 1 starts where every try does (measured with the
    robot holding still); a later step wherever a saved state has the step before it done (Morph's own route, the
    handoffs and closest moments of earlier replays: starts.npz)."""

    def __init__(self, task: dict):
        from .task_model import build_task_model, indices
        self.task = task
        self.fx = task.get("fixtures", {})
        self.m = build_task_model(task, accurate=True)
        self.ix = indices(self.m, task)
        self.start = self._start()
        self.saved = self._saved()
        # Morph's own route (scaffold), in order: the saved states up to where their times stop increasing
        self.route = []
        if (task.get("starts") or {}).get("scaffold") and self.saved:
            import numpy as _np
            t = _np.load(Path(task.get("_dir", ".")) / task["starts"]["file"])["t"]
            n = next((i for i in range(1, len(t)) if t[i] < t[i - 1]), len(t))
            self.route = self.saved[:n]

    def _measure(self, d) -> dict:
        from .task_model import contact_held, contact_touch, measure
        return measure(np, d.xpos, d.xmat, d.qvel, self.ix, self.task, held=contact_held(self.m, d, self.ix),
                       ctrl=d.ctrl, qpos=d.qpos, touch=contact_touch(self.m, d, self.ix))

    def _start(self) -> list[dict]:
        from .learn_task import CTRL_DT, rollout
        from .sim import PhysicalParams, SimulationEnvironment
        env = SimulationEnvironment(self.m, control_dt=CTRL_DT)
        seen: list[dict] = []

        class _Keep:
            def update(_, v):
                seen.append(v)

        rollout(env, self.m, self.ix, self.task, lambda o: np.zeros(self.m.nu), self.task["start"], PhysicalParams(),
                0, stop=lambda: len(seen) >= START_TICKS, watch=(_Keep(),), settle_s=0.0)
        return seen

    def _saved(self) -> list[dict]:
        import mujoco
        st = self.task.get("starts") or {}
        f = Path(self.task.get("_dir", ".")) / st.get("file", "starts.npz")
        if not st or not f.exists():
            return []
        z = np.load(f)
        d = mujoco.MjData(self.m)
        out = []
        for i, q in enumerate(z["qpos"]):
            if len(q) != self.m.nq:
                return []
            d.qpos[:] = q
            d.qvel[:] = 0.0
            if "ctrl" in z.files and not np.isnan(z["ctrl"][i]).any():
                d.ctrl[:] = z["ctrl"][i]
            mujoco.mj_forward(self.m, d)
            out.append(self._measure(d))
        return out

    def order(self, steps: list[dict]) -> list[str]:
        """Along Morph's own route, in their order: a step done there (that wasn't already true at the start) while an
        earlier one never happened is out of place, or measures the wrong thing."""
        if len(self.route) < 3 or not steps:
            return []
        try:
            done = [parse(st["done_when"], self.fx) for st in steps]
        except Exception:  # noqa: BLE001
            return []
        miss = lambda k, v: float(violation(np, done[k], v))  # noqa: E731
        at_start = [miss(k, self.route[0]) <= 1e-3 for k in range(len(steps))]
        k, closest = 0, 9.0
        for v in self.route:
            while k < len(steps) and miss(k, v) <= 1e-3:
                k, closest = k + 1, 9.0
            if k >= len(steps):
                return []
            closest = min(closest, miss(k, v))
            for j in range(k + 1, len(steps)):
                if not at_start[j] and miss(j, v) <= 1e-3:
                    if closest <= NEAR:  # the route came close (its states are a few ticks apart): passed, not skipped
                        k, closest = j + 1, 9.0
                        break
                    return [f"Morph's own route to the goal does step {j + 1}, \"{steps[j]['name']}\" ({steps[j]['done_when']}), "
                            f"without ever doing step {k + 1}, \"{steps[k]['name']}\" ({steps[k]['done_when']}), first: step "
                            f"{k + 1} may measure the wrong thing (the object where the hand was meant?) or be in the wrong place"]
        return []

    def __call__(self, steps: list[dict]) -> list[str]:
        out = self.order(steps)
        for k, st in enumerate(steps):
            if not st.get("keep"):
                continue
            try:
                keep = parse(st["keep"], self.fx)
                before = parse(steps[k - 1]["done_when"], self.fx) if k else None
                done = parse(st["done_when"], self.fx)
            except Exception:  # noqa: BLE001  (check_steps already says what it can't read)
                continue
            if k == 0:
                at = self.start
                broken = [v for v in at if float(violation(np, keep, v)) > 1e-3]
                bad = at and len(broken) == len(at)  # broken the whole first 0.2 s, not one flicker
            else:
                at = [v for v in self.saved if float(violation(np, before, v)) <= 1e-3
                      and float(violation(np, done, v)) > 1e-3]
                broken = [v for v in at if float(violation(np, keep, v)) > 1e-3]
                bad = at and 2 * len(broken) > len(at)
            if not bad:
                continue
            v = broken[0]
            vals = ", ".join(f"{n} = {float(v[n]):.3g}" for n in sorted(_names(st["keep"])) if n in v
                             and np.ndim(v[n]) == 0)
            where = "where every try starts" if k == 0 else f"in {len(broken)} of {len(at)} saved states where step {k} is done"
            out.append(f"step {k + 1}, \"{st['name']}\": its keep rule ({st['keep']}) is already broken {where}"
                       + (f" ({vals})" if vals else "") + ", so the step is penalised from its first moment. A keep rule "
                       "must hold for the whole step, starting from where the step before leaves off: loosen it, or "
                       "move that requirement into done_when")
        return out


def plan_checker(task: dict, conditions: list[dict], log=print):
    """steps (dicts) -> problems found, for the plan's makers (plan_steps, revise_task_steps, the variant search)."""
    try:
        keep = KeepCheck(task)
    except Exception as e:  # noqa: BLE001  (a model Morph can't build here: the keep rules go unchecked)
        log(f"  (couldn't check the plan's keep rules: {e})")
        keep = None

    def check(steps: list[dict]) -> list[str]:
        return (unfinished(steps, conditions) + before_the_grasp(steps, task) + grip_first(steps, task)
                + between_first(steps, task) + exact_values(steps)
                + (keep(steps) if keep else []))
    return check
