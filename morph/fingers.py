"""Hands and fingers: which bodies form each finger, which finger is the thumb, and which fingers touch the object.

  hands(m)            every multi-finger hand in a model: {"palm", "side", "fingers": {label: [body ids, base to tip]}}
  finger_of_body(m)   body id -> finger key ("thumb", "index", ..., or "left_thumb" with two hands)
  touching(m, d)      from the contacts: {key: 1.0 / 0.0} for every finger and every robot body, plus "fingers_touching"

A hand is a body with 3 or more fingers hanging off it, a finger being a short unbranched chain with at least two
joints. A body that floats freely (a robot dog's trunk, whose legs look the same) is never a palm. Fingers are labelled from their names (Shadow: rh_thdistal, rh_ffdistal; Allegro: th_*, ff_*; G1: *_thumb_*);
when the names don't say, the thumb is the finger whose base sits apart from the others and the rest are numbered
outwards from it (index, middle, ring, little).
"""

from __future__ import annotations

import re

import mujoco
import numpy as np

LABELS = ("thumb", "index", "middle", "ring", "little")
_WORDS = {  # per label: whole words, or two-letter prefixes as Shadow/Allegro use them (thdistal, ff_base)
    "thumb": (r"thumb", r"th"), "index": (r"index|first|pointer", r"ff|if"), "middle": (r"middle", r"mf"),
    "ring": (r"ring", r"rf"), "little": (r"little|pinky|small", r"lf"),
}
MAX_FINGER_M = 0.3
_CACHE: dict = {}


def _label_from_name(names: list[str]) -> str | None:
    votes: dict[str, int] = {}
    for n in names:
        found = set()
        for tok in re.split(r"[^a-z0-9]+", n.lower()):
            # prefix first: Shadow's thumb has a "thmiddle" link
            hit = [lab for lab, (_, prefix) in _WORDS.items() if re.match(prefix, tok)] or \
                  [lab for lab, (word, _) in _WORDS.items() if re.search(word, tok)]
            found.update(hit[:1])
        for lab in found:
            votes[lab] = votes.get(lab, 0) + 1
    # the word on most of its parts: a LinkerHand's index finger has an "index_middle" segment
    top = sorted(votes.items(), key=lambda kv: -kv[1])
    return top[0][0] if top and (len(top) == 1 or top[0][1] > top[1][1]) else None


def _side(m: mujoco.MjModel, body: int) -> str:
    b = body
    while b > 0:
        for tok in re.split(r"[^a-z]+", m.body(b).name.lower()):
            if tok in ("left", "l", "lh"):
                return "left"
            if tok in ("right", "r", "rh"):
                return "right"
        b = int(m.body_parentid[b])
    return ""


def _chain(m: mujoco.MjModel, children: dict, start: int) -> list[int] | None:
    """The bodies from start down to its tip, if it's an unbranched finger: else None."""
    out, b = [start], start
    while children.get(b):
        if len(children[b]) > 1:
            return None
        b = children[b][0]
        out.append(b)
    hinges = sum(int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)
                 for x in out for j in range(m.body_jntadr[x], m.body_jntadr[x] + m.body_jntnum[x]))
    return out if hinges >= 2 else None


def hands(m: mujoco.MjModel) -> list[dict]:
    key = id(m)
    if key in _CACHE and _CACHE[key][0] is m:
        return _CACHE[key][1]
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    children: dict[int, list[int]] = {}
    for b in range(1, m.nbody):
        children.setdefault(int(m.body_parentid[b]), []).append(b)
    found = []
    for palm, kids in children.items():
        if palm == 0 or len(kids) < 3 or any(int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE) for j in
                                             range(m.body_jntadr[palm], m.body_jntadr[palm] + m.body_jntnum[palm])):
            continue
        chains = [c for c in (_chain(m, children, k) for k in kids) if c]
        chains = [c for c in chains if np.linalg.norm(d.xpos[c[-1]] - d.xpos[palm]) < MAX_FINGER_M]
        if len(chains) < 3:
            continue
        labels = [_label_from_name([m.body(b).name for b in c]) for c in chains]
        if None in labels or len(set(labels)) < len(labels):
            labels = _label_by_shape(d, chains)
        found.append({"palm": palm, "side": _side(m, palm), "fingers": dict(zip(labels, chains))})
    if len(found) > 1:
        for k, h in enumerate(found):
            side = h["side"] or f"hand{k + 1}"
            if sum(x["side"] == h["side"] for x in found) > 1:
                side = f"hand{k + 1}"
            h["fingers"] = {f"{side}_{lab}": c for lab, c in h["fingers"].items()}
    if len(_CACHE) > 16:
        _CACHE.clear()
    _CACHE[key] = (m, found)
    return found


def _label_by_shape(d: mujoco.MjData, chains: list[list[int]]) -> list[str]:
    """No usable names: the thumb's base is the one farthest from the other bases; the rest by distance from it."""
    base = np.array([d.xpos[c[0]] for c in chains])
    apart = [np.mean([np.linalg.norm(base[i] - base[j]) for j in range(len(chains)) if j != i]) for i in range(len(chains))]
    thumb = int(np.argmax(apart))
    rest = sorted((i for i in range(len(chains)) if i != thumb), key=lambda i: np.linalg.norm(base[i] - base[thumb]))
    out = [""] * len(chains)
    out[thumb] = "thumb"
    for k, i in enumerate(rest):
        out[i] = LABELS[1 + k] if k + 1 < len(LABELS) else f"finger{k + 2}"
    return out


def finger_of_body(m: mujoco.MjModel) -> dict[int, str]:
    return {b: lab for h in hands(m) for lab, chain in h["fingers"].items() for b in chain}


def finger_names(m: mujoco.MjModel, strip: str = "") -> list[dict]:
    """hands() with body names instead of ids (for profiles and prompts)."""
    return [{"palm": m.body(h["palm"]).name.removeprefix(strip), "side": h["side"],
             "fingers": {lab: [m.body(b).name.removeprefix(strip) for b in c] for lab, c in h["fingers"].items()}}
            for h in hands(m)]


def part_names(m: mujoco.MjModel, strip: str = "robot_") -> dict[str, list[str]]:
    """What goal formulas can name on this robot: joints (joint_frac), end parts (part_x) and touching()."""
    robot = robot_bodies(m)
    has_child = {int(m.body_parentid[b]) for b in range(1, m.nbody)}
    joints = [m.joint(j).name.removeprefix(strip) for j in range(m.njnt)
              if int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))]
    bodies = [m.body(b).name.removeprefix(strip) for b in sorted(robot)]
    return {"joints": joints, "parts": [m.body(b).name.removeprefix(strip) for b in sorted(robot) if b not in has_child],
            "touch": bodies + sorted(set(finger_of_body(m).values()))}


def robot_bodies(m: mujoco.MjModel) -> set[int]:
    """The robot's bodies in one of Morph's models: named robot_*, or else everything that moves except the object."""
    named = {b for b in range(1, m.nbody) if m.body(b).name.startswith("robot_")}
    if named:
        return named
    block = m.body("block").id if "block" in [m.body(b).name for b in range(m.nbody)] else -1
    out = set()
    for b in range(1, m.nbody):
        a = b
        while a > 0:
            if m.body_jntnum[a] and a != block:
                out.add(b)
                break
            a = int(m.body_parentid[a])
    return out - {block}


def touching(m: mujoco.MjModel, d: mujoco.MjData, block_body: int | None = None) -> dict:
    """Which parts of the robot touch the object now, from the contacts: every robot body by name (without robot_)
    and every finger by its label, 1.0 or 0.0; "fingers_touching" counts the fingers (or, with no hand, the bodies)."""
    if block_body is None:
        block_body = m.body("block").id
    robot = robot_bodies(m)
    hit = set()
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist > 0.002:
            continue
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if b1 == block_body and b2 in robot:
            hit.add(b2)
        elif b2 == block_body and b1 in robot:
            hit.add(b1)
    out = {m.body(b).name.removeprefix("robot_"): float(b in hit) for b in robot}
    fingers = finger_of_body(m)
    for lab in set(fingers.values()):
        out[lab] = 0.0
    for b in hit:
        if b in fingers:
            out[fingers[b]] = 1.0
    out["fingers_touching"] = float(sum(out[lab] for lab in set(fingers.values())) if fingers else len(hit))
    return out
