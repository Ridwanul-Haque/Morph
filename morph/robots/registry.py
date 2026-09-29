"""Name -> robot adapter. Arms share one interface (see arm_skills.read_state)."""

from __future__ import annotations

ARM_NAMES = ("arm_A", "arm_B", "franka_panda", "ufactory_xarm7")


def make_arm(name: str):
    if name in ("arm_A", "arm_B"):
        from ..scene import ARM_A, ARM_B
        from .arm import ArmAdapter
        return ArmAdapter(ARM_A if name == "arm_A" else ARM_B)
    from .menagerie_arm import MENAGERIE_ARMS, MenagerieArmAdapter
    if name in MENAGERIE_ARMS:
        return MenagerieArmAdapter(MENAGERIE_ARMS[name])
    raise KeyError(f"unknown arm {name!r}; choose from {ARM_NAMES}")
