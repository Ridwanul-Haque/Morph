"""Capability description shared by all robot adapters.

Describes what the implemented controllers can do in this simulator, not what the
hardware could do in principle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass
class Capabilities:
    robot: str
    embodiment: Literal["fixed_arm", "legged", "mobile_pusher"]
    model_source: str
    base_mobility: Literal["fixed", "legged_planar", "holonomic_planar"]
    end_effectors: list[str]
    contact_modes: list[Literal["grasp", "push"]]
    prehensile: bool
    # Where the object centre may be for the robot to act on it.
    object_workspace: dict
    max_payload_kg: float
    max_contact_force_n: float
    can_set_planar_yaw: bool
    can_flip_object: bool
    action_primitives: list[str]
    modeled_limits: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)  # evidence: which primitives were tested and how
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)
