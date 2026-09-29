"""TaskSpec: robot-independent description of the desired outcome.

No joints, grasps or trajectories in here. Each field records where its value came
from (measured, inferred, user, default) so the report can show it.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .geometry import wrap_angle, yaw_error
from .sim import BlockPose

SCHEMA_VERSION = "morph.taskspec/1"
Provenance = Literal["measured", "inferred_rule", "inferred_llm", "user", "default"]


class Field_(BaseModel):
    """A value plus where it came from."""

    value: float | str | bool | list | dict
    provenance: Provenance
    note: str = ""


class ObjectIdentity(BaseModel):
    name: str = "arrow_block"
    size_m: list[float] = Field(default_factory=lambda: [0.08, 0.08, 0.06])
    marker: str = "arrow on top face points along the block's local +x axis"
    mass_kg: float = 0.2
    friction: float = 0.6


class Goal(BaseModel):
    x: Field_
    y: Field_
    yaw: Field_  # radians, heading of the arrow (world +x = east, +y = north)
    pos_tol_m: Field_
    yaw_tol_rad: Field_
    keep_upright: Field_
    max_tilt_rad: Field_ = Field_(value=math.radians(10), provenance="default")
    orientation_mode: Field_ = Field_(value="planar_yaw", provenance="default",
                                      note="planar_yaw | face_up_flip (flip = a different top face)")
    target_top_face: Field_ = Field_(value="+z", provenance="default")


class TaskSpec(BaseModel):
    schema_version: str = SCHEMA_VERSION
    status: Literal["draft", "confirmed"] = "draft"
    object: ObjectIdentity = Field(default_factory=ObjectIdentity)
    goal: Goal
    frame: str = "world: z up, +x east, +y north, floor plane z=0"
    permitted_contacts: Field_ = Field_(value=["any_robot_link"], provenance="default")
    time_limit_s: Field_ = Field_(value=20.0, provenance="default")
    instruction: str = ""
    assumptions: list[str] = Field(default_factory=list)
    source_run: str | None = None

    @field_validator("schema_version")
    @classmethod
    def _v(cls, v: str) -> str:
        if v != SCHEMA_VERSION:
            raise ValueError(f"unsupported TaskSpec version {v}")
        return v

    # convenience accessors
    @property
    def gx(self) -> float:
        return float(self.goal.x.value)

    @property
    def gy(self) -> float:
        return float(self.goal.y.value)

    @property
    def gyaw(self) -> float:
        return float(self.goal.yaw.value)

    def confirm(self) -> "TaskSpec":
        return self.model_copy(update={"status": "confirmed"})


class Outcome(BaseModel):
    success: bool
    pos_err_m: float
    yaw_err_rad: float
    tilt_rad: float
    on_floor: bool
    at_rest: bool
    released: bool
    reasons: list[str]
    final_pose: dict

    @property
    def yaw_err_deg(self) -> float:
        return math.degrees(self.yaw_err_rad)


def evaluate(spec: TaskSpec, pose: BlockPose, speed: float, released: bool, half_z: float = 0.03) -> Outcome:
    """Identical measured-final-pose test for every robot."""
    pos_err = math.hypot(pose.x - spec.gx, pose.y - spec.gy)
    yerr = yaw_error(pose.yaw, spec.gyaw)
    on_floor = abs(pose.z - half_z) < 0.01
    at_rest = speed < 0.02
    reasons = []
    if pos_err > float(spec.goal.pos_tol_m.value):
        reasons.append(f"position error {pos_err * 100:.1f} cm > {float(spec.goal.pos_tol_m.value) * 100:.1f} cm")
    if yerr > float(spec.goal.yaw_tol_rad.value):
        reasons.append(f"yaw error {math.degrees(yerr):.1f} deg > {math.degrees(float(spec.goal.yaw_tol_rad.value)):.1f} deg")
    if spec.goal.keep_upright.value and pose.tilt > float(spec.goal.max_tilt_rad.value):
        reasons.append(f"block tilted {math.degrees(pose.tilt):.1f} deg")
    if not on_floor:
        reasons.append(f"block not resting on floor (z={pose.z:.3f})")
    if not at_rest:
        reasons.append(f"block still moving ({speed:.3f})")
    if not released:
        reasons.append("robot still holding the block")
    return Outcome(success=not reasons, pos_err_m=pos_err, yaw_err_rad=yerr, tilt_rad=pose.tilt,
                   on_floor=on_floor, at_rest=at_rest, released=released, reasons=reasons,
                   final_pose=pose.as_dict())


def default_spec(x: float, y: float, yaw: float, provenance: Provenance = "default") -> TaskSpec:
    return TaskSpec(goal=Goal(
        x=Field_(value=x, provenance=provenance), y=Field_(value=y, provenance=provenance),
        yaw=Field_(value=wrap_angle(yaw), provenance=provenance),
        pos_tol_m=Field_(value=0.05, provenance="default", note="prompt default: 5 cm"),
        yaw_tol_rad=Field_(value=math.radians(15), provenance="default", note="prompt default: 15 deg"),
        keep_upright=Field_(value=True, provenance="default")))
