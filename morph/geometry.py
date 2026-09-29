"""Pose helpers. Angles are radians unless the name ends in _deg."""

from __future__ import annotations

import math

import numpy as np


def wrap_angle(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_error(a: float, b: float) -> float:
    """Absolute wrapped difference between two headings."""
    return abs(wrap_angle(a - b))


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """MuJoCo quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def yaw_from_mat(R: np.ndarray) -> float:
    """Heading of the body's local +x axis projected on the floor plane."""
    return math.atan2(R[1, 0], R[0, 0])


def tilt_from_mat(R: np.ndarray) -> float:
    """Angle between the body's local +z axis and world +z (0 = upright)."""
    return math.acos(max(-1.0, min(1.0, R[2, 2])))


def quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def rot2(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])
