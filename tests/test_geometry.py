import math

import numpy as np
import pytest

from morph.geometry import quat_from_yaw, quat_to_mat, tilt_from_mat, wrap_angle, yaw_error, yaw_from_mat


@pytest.mark.parametrize("a,expected", [(0, 0), (math.pi, -math.pi), (-math.pi, -math.pi), (3 * math.pi, -math.pi),
                                        (2 * math.pi + 0.1, 0.1), (-2 * math.pi - 0.1, -0.1)])
def test_wrap_angle(a, expected):
    assert wrap_angle(a) == pytest.approx(expected, abs=1e-12)


def test_yaw_error_uses_shortest_arc():
    assert yaw_error(math.radians(179), math.radians(-179)) == pytest.approx(math.radians(2))
    assert yaw_error(0.0, 2 * math.pi) == pytest.approx(0.0, abs=1e-12)
    # an arrow pointing the opposite way is 180 deg wrong, not 0 (no 90/180 symmetry for the arrow)
    assert yaw_error(0.0, math.pi) == pytest.approx(math.pi)


def test_yaw_roundtrip_and_tilt():
    for yaw in np.linspace(-3.1, 3.1, 25):
        R = quat_to_mat(quat_from_yaw(yaw))
        assert yaw_from_mat(R) == pytest.approx(yaw, abs=1e-9)
        assert tilt_from_mat(R) == pytest.approx(0.0, abs=1e-6)
    # 90 deg about x: block on its side
    R = quat_to_mat(np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]))
    assert tilt_from_mat(R) == pytest.approx(math.pi / 2)
