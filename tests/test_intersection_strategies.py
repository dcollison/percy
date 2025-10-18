"""
Intersection Strategy Test Suite.

This script provides a series of automated tests for the FoRIntersectionStrategy
implementations. Each test case represents a specific geometric scenario
designed to validate the intersection logic against all available strategies,
including edge cases.

Core Features:
- Use of pytest.mark.parametrize to test multiple strategies cleanly.
- Use of `assert` to programmatically verify expected outcomes.
- A global `ENABLE_PLOTTING` flag to easily turn 3D visualisations for
  each test case on or off.
"""

from typing import Type

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from percy import (
    FoRIntersectionStrategy,
    Platform,
    PyramidalSATStrategy,
    SensorConfiguration,
    SphericalAccurateStrategy,
)
from tests.plotting_tools import visualise_scene

# ======================================================================
# Global Test Configuration
# ======================================================================

ENABLE_PLOTTING: bool = False

# ======================================================================
# Helper Functions
# ======================================================================


def create_box_vertices(center: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Creates the 8 vertices for an axis-aligned bounding box."""
    half_size = np.array(size) / 2.0
    return np.array(
        [
            center + [-half_size[0], -half_size[1], -half_size[2]],
            center + [half_size[0], -half_size[1], -half_size[2]],
            center + [half_size[0], half_size[1], -half_size[2]],
            center + [-half_size[0], half_size[1], -half_size[2]],
            center + [-half_size[0], -half_size[1], half_size[2]],
            center + [half_size[0], -half_size[1], half_size[2]],
            center + [half_size[0], half_size[1], half_size[2]],
            center + [-half_size[0], half_size[1], half_size[2]],
        ],
        dtype=np.float64,
    )


def create_tilted_prism_vertices(
    center: np.ndarray, size: np.ndarray, rpy_deg: np.ndarray
) -> np.ndarray:
    """Creates vertices for a prism (box) and applies a rotation."""
    box = create_box_vertices(center=np.zeros(3), size=size)
    rotation = Rotation.from_euler(seq="zyx", angles=rpy_deg, degrees=True)
    return rotation.apply(box) + center


def run_test_case(
    name: str,
    volume_vertices: np.ndarray,
    platform: Platform,
    expected: bool,
) -> None:
    """
    Executes a single frustum culling test case.

    :param name: The descriptive name of the test case.
    :param volume_vertices: The vertices of the volume to test against.
    :param platform: The Platform object containing the sensor to test.
    :param expected: The expected boolean result for the intersection check.
    """
    strategy_name = platform.components[0].strategy.__class__.__name__
    test_name = f"{name} ({strategy_name})"
    print(f"--- Running Test: {test_name} ---")

    can_see = platform.can_observe_volume(volume_vertices)

    print(f"  Result={can_see}, Expected={expected}")

    if ENABLE_PLOTTING and can_see != expected:
        visualise_scene(platforms=[platform], volumes=[volume_vertices])

    assert can_see == expected, f"Failed test '{test_name}'"
    print("  ... PASSED ...")


# ======================================================================
# Test Case Definitions
# ======================================================================


ALL_STRATEGIES = [PyramidalSATStrategy, SphericalAccurateStrategy]


@pytest.mark.parametrize("strategy_class", ALL_STRATEGIES)
def test_simple_hit(strategy_class: type[FoRIntersectionStrategy]):
    """Volume is directly in front of and inside the sensor's FoR."""
    sensor_config = SensorConfiguration(
        "cam",
        np.zeros(3),
        np.zeros(3),
        1,
        50,
        np.pi / 4,
        np.pi / 4,
        strategy_class(),
    )
    platform = Platform("p1", np.zeros(3), np.zeros(3), components=[sensor_config])
    volume = create_box_vertices(center=np.array([5, 0, 0]), size=np.array([1, 1, 1]))
    run_test_case("Simple Hit", volume, platform, expected=True)


@pytest.mark.parametrize("strategy_class", ALL_STRATEGIES)
def test_simple_miss(strategy_class: type[FoRIntersectionStrategy]):
    """Volume is far to the side, completely outside the FoR."""
    sensor_config = SensorConfiguration(
        "cam",
        np.zeros(3),
        np.zeros(3),
        1,
        50,
        np.pi / 8,
        np.pi / 8,
        strategy_class(),
    )
    platform = Platform("p1", np.zeros(3), np.zeros(3), components=[sensor_config])
    volume = create_box_vertices(center=np.array([5, 5, 5]), size=np.array([1, 1, 1]))
    run_test_case("Simple Miss", volume, platform, expected=False)


@pytest.mark.parametrize("strategy_class", ALL_STRATEGIES)
def test_volume_behind_sensor(strategy_class: type[FoRIntersectionStrategy]):
    """Volume is entirely behind the sensor's origin."""
    sensor_config = SensorConfiguration(
        "cam",
        np.zeros(3),
        np.zeros(3),
        1,
        50,
        np.pi / 4,
        np.pi / 4,
        strategy_class(),
    )
    platform = Platform("p1", np.zeros(3), np.zeros(3), components=[sensor_config])
    volume = create_box_vertices(center=np.array([-5, 0, 0]), size=np.array([1, 1, 1]))
    run_test_case("Volume Behind Sensor", volume, platform, expected=False)


@pytest.mark.parametrize("strategy_class", ALL_STRATEGIES)
def test_volume_too_close(strategy_class: type[FoRIntersectionStrategy]):
    """Volume is in the view cone but closer than the minimum range."""
    sensor_config = SensorConfiguration(
        "cam",
        np.zeros(3),
        np.zeros(3),
        5,
        50,
        np.pi / 4,
        np.pi / 4,
        strategy_class(),
    )
    platform = Platform("p1", np.zeros(3), np.zeros(3), components=[sensor_config])
    volume = create_box_vertices(center=np.array([2, 0, 0]), size=np.array([1, 1, 1]))
    run_test_case("Volume Too Close", volume, platform, expected=False)


@pytest.mark.parametrize("strategy_class", ALL_STRATEGIES)
def test_volume_too_far(strategy_class: type[FoRIntersectionStrategy]):
    """Volume is in the view cone but beyond the maximum range."""
    sensor_config = SensorConfiguration(
        "cam",
        np.zeros(3),
        np.zeros(3),
        1,
        50,
        np.pi / 4,
        np.pi / 4,
        strategy_class(),
    )
    platform = Platform("p1", np.zeros(3), np.zeros(3), components=[sensor_config])
    volume = create_box_vertices(center=np.array([100, 0, 0]), size=np.array([1, 1, 1]))
    run_test_case("Volume Too Far", volume, platform, expected=False)


@pytest.mark.parametrize("strategy_class", ALL_STRATEGIES)
def test_rotated_sensor_hit(strategy_class: type[FoRIntersectionStrategy]):
    """Sensor is rotated to face a volume that would otherwise be missed."""
    sensor_config = SensorConfiguration(
        "cam",
        np.zeros(3),
        np.array([0, 0, np.pi / 4]),  # 45-degree yaw
        1,
        50,
        np.pi / 12,
        np.pi / 12,
        strategy_class(),
    )
    platform = Platform("p1", np.zeros(3), np.zeros(3), components=[sensor_config])
    target_pos = 10 * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0])
    volume = create_box_vertices(center=target_pos, size=np.array([1, 1, 1]))
    run_test_case("Rotated Sensor Hit", volume, platform, expected=True)


@pytest.mark.parametrize("strategy_class", ALL_STRATEGIES)
def test_piercing_volume_hit(strategy_class: type[FoRIntersectionStrategy]):
    """A long, thin volume pierces the FoR without any vertices inside."""
    sensor_config = SensorConfiguration(
        "cam",
        np.zeros(3),
        np.zeros(3),
        5,
        20,
        np.pi / 4,
        np.pi / 4,
        strategy_class(),
    )
    platform = Platform("p1", np.zeros(3), np.zeros(3), components=[sensor_config])
    volume = create_box_vertices(
        center=np.array([10, 0, 0]),
        size=np.array([0.1, 20, 0.1]),
    )
    run_test_case("Piercing Volume Hit", volume, platform, expected=True)


@pytest.mark.parametrize("strategy_class", ALL_STRATEGIES)
def test_sensor_inside_volume(strategy_class: type[FoRIntersectionStrategy]):
    """The sensor's origin is located inside the target volume."""
    sensor_config = SensorConfiguration(
        "cam",
        np.zeros(3),
        np.zeros(3),
        1,
        50,
        np.pi / 4,
        np.pi / 4,
        strategy_class(),
    )
    platform = Platform(
        "p1", np.array([10, 0, 0]), np.zeros(3), components=[sensor_config]
    )
    volume = create_box_vertices(center=np.array([10, 0, 0]), size=np.array([5, 5, 5]))
    run_test_case("Sensor Inside Volume", volume, platform, expected=True)


@pytest.mark.parametrize("strategy_class", ALL_STRATEGIES)
def test_grazing_miss(strategy_class: type[FoRIntersectionStrategy]):
    """Volume is very close to the edge of the FoR but does not intersect."""
    az_half = np.pi / 8
    sensor_config = SensorConfiguration(
        "cam", np.zeros(3), np.zeros(3), 1, 50, az_half, np.pi / 8, strategy_class()
    )
    platform = Platform("p1", np.zeros(3), np.zeros(3), components=[sensor_config])
    # Position the box just outside the frustum boundary with a small buffer.
    y_pos = 10 * np.tan(az_half) + 1.0 + 0.5  # half-size of box is 1.0
    volume = create_box_vertices(
        center=np.array([10, y_pos, 0]), size=np.array([2, 2, 2])
    )
    run_test_case("Grazing Miss", volume, platform, expected=False)


@pytest.mark.parametrize("strategy_class", ALL_STRATEGIES)
def test_complex_rotated_miss(strategy_class: type[FoRIntersectionStrategy]):
    """A thin, rotated prism that should be missed by a complexly rotated sensor."""
    sensor_config = SensorConfiguration(
        "cam",
        np.zeros(3),
        np.array([np.pi / 4, 0, np.pi / 4]),  # Roll and Yaw
        1,
        50,
        np.pi / 12,
        np.pi / 12,
        strategy_class(),
    )
    platform = Platform("p1", np.zeros(3), np.zeros(3), components=[sensor_config])
    volume = create_tilted_prism_vertices(
        center=np.array([10, 0, 15]),  # Shifted further to be a clear miss
        size=np.array([20, 0.5, 0.5]),
        rpy_deg=np.array([0, 0, 90]),
    )
    run_test_case("Complex Rotated Miss", volume, platform, expected=False)


def test_pyramidal_sees_spherical_misses():
    """
    Tests the specific geometric case where a pyramidal FoR sees an object
    in its corner that a spherical FoR with the same angles misses.
    """
    az_half = np.pi / 6
    el_half = np.pi / 6
    distance = 10.0

    # The point at the corner of the pyramid's base
    y_pos = distance * np.tan(az_half)
    z_pos = distance * np.tan(el_half)
    volume = create_box_vertices(
        center=np.array([distance, y_pos, z_pos]), size=np.array([0.5, 0.5, 0.5])
    )

    # --- Test Pyramidal Strategy (should be a HIT) ---
    pyramidal_config = SensorConfiguration(
        "cam", np.zeros(3), np.zeros(3), 1, 50, az_half, el_half, PyramidalSATStrategy()
    )
    platform_pyr = Platform(
        "p1", np.zeros(3), np.zeros(3), components=[pyramidal_config]
    )
    run_test_case("Pyramid Corner Hit", volume, platform_pyr, expected=True)

    # --- Test Spherical Strategy (should be a MISS) ---
    spherical_config = SensorConfiguration(
        "cam",
        np.zeros(3),
        np.zeros(3),
        1,
        50,
        az_half,
        el_half,
        SphericalAccurateStrategy(),
    )
    platform_sph = Platform(
        "p1", np.zeros(3), np.zeros(3), components=[spherical_config]
    )
    run_test_case("Spherical Corner Miss", volume, platform_sph, expected=False)
