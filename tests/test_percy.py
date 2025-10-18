"""
Unit Test Suite for the Percy Hierarchical Sensor Framework.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from percy import Platform, SensorSystem, SensorConfiguration, PyramidalSATStrategy
from plotting_tools import visualise_scene

ENABLE_PLOTTING: bool = False


def create_box(center: list[float], size: float = 1.0) -> np.ndarray:
    h = size / 2.0
    c = np.array(center)
    return np.array(
        [
            c + [-h, -h, -h],
            c + [h, -h, -h],
            c + [h, h, -h],
            c + [-h, h, -h],
            c + [-h, -h, h],
            c + [h, -h, h],
            c + [h, h, h],
            c + [-h, h, h],
        ]
    )


def test_platform_no_rotation():
    sensor_config = SensorConfiguration(
        "cam",
        np.array([1, 2, 3]),
        np.zeros(3),
        1,
        10,
        np.pi / 4,
        np.pi / 4,
        PyramidalSATStrategy(),
    )
    platform = Platform(
        "p1", np.array([10, 20, 30]), np.zeros(3), components=[sensor_config]
    )

    sensor = platform.get_all_world_sensors()[0]

    np.testing.assert_allclose(sensor.position, np.array([11, 22, 33]))
    np.testing.assert_allclose(sensor.rpy, np.zeros(3))


def test_platform_with_rotation():
    sensor_config = SensorConfiguration(
        "cam",
        np.array([1, 0, 0]),
        np.zeros(3),
        1,
        10,
        np.pi / 4,
        np.pi / 4,
        PyramidalSATStrategy(),
    )
    platform = Platform(
        "p1",
        np.array([10, 0, 0]),
        np.array([0, 0, np.pi / 2]),
        components=[sensor_config],
    )

    sensor = platform.get_all_world_sensors()[0]

    np.testing.assert_allclose(sensor.position, np.array([10, 1, 0]), atol=1e-6)
    np.testing.assert_allclose(sensor.rpy, np.array([0, 0, np.pi / 2]), atol=1e-6)


def test_sensor_with_relative_rotation():
    sensor_config = SensorConfiguration(
        "cam",
        np.zeros(3),
        np.array([0, np.pi / 4, 0]),
        1,
        10,
        np.pi / 4,
        np.pi / 4,
        PyramidalSATStrategy(),
    )
    platform = Platform(
        "p1", np.zeros(3), np.array([0, 0, np.pi / 2]), components=[sensor_config]
    )

    sensor = platform.get_all_world_sensors()[0]

    expected_rot = Rotation.from_euler("zyx", [np.pi / 2, np.pi / 4, 0])
    actual_rot = Rotation.from_euler("zyx", sensor.rpy[[2, 1, 0]])

    np.testing.assert_allclose(
        actual_rot.as_matrix(), expected_rot.as_matrix(), atol=1e-6
    )


def test_nested_sensor_system():
    camera_config = SensorConfiguration(
        "cam",
        np.array([0, 0, -1]),
        np.array([0, np.pi / 6, 0]),
        1,
        10,
        np.pi / 4,
        np.pi / 4,
        PyramidalSATStrategy(),
    )
    pod = SensorSystem(components=[camera_config])
    platform = Platform("p1", np.array([0, 0, 100]), np.zeros(3), components=[pod])

    sensor = platform.get_all_world_sensors()[0]

    np.testing.assert_allclose(sensor.position, np.array([0, 0, 99]), atol=1e-6)
    np.testing.assert_allclose(sensor.rpy, np.array([0, np.pi / 6, 0]), atol=1e-6)


def test_visibility_with_hierarchy():
    sensor_config = SensorConfiguration(
        "cam",
        np.zeros(3),
        np.array([0, 0, -np.pi / 2]),
        1,
        50,
        np.pi / 8,
        np.pi / 8,
        PyramidalSATStrategy(),
    )
    platform = Platform(
        "p1", np.array([100, 0, 0]), np.array([0, 0, np.pi]), components=[sensor_config]
    )

    target_visible = create_box([100, 20, 0])
    target_invisible = create_box([100, -20, 0])

    if ENABLE_PLOTTING:
        visualise_scene(
            platforms=[platform], volumes=[target_visible, target_invisible]
        )

    assert platform.can_observe_volume(target_visible) is True
    assert platform.can_observe_volume(target_invisible) is False
