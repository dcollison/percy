from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import numba
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation


# NOTE: This library assumes all input volumes are CONVEX.
# For concave shapes, the user must first decompose them into convex parts and
# check each part individually.


# ======================================================================
# Numba-Optimized Functions
# ======================================================================


@numba.njit
def _project_jit(vertices: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    """JIT-compiled function to project vertices onto an axis."""
    min_val = np.inf
    max_val = -np.inf
    for i in range(vertices.shape[0]):
        dot = vertices[i] @ axis
        if dot < min_val:
            min_val = dot
        if dot > max_val:
            max_val = dot
    return min_val, max_val


@numba.njit
def _is_overlapping_jit(axes: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> bool:
    """JIT-compiled function to check for overlap on all axes."""
    for i in range(axes.shape[0]):
        p1_min, p1_max = _project_jit(v1, axes[i])
        p2_min, p2_max = _project_jit(v2, axes[i])
        if p1_max < p2_min or p2_max < p1_min:
            return False
    return True


# ======================================================================
# JIT Warm-up
# ======================================================================


def _warmup_jit_functions():
    """
    Calls the JIT-compiled functions with realistic, non-trivial inputs to
    ensure they are compiled before the first real use. This avoids a
    performance penalty on the first call in a time-sensitive context.
    """
    try:
        box = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
        box2 = box + 2.0
        axes = np.eye(3, dtype=np.float64)
        _is_overlapping_jit(axes, box, box2)

    except Exception:
        pass


_warmup_jit_functions()


# ======================================================================
# Hierarchical Composite Pattern
# ======================================================================


@dataclass(frozen=True)
class WorldSpaceSensor:
    name: str
    position: np.ndarray
    rpy: np.ndarray
    r_min: float
    r_max: float
    az_half_angle: float
    el_half_angle: float
    strategy: "FoRIntersectionStrategy"

    def can_observe_volume(self, volume_vertices: np.ndarray) -> bool:
        return self.strategy.check_intersection(self, volume_vertices)


class BaseComponent(ABC):
    @abstractmethod
    def get_world_sensors(
        self, parent_rotation: Rotation, parent_position: np.ndarray
    ) -> list[WorldSpaceSensor]: ...


@dataclass
class SensorConfiguration(BaseComponent):
    name: str
    relative_position: np.ndarray
    relative_rpy: np.ndarray
    r_min: float
    r_max: float
    az_half_angle: float
    el_half_angle: float
    strategy: "FoRIntersectionStrategy"

    def get_world_sensors(
        self, parent_rotation: Rotation, parent_position: np.ndarray
    ) -> list[WorldSpaceSensor]:
        my_rotation = Rotation.from_euler("zyx", self.relative_rpy[[2, 1, 0]])
        world_rotation = parent_rotation * my_rotation
        world_position = parent_position + parent_rotation.apply(self.relative_position)
        return [
            WorldSpaceSensor(
                name=self.name,
                position=world_position,
                rpy=world_rotation.as_euler("zyx")[[2, 1, 0]],
                r_min=self.r_min,
                r_max=self.r_max,
                az_half_angle=self.az_half_angle,
                el_half_angle=self.el_half_angle,
                strategy=self.strategy,
            )
        ]


@dataclass
class SensorSystem(BaseComponent):
    components: Sequence[BaseComponent]

    def get_world_sensors(
        self, parent_rotation: Rotation, parent_position: np.ndarray
    ) -> list[WorldSpaceSensor]:
        all_sensors = []
        for component in self.components:
            all_sensors.extend(
                component.get_world_sensors(parent_rotation, parent_position)
            )
        return all_sensors


@dataclass
class Platform:
    name: str
    position: np.ndarray
    rpy: np.ndarray
    components: Sequence[BaseComponent]

    def set_state(self, position: np.ndarray, rpy: np.ndarray) -> None:
        self.position = position
        self.rpy = rpy

    def update_state(
        self, dt: float, velocity: np.ndarray, angular_velocity: np.ndarray
    ) -> None:
        self.position += velocity * dt
        self.rpy = (self.rpy + angular_velocity * dt + np.pi) % (2 * np.pi) - np.pi

    def get_all_world_sensors(self) -> list[WorldSpaceSensor]:
        platform_rotation = Rotation.from_euler("zyx", self.rpy[[2, 1, 0]])
        all_sensors = []
        for component in self.components:
            all_sensors.extend(
                component.get_world_sensors(platform_rotation, self.position)
            )
        return all_sensors

    def can_observe_volume(
        self, volume_vertices: np.ndarray, sensor_names: Optional[list[str]] = None
    ) -> bool:
        sensors_to_check = self.get_all_world_sensors()
        if sensor_names:
            sensors_to_check = [s for s in sensors_to_check if s.name in sensor_names]

        for sensor in sensors_to_check:
            if sensor.can_observe_volume(volume_vertices):
                return True
        return False


# ======================================================================
# FoR Intersection Strategy Pattern
# ======================================================================


class FoRIntersectionStrategy(ABC):
    @abstractmethod
    def check_intersection(
        self, sensor: WorldSpaceSensor, volume_vertices: np.ndarray
    ) -> bool: ...


class PyramidalSATStrategy(FoRIntersectionStrategy):
    """Intersection strategy for a pyramidal FoR using the exact convex hull (SAT)."""

    def check_intersection(
        self, sensor: WorldSpaceSensor, volume_vertices: np.ndarray
    ) -> bool:
        frustum_vertices = self._get_world_vertices(sensor)
        if np.any(np.max(frustum_vertices, 0) < np.min(volume_vertices, 0)) or np.any(
            np.max(volume_vertices, 0) < np.min(frustum_vertices, 0)
        ):
            return False
        return self._run_sat_check(frustum_vertices, volume_vertices)

    def _get_world_vertices(self, sensor: WorldSpaceSensor) -> np.ndarray:
        tan_az, tan_el = np.tan(sensor.az_half_angle), np.tan(sensor.el_half_angle)
        # In a Z-forward frame (X-right, Y-up), az is in XZ plane, el is in YZ plane.
        nx = sensor.r_min * tan_az
        ny = sensor.r_min * tan_el
        fx = sensor.r_max * tan_az
        fy = sensor.r_max * tan_el
        local_verts = np.array(
            [
                [-nx, -ny, sensor.r_min],
                [nx, -ny, sensor.r_min],
                [nx, ny, sensor.r_min],
                [-nx, ny, sensor.r_min],
                [-fx, -fy, sensor.r_max],
                [fx, -fy, sensor.r_max],
                [fx, fy, sensor.r_max],
                [-fx, -fy, sensor.r_max],
            ]
        )
        pre_rotation = Rotation.from_euler("y", np.pi / 2, degrees=False)
        x_forward_verts = pre_rotation.apply(local_verts)
        world_rotation = Rotation.from_euler("zyx", sensor.rpy[[2, 1, 0]])
        return world_rotation.apply(x_forward_verts) + sensor.position

    def _run_sat_check(self, v1: np.ndarray, v2: np.ndarray) -> bool:
        v1 = np.ascontiguousarray(v1, dtype=np.float64)
        v2 = np.ascontiguousarray(v2, dtype=np.float64)
        try:
            h1, h2 = Delaunay(v1, qhull_options="QJ"), Delaunay(v2, qhull_options="QJ")
        except Exception:
            return True

        axes1 = self._get_face_normals(h1)
        if not _is_overlapping_jit(axes1, v1, v2):
            return False

        axes2 = self._get_face_normals(h2)
        if not _is_overlapping_jit(axes2, v1, v2):
            return False

        edges1 = self._get_edges(h1)
        edges2 = self._get_edges(h2)
        for e1 in edges1:
            for e2 in edges2:
                axis = np.cross(e1, e2)
                norm = np.linalg.norm(axis)
                if norm > 1e-6:
                    axis = np.ascontiguousarray(axis / norm)
                    if not _is_overlapping_jit(axis.reshape(1, 3), v1, v2):
                        return False
        return True

    def _get_face_normals(self, hull: Delaunay) -> np.ndarray:
        normals = []
        for simplex in hull.convex_hull:
            v0, v1, v2 = hull.points[simplex]
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm > 1e-6:
                normals.append(normal / norm)
        return np.ascontiguousarray(normals, dtype=np.float64)

    def _get_edges(self, hull: Delaunay) -> np.ndarray:
        edges = set()
        for simplex in hull.convex_hull:
            for i in range(3):
                p1_idx, p2_idx = tuple(sorted((simplex[i], simplex[(i + 1) % 3])))
                edges.add((p1_idx, p2_idx))
        return np.ascontiguousarray(
            [hull.points[p2] - hull.points[p1] for p1, p2 in edges], dtype=np.float64
        )


class SphericalAccurateStrategy(FoRIntersectionStrategy):
    """
    A robust strategy for a directional spherical sector FoR that handles
    vertex containment, origin containment, and piercing scenarios.
    """

    def check_intersection(
        self, sensor: WorldSpaceSensor, volume_vertices: np.ndarray
    ) -> bool:
        world_rotation = Rotation.from_euler("zyx", sensor.rpy[[2, 1, 0]])
        local_vertices = world_rotation.apply(
            volume_vertices - sensor.position, inverse=True
        )

        min_vals = np.min(local_vertices, axis=0)
        max_vals = np.max(local_vertices, axis=0)
        if np.all(min_vals < 0) and np.all(max_vals > 0):
            return True

        x, y, z = local_vertices.T
        tan_az, tan_el = np.tan(sensor.az_half_angle), np.tan(sensor.el_half_angle)

        inside_cone = (y**2 * tan_el**2 + z**2 * tan_az**2) <= (
            x**2 * tan_az**2 * tan_el**2 + 1e-9
        )

        ranges_sq = x**2 + y**2 + z**2
        is_inside = (
            inside_cone
            & (ranges_sq >= sensor.r_min**2)
            & (ranges_sq <= sensor.r_max**2)
            & (x > 0)
        )

        if np.any(is_inside):
            return True

        min_y, min_z = min_vals[1], min_vals[2]
        max_y, max_z = max_vals[1], max_vals[2]
        min_x, max_x = min_vals[0], max_vals[0]

        x_axis_pierces_yz_plane = (
            min_y <= 0 and max_y >= 0 and min_z <= 0 and max_z >= 0
        )
        x_ranges_overlap = max_x >= sensor.r_min and min_x <= sensor.r_max

        if x_axis_pierces_yz_plane and x_ranges_overlap:
            return True

        return False


# ======================================================================
# Example Usage
# ======================================================================

if __name__ == "__main__":

    def create_box(center, size=1.0) -> np.ndarray:
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

    camera_config = SensorConfiguration(
        name="camera",
        relative_position=np.zeros(3),
        relative_rpy=np.zeros(3),
        r_min=1.0,
        r_max=40.0,
        az_half_angle=np.pi / 12,
        el_half_angle=np.pi / 12,
        strategy=PyramidalSATStrategy(),
    )
    radar_config = SensorConfiguration(
        name="radar",
        relative_position=np.zeros(3),
        relative_rpy=np.array([0, 0, np.pi / 4]),
        r_min=1.0,
        r_max=80.0,
        az_half_angle=np.pi / 6,
        el_half_angle=np.pi / 6,
        strategy=SphericalAccurateStrategy(),
    )

    aircraft = Platform(
        name="aircraft",
        position=np.array([0, 0, 1000]),
        rpy=np.array([np.pi / 6, 0, 0]),
        components=[camera_config, radar_config],
    )

    target = create_box(center=[50, 0, 1000])

    print("\n--- Running Simplified Example ---")
    print(f"Can aircraft see target? -> {aircraft.can_observe_volume(target)}")
