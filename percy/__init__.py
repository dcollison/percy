from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Optional

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


@numba.njit
def _gjk_support(vertices: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Finds the vertex on a shape that is furthest in a given direction."""
    max_dot = -np.inf
    best_vertex = vertices[0]
    for i in range(vertices.shape[0]):
        dot = vertices[i] @ direction
        if dot > max_dot:
            max_dot = dot
            best_vertex = vertices[i]
    return best_vertex


@numba.njit
def _gjk_handle_simplex(
    simplex: list[np.ndarray], direction: np.ndarray
) -> tuple[bool, np.ndarray]:
    """
    Processes the GJK simplex to check for origin containment and find the
    new best search direction. This is the core of the GJK algorithm.
    """
    if len(simplex) == 2:  # Line case
        a, b = simplex[1], simplex[0]
        ab, ao = b - a, -a
        if np.dot(ab, ao) > 0:
            return False, np.cross(np.cross(ab, ao), ab)
        else:
            simplex.pop(0)
            return False, ao

    elif len(simplex) == 3:  # Triangle case
        a, b, c = simplex[2], simplex[1], simplex[0]
        ab, ac, ao = b - a, c - a, -a
        abc_perp = np.cross(ab, ac)
        if np.dot(np.cross(abc_perp, ac), ao) > 0:
            simplex.pop(1)
            return False, np.cross(np.cross(ac, ao), ac)
        if np.dot(np.cross(ab, abc_perp), ao) > 0:
            simplex.pop(0)
            return False, np.cross(np.cross(ab, ao), ab)
        if np.dot(abc_perp, ao) > 0:
            return False, abc_perp
        else:
            return False, -abc_perp

    elif len(simplex) == 4:  # Tetrahedron case
        a, b, c, d = simplex[3], simplex[2], simplex[1], simplex[0]
        ab, ac, ad, ao = b - a, c - a, d - a, -a
        abc_perp = np.cross(ab, ac)
        if np.dot(abc_perp, ao) > 0:
            simplex.pop(0)
            return False, abc_perp
        acd_perp = np.cross(ac, ad)
        if np.dot(acd_perp, ao) > 0:
            simplex.pop(1)
            return False, acd_perp
        adb_perp = np.cross(ad, ab)
        if np.dot(adb_perp, ao) > 0:
            simplex.pop(2)
            return False, adb_perp
        return True, direction

    return False, direction


@numba.njit
def _run_gjk_check_jit(v1: np.ndarray, v2: np.ndarray) -> bool:
    """A self-contained, JIT-compiled implementation of the GJK algorithm."""
    direction = np.array([1.0, 0.0, 0.0])
    simplex = [_gjk_support(v1, direction) - _gjk_support(v2, -direction)]
    direction = -simplex[0]
    for _ in range(64):
        a = _gjk_support(v1, direction) - _gjk_support(v2, -direction)
        if np.dot(a, direction) < 0:
            return False
        simplex.append(a)
        collides, direction = _gjk_handle_simplex(simplex, direction)
        if collides:
            return True
    return False


# JIT Warm-up
try:
    _warmup_verts = np.zeros((4, 3), dtype=np.float64)
    _warmup_axes = np.zeros((1, 3), dtype=np.float64)
    _is_overlapping_jit(_warmup_axes, _warmup_verts, _warmup_verts)
    _run_gjk_check_jit(_warmup_verts, _warmup_verts)
except Exception:
    pass


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
    """Intersection strategy for a pyramidal FoR using SAT."""

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
        nx, ny = sensor.r_min * tan_az, sensor.r_min * tan_el
        fx, fy = sensor.r_max * tan_az, sensor.r_max * tan_el
        local_verts = np.array(
            [
                [-nx, -ny, sensor.r_min],
                [nx, -ny, sensor.r_min],
                [nx, ny, sensor.r_min],
                [-nx, ny, sensor.r_min],
                [-fx, -fy, sensor.r_max],
                [fx, -fy, sensor.r_max],
                [fx, fy, sensor.r_max],
                [-fx, fy, sensor.r_max],
            ]
        )
        rot = Rotation.from_euler("zyx", sensor.rpy[[2, 1, 0]]).as_matrix()
        return local_verts @ rot + sensor.position

    def _run_sat_check(self, v1: np.ndarray, v2: np.ndarray) -> bool:
        v1, v2 = v1.astype(np.float64), v2.astype(np.float64)
        try:
            h1, h2 = Delaunay(v1, qhull_options="QJ"), Delaunay(v2, qhull_options="QJ")
        except:
            return True
        if not _is_overlapping_jit(self._get_face_normals(h1), v1, v2):
            return False
        if not _is_overlapping_jit(self._get_face_normals(h2), v1, v2):
            return False
        for e1 in self._get_edges(h1):
            for e2 in self._get_edges(h2):
                axis = np.cross(e1, e2)
                norm = np.linalg.norm(axis)
                if norm > 1e-6 and not _is_overlapping_jit(
                    np.array([axis / norm]), v1, v2
                ):
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
        return np.array(normals)

    def _get_edges(self, hull: Delaunay) -> np.ndarray:
        edges = set()
        for simplex in hull.convex_hull:
            for i in range(3):
                p1_idx, p2_idx = tuple(sorted((simplex[i], simplex[(i + 1) % 3])))
                edges.add((p1_idx, p2_idx))
        return np.array([hull.points[p2] - hull.points[p1] for p1, p2 in edges])


class PyramidalGJKStrategy(FoRIntersectionStrategy):
    """Intersection strategy for a pyramidal FoR using GJK."""

    def check_intersection(
        self, sensor: WorldSpaceSensor, volume_vertices: np.ndarray
    ) -> bool:
        frustum_vertices = self._get_world_vertices(sensor)
        if np.any(np.max(frustum_vertices, 0) < np.min(volume_vertices, 0)) or np.any(
            np.max(volume_vertices, 0) < np.min(frustum_vertices, 0)
        ):
            return False
        return _run_gjk_check_jit(
            frustum_vertices.astype(np.float64), volume_vertices.astype(np.float64)
        )

    def _get_world_vertices(self, sensor: WorldSpaceSensor) -> np.ndarray:
        tan_az, tan_el = np.tan(sensor.az_half_angle), np.tan(sensor.el_half_angle)
        nx, ny = sensor.r_min * tan_az, sensor.r_min * tan_el
        fx, fy = sensor.r_max * tan_az, sensor.r_max * tan_el
        local_verts = np.array(
            [
                [-nx, -ny, sensor.r_min],
                [nx, -ny, sensor.r_min],
                [nx, ny, sensor.r_min],
                [-nx, ny, sensor.r_min],
                [-fx, -fy, sensor.r_max],
                [fx, -fy, sensor.r_max],
                [fx, fy, sensor.r_max],
                [-fx, fy, sensor.r_max],
            ]
        )
        rot = Rotation.from_euler("zyx", sensor.rpy[[2, 1, 0]]).as_matrix()
        return local_verts @ rot + sensor.position


class SphericalAccurateStrategy(FoRIntersectionStrategy):
    """A robust strategy for a directional spherical sector FoR."""

    def check_intersection(
        self, sensor: WorldSpaceSensor, volume_vertices: np.ndarray
    ) -> bool:
        rot = Rotation.from_euler("zyx", sensor.rpy[[2, 1, 0]]).as_matrix()
        local_vertices = (volume_vertices - sensor.position) @ rot.T
        for p in local_vertices:
            dist = np.linalg.norm(p)
            if (
                sensor.r_min <= dist <= sensor.r_max
                and p[2] > 0
                and abs(np.arctan2(p[0], p[2])) <= sensor.az_half_angle
                and abs(np.arctan2(p[1], p[2])) <= sensor.el_half_angle
            ):
                return True
        min_p, max_p = np.min(local_vertices, 0), np.max(local_vertices, 0)
        if (
            min_p[0] <= 0 <= max_p[0]
            and min_p[1] <= 0 <= max_p[1]
            and max_p[2] >= sensor.r_min
            and min_p[2] <= sensor.r_max
        ):
            return True
        w_min, w_max = np.min(volume_vertices, 0), np.max(volume_vertices, 0)
        if np.all(w_min < sensor.position) and np.all(sensor.position < w_max):
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
        strategy=PyramidalGJKStrategy(),
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
