"""
Plotting Tools for the Percy Sensor Simulation Framework.

This module provides functions to generate interactive 3D visualizations
of Platforms, their Sensor Fields of Regard, and target volumes using Plotly.

The main entry point is `visualise_scene`, which can render a complete scenario
with multiple platforms, automatically color-coding the results of visibility checks.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation

# Import the core simulation classes from the main library
from percy import (
    Platform,
    PyramidalSATStrategy,
    SphericalAccurateStrategy,
    WorldSpaceSensor,
)


# ======================================================================
# Core Geometric Plotting Utilities
# ======================================================================


def get_volume_plot(
    vertices: np.ndarray, name: str, color: str = "blue", opacity: float = 0.5
) -> go.Mesh3d | go.Scatter3d:
    """Creates a 3D mesh for an arbitrary CONVEX volume using its convex hull."""
    if vertices.shape[0] < 4:
        return go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            mode="markers",
            name=name,
        )
    try:
        hull = Delaunay(vertices, qhull_options="QJ")
        return go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=hull.convex_hull[:, 0],
            j=hull.convex_hull[:, 1],
            k=hull.convex_hull[:, 2],
            opacity=opacity,
            color=color,
            name=name,
            flatshading=True,
        )
    except Exception:
        return go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            mode="markers",
            name=name,
        )


def get_platform_axes_plot(
    platform: Platform, length: float = 5.0
) -> list[go.Scatter3d]:
    """Creates three lines (RGB) representing the platform's local XYZ axes."""
    rot = Rotation.from_euler("zyx", platform.rpy[[2, 1, 0]])
    origin = platform.position
    # Define the local axes and rotate them into the world frame
    x_axis, y_axis, z_axis = rot.apply(np.eye(3) * length)

    return [
        go.Scatter3d(
            x=[origin[0], origin[0] + x_axis[0]],
            y=[origin[1], origin[1] + x_axis[1]],
            z=[origin[2], origin[2] + x_axis[2]],
            mode="lines",
            line=dict(color="red", width=4),
            name=f"{platform.name} X",
        ),
        go.Scatter3d(
            x=[origin[0], origin[0] + y_axis[0]],
            y=[origin[1], origin[1] + y_axis[1]],
            z=[origin[2], origin[2] + y_axis[2]],
            mode="lines",
            line=dict(color="green", width=4),
            name=f"{platform.name} Y",
        ),
        go.Scatter3d(
            x=[origin[0], origin[0] + z_axis[0]],
            y=[origin[1], origin[1] + z_axis[1]],
            z=[origin[2], origin[2] + z_axis[2]],
            mode="lines",
            line=dict(color="blue", width=4),
            name=f"{platform.name} Z",
        ),
    ]


# ======================================================================
# Field of Regard Visualization Functions
# ======================================================================


def plot_pyramidal_for(
    sensor: WorldSpaceSensor, color: str = "red"
) -> go.Mesh3d | go.Scatter3d:
    """Generates a plottable Mesh3d object for a pyramidal FoR."""
    # This is a protected member, but for a plotting utility it's acceptable.
    if hasattr(sensor.strategy, "_get_world_vertices"):
        world_verts = sensor.strategy._get_world_vertices(sensor)
        return get_volume_plot(
            world_verts, name=f"{sensor.name} (Pyramidal)", color=color, opacity=0.1
        )
    return go.Scatter3d()


def plot_spherical_for(
    sensor: WorldSpaceSensor, color: str = "cyan"
) -> list[go.Surface]:
    """Generates plottable Surface objects for a spherical sector FoR."""
    resolution = 50j
    # We build the sphere pointing along +X to match the standard coordinate system
    az, el = np.mgrid[
        -sensor.az_half_angle : sensor.az_half_angle : resolution,
        -sensor.el_half_angle : sensor.el_half_angle : resolution,
    ]
    x_local = np.cos(el) * np.cos(az)
    y_local = np.cos(el) * np.sin(az)
    z_local = np.sin(el)

    world_rotation = Rotation.from_euler("zyx", sensor.rpy[[2, 1, 0]])

    def _create_surface_points(radius: float) -> np.ndarray:
        points = np.stack([x.ravel() * radius for x in [x_local, y_local, z_local]], -1)
        return world_rotation.apply(points) + sensor.position

    near_world = _create_surface_points(sensor.r_min).reshape(*x_local.shape, 3)
    far_world = _create_surface_points(sensor.r_max).reshape(*x_local.shape, 3)

    name = f"{sensor.name} (Spherical)"
    surfaces = [
        go.Surface(
            x=near_world[..., 0],
            y=near_world[..., 1],
            z=near_world[..., 2],
            colorscale=[[0, color], [1, color]],
            opacity=0.2,
            showscale=False,
            name=name,
        ),
        go.Surface(
            x=far_world[..., 0],
            y=far_world[..., 1],
            z=far_world[..., 2],
            colorscale=[[0, color], [1, color]],
            opacity=0.2,
            showscale=False,
            showlegend=False,
        ),
    ]

    # Add side surfaces to connect near and far planes
    for i in [0, -1]:
        # Top and Bottom sides (connecting along elevation edges)
        surfaces.append(
            go.Surface(
                x=np.vstack([near_world[:, i, 0], far_world[:, i, 0]]),
                y=np.vstack([near_world[:, i, 1], far_world[:, i, 1]]),
                z=np.vstack([near_world[:, i, 2], far_world[:, i, 2]]),
                colorscale=[[0, color], [1, color]],
                showscale=False,
                showlegend=False,
                opacity=0.2,
            )
        )
        # Left and Right sides (connecting along azimuth edges)
        surfaces.append(
            go.Surface(
                x=np.vstack([near_world[i, :, 0], far_world[i, :, 0]]),
                y=np.vstack([near_world[i, :, 1], far_world[i, :, 1]]),
                z=np.vstack([near_world[i, :, 2], far_world[i, :, 2]]),
                colorscale=[[0, color], [1, color]],
                showscale=False,
                showlegend=False,
                opacity=0.2,
            )
        )

    return surfaces


# ======================================================================
# Main Scene Visualization Function
# ======================================================================


def visualise_scene(platforms: Sequence[Platform], volumes: list[np.ndarray]):
    """
    Creates and displays a 3D plot of platforms, their sensors, and target
    volumes, with visibility results automatically color-coded.
    """
    print("Performing visibility checks and generating 3D scene...")
    fig = go.Figure()

    # --- Step 1: Perform all visibility checks to determine colors ---
    sensors_with_hits = set()
    volumes_that_are_seen = set()

    all_world_sensors = [s for p in platforms for s in p.get_all_world_sensors()]

    for i, volume in enumerate(volumes):
        for sensor in all_world_sensors:
            if sensor.can_observe_volume(volume):
                sensors_with_hits.add(id(sensor))
                volumes_that_are_seen.add(i)

    # --- Step 2: Plot all scene objects with appropriate colors ---
    for platform in platforms:
        for trace in get_platform_axes_plot(platform):
            fig.add_trace(trace)

    hit_colour = "limegreen"
    miss_colour = "red"
    for sensor in all_world_sensors:
        is_hit = id(sensor) in sensors_with_hits

        if isinstance(sensor.strategy, PyramidalSATStrategy):
            fig.add_trace(
                plot_pyramidal_for(sensor, color=hit_colour if is_hit else miss_colour)
            )
        elif isinstance(sensor.strategy, SphericalAccurateStrategy):
            for trace in plot_spherical_for(
                sensor, color=hit_colour if is_hit else miss_colour
            ):
                fig.add_trace(trace)

    for i, volume in enumerate(volumes):
        color = hit_colour if i in volumes_that_are_seen else miss_colour
        fig.add_trace(get_volume_plot(volume, name=f"Target {i + 1}", color=color))

    # --- Step 3: Configure and show the final plot ---
    fig.update_layout(
        title_text="Sensor Platform Visualization with Visibility Results",
        scene=dict(
            xaxis_title="World X",
            yaxis_title="World Y",
            zaxis_title="World Z",
            aspectmode="data",
        ),
        legend_title="Scene Objects",
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.show()
