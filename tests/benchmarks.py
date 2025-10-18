"""
Benchmark Suite for the Percy Sensor Field of Regard (FoR) Framework.

This script quantitatively compares the different FoR strategies in two key areas:
1.  Performance: Raw execution speed against a large number of random volumes.
2.  Accuracy: Compares the approximate strategies against the highly
    accurate SAT strategy to quantify their trade-offs.
"""

import time
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation

from percy import (
    Platform,
    PyramidalSATStrategy,
    SensorConfiguration,
    SphericalAccurateStrategy,
)

# ======================================================================
# Helper Functions
# ======================================================================


def create_random_cuboid(
    center_range: tuple[float, float], size_range: tuple[float, float]
) -> np.ndarray:
    size = np.random.uniform(*size_range, size=3)
    half_size = size / 2.0
    base_vertices = np.array(
        [
            [x, y, z]
            for x in [-half_size[0], half_size[0]]
            for y in [-half_size[1], half_size[1]]
            for z in [-half_size[2], half_size[2]]
        ]
    )
    random_rotation = Rotation.random()
    random_position = np.random.uniform(*center_range, size=3)
    return random_rotation.apply(base_vertices) + random_position


# ======================================================================
# Benchmark Scenarios
# ======================================================================


def benchmark_performance(num_volumes: int):
    print("\n" + "=" * 50)
    print("  Performance Benchmark")
    print("=" * 50)
    volumes = [create_random_cuboid((-50, 50), (0.5, 5)) for _ in range(num_volumes)]
    sensor_params = {
        "relative_position": np.zeros(3),
        "relative_rpy": np.zeros(3),
        "r_min": 0.1,
        "r_max": 40.0,
        "az_half_angle": np.pi / 4,
        "el_half_angle": np.pi / 4,
    }

    models = {
        "Pyramidal (Exact SAT)": SensorConfiguration(
            "sat", **sensor_params, strategy=PyramidalSATStrategy()
        ),
        "Spherical (Accurate)": SensorConfiguration(
            "acc_sph", **sensor_params, strategy=SphericalAccurateStrategy()
        ),
    }

    times = {}
    for name, model_config in models.items():
        print(f"Running {name} benchmark...")
        platform = Platform("p1", np.zeros(3), np.zeros(3), components=[model_config])
        start = time.perf_counter_ns()
        for vol in volumes:
            platform.can_observe_volume(vol)
        times[name] = (time.perf_counter_ns() - start) / 1_000_000

    print("\n--- Performance Results (Lower is Better) ---")
    print(f"Total volumes tested: {num_volumes}")
    for name, t in sorted(times.items(), key=lambda item: item[1]):
        print(f"  {name:<40}: {t:.2f} ms")

    sorted_times = dict(sorted(times.items(), key=lambda item: item[1]))
    return (
        "FoR Strategy Performance",
        list(sorted_times.keys()),
        list(sorted_times.values()),
    )


def benchmark_accuracy(num_volumes: int) -> dict[str, defaultdict[str, int]]:
    print("\n" + "=" * 50)
    print("  Accuracy Benchmark: Approximate vs. Exact")
    print("=" * 50)
    volumes = [create_random_cuboid((-50, 50), (0.5, 5)) for _ in range(num_volumes)]
    sensor_params = {
        "relative_position": np.zeros(3),
        "relative_rpy": np.zeros(3),
        "r_min": 0.1,
        "r_max": 40.0,
        "az_half_angle": np.pi / 4,
        "el_half_angle": np.pi / 4,
    }

    test_config = SensorConfiguration(
        "sph", **sensor_params, strategy=SphericalAccurateStrategy()
    )

    # --- Create a more comparable ground truth ---
    # To make a fair comparison, the ground truth pyramid must be adjusted to
    # closely match the volume of the spherical sector. We calculate a new
    # angle that makes the pyramid's corners touch the spherical boundary.
    az_half = sensor_params["az_half_angle"]
    el_half = sensor_params["el_half_angle"]
    bounding_angle = np.arctan(np.sqrt(np.tan(az_half) ** 2 + np.tan(el_half) ** 2))

    ground_truth_params = sensor_params.copy()
    ground_truth_params["az_half_angle"] = bounding_angle
    ground_truth_params["el_half_angle"] = bounding_angle

    ground_truth_config = SensorConfiguration(
        "ground_truth", **ground_truth_params, strategy=PyramidalSATStrategy()
    )

    all_results: dict[str, defaultdict[str, int]] = {}
    print(f"Running benchmark on {num_volumes} volumes...")

    platform = Platform(
        "p1", np.zeros(3), np.zeros(3), components=[ground_truth_config, test_config]
    )
    stats: defaultdict[str, int] = defaultdict(int)
    for volume in volumes:
        ground_truth = platform.can_observe_volume(
            volume, sensor_names=["ground_truth"]
        )
        strategy_result = platform.can_observe_volume(volume, sensor_names=["sph"])

        if strategy_result and ground_truth:
            stats["TP"] += 1
        elif strategy_result and not ground_truth:
            stats["FP"] += 1
        elif not strategy_result and not ground_truth:
            stats["TN"] += 1
        elif not strategy_result and ground_truth:
            stats["FN"] += 1
    all_results["Spherical (Accurate)"] = stats

    print("\n--- Accuracy Results (vs. Volume-Matched Pyramidal SAT) ---")
    for name, stats in all_results.items():
        tp, fp, tn, fn = stats["TP"], stats["FP"], stats["TN"], stats["FN"]
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        print(f"  Strategy: {name}")
        print(f"    TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        print(f"    Accuracy: {accuracy:.2%}, Precision: {precision:.2%}")

    return all_results


if __name__ == "__main__":
    NUM_OBJECTS = 5000

    perf_title, perf_labels, perf_values = benchmark_performance(NUM_OBJECTS)
    accuracy_results = benchmark_accuracy(NUM_OBJECTS)

    print("\nGenerating comparison graphs...")

    # --- Performance Figure ---
    fig1 = go.Figure(
        data=[
            go.Bar(
                x=perf_labels,
                y=perf_values,
                text=[f"{v:.2f} ms" for v in perf_values],
                textposition="auto",
            )
        ]
    )
    fig1.update_layout(
        title_text=f"<b>{perf_title}</b><br>({NUM_OBJECTS} objects)",
        yaxis_title="Total Execution Time (ms)",
    )

    # --- Accuracy Figure ---
    fig2 = go.Figure()
    for name, stats in accuracy_results.items():
        labels = ["TP", "FP", "TN", "FN"]
        values = [stats["TP"], stats["FP"], stats["TN"], stats["FN"]]
        fig2.add_trace(go.Bar(name=name, x=labels, y=values, text=values))

    fig2.update_layout(
        title_text=f"<b>Accuracy Benchmark Results</b><br>({NUM_OBJECTS} objects vs. SAT)",
        yaxis_title="Count",
        barmode="group",
    )

    fig1.show()
    fig2.show()
