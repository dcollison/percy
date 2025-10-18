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


def benchmark_accuracy(num_volumes: int):
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

    configs = {
        "Spherical (Accurate)": SensorConfiguration(
            "sph", **sensor_params, strategy=SphericalAccurateStrategy()
        ),
    }
    ground_truth_config = SensorConfiguration(
        "ground_truth", **sensor_params, strategy=PyramidalSATStrategy()
    )

    all_results = {}
    print(f"Running benchmark on {num_volumes} volumes...")
    for name, test_config in configs.items():
        platform = Platform(
            "p1",
            np.zeros(3),
            np.zeros(3),
            components=[ground_truth_config, test_config],
        )
        stats = defaultdict(int)
        for volume in volumes:
            ground_truth = platform.can_observe_volume(
                volume, sensor_names=["ground_truth"]
            )
            strategy_result = platform.can_observe_volume(
                volume, sensor_names=[test_config.name]
            )

            if strategy_result and ground_truth:
                stats["TP"] += 1
            elif strategy_result and not ground_truth:
                stats["FP"] += 1
            elif not strategy_result and not ground_truth:
                stats["TN"] += 1
            elif not strategy_result and ground_truth:
                stats["FN"] += 1
        all_results[name] = stats

    print("\n--- Accuracy Results (vs. Pyramidal SAT Ground Truth) ---")
    for name, stats in all_results.items():
        tp, fp, tn, fn = stats["TP"], stats["FP"], stats["TN"], stats["FN"]
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        print(f"  Strategy: {name}")
        print(f"    TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        print(f"    Accuracy: {accuracy:.2%}, Precision: {precision:.2%}")


if __name__ == "__main__":
    NUM_OBJECTS = 5000

    perf_title, perf_labels, perf_values = benchmark_performance(NUM_OBJECTS)
    benchmark_accuracy(NUM_OBJECTS)

    print("\nGenerating comparison graphs...")
    fig = go.Figure(
        data=[
            go.Bar(
                x=perf_labels,
                y=perf_values,
                text=[f"{v:.2f} ms" for v in perf_values],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title_text=f"<b>{perf_title}</b><br>({NUM_OBJECTS} objects)",
        yaxis_title="Total Execution Time (ms)",
    )
    fig.show()
