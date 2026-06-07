"""
Analyze benchmark results and create summary tables/figures.

Current functionality:
    - Experiment 1: matrix multiplication size sweep
    - Experiment 2: CIFAR-10 batch-size benchmark

This script reads benchmark CSV files, computes useful derived quantities,
saves summary tables, and generates presentation-ready figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


# ------------------------------------------------------------
# Shared plot styling
# ------------------------------------------------------------

LOCAL_COLOR = "tab:purple"
ORCA_COLOR = "#7A9A22"
REFERENCE_COLOR = "coral"

LOCAL_VECTOR_COLOR = "#4B0082"   # dark purple
LOCAL_LOOP_COLOR = "#D55EFF"     # bright magenta-purple

ORCA_VECTOR_COLOR = "#2F6B1F"    # dark green
ORCA_LOOP_COLOR = "#B6D800"      # bright yellow-green

CPU_LINESTYLE = "--"
GPU_LINESTYLE = "-"
CPU_MARKER = "s"
GPU_MARKER = "o"


def environment_color(environment: str) -> str:
    """Return the plot color for an environment."""
    if environment == "local":
        return LOCAL_COLOR

    if environment == "orca":
        return ORCA_COLOR

    return "black"


def device_linestyle(device: str) -> str:
    """Return line style for CPU/GPU device."""
    if device == "cpu":
        return CPU_LINESTYLE

    if device == "cuda":
        return GPU_LINESTYLE

    return "-"


def device_marker(device: str) -> str:
    """Return marker style for CPU/GPU device."""
    if device == "cpu":
        return CPU_MARKER

    if device == "cuda":
        return GPU_MARKER

    return "o"


def display_label(environment: str, device: str) -> str:
    """Return a readable label such as Local CPU or ORCA GPU."""
    env_label = "Local" if environment == "local" else "ORCA"
    device_label = "GPU" if device == "cuda" else "CPU"
    return f"{env_label} {device_label}"


# ------------------------------------------------------------
# Experiment 1: Matrix multiplication size sweep
# ------------------------------------------------------------

def load_matmul_results(local_path: Path, orca_path: Path) -> pd.DataFrame:
    """Load local and ORCA matrix multiplication benchmark results."""
    local = pd.read_csv(local_path)
    local["environment"] = "local"

    orca = pd.read_csv(orca_path)
    orca["environment"] = "orca"

    combined = pd.concat([local, orca], ignore_index=True)

    required_columns = {
        "experiment",
        "n",
        "device",
        "dtype",
        "mean_seconds",
        "std_seconds",
        "min_seconds",
        "max_seconds",
        "repeats",
        "warmup",
        "environment",
    }

    missing = required_columns.difference(combined.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return combined


def make_matmul_summary(results: pd.DataFrame) -> pd.DataFrame:
    """
    Create a wide summary table for matrix multiplication results.

    The summary has one row per matrix size and columns for:
        local CPU/GPU runtime
        ORCA CPU/GPU runtime
        local GPU speedup
        ORCA GPU speedup
    """
    matmul = results[results["experiment"] == "matmul"].copy()

    # Convert seconds to milliseconds for easier presentation.
    matmul["mean_ms"] = 1000.0 * matmul["mean_seconds"]
    matmul["std_ms"] = 1000.0 * matmul["std_seconds"]

    pivot_mean = matmul.pivot_table(
        index="n",
        columns=["environment", "device"],
        values="mean_ms",
        aggfunc="mean",
    )

    pivot_std = matmul.pivot_table(
        index="n",
        columns=["environment", "device"],
        values="std_ms",
        aggfunc="mean",
    )

    summary = pd.DataFrame(index=pivot_mean.index)
    summary.index.name = "n"

    # Mean runtimes.
    summary["local_cpu_mean_ms"] = pivot_mean[("local", "cpu")]
    summary["local_gpu_mean_ms"] = pivot_mean[("local", "cuda")]
    summary["orca_cpu_mean_ms"] = pivot_mean[("orca", "cpu")]
    summary["orca_gpu_mean_ms"] = pivot_mean[("orca", "cuda")]

    # Standard deviations.
    summary["local_cpu_std_ms"] = pivot_std[("local", "cpu")]
    summary["local_gpu_std_ms"] = pivot_std[("local", "cuda")]
    summary["orca_cpu_std_ms"] = pivot_std[("orca", "cpu")]
    summary["orca_gpu_std_ms"] = pivot_std[("orca", "cuda")]

    # Speedups are computed within each environment as CPU runtime / GPU runtime.
    # Values above 1 mean the GPU was faster than the CPU for that environment.
    summary["local_gpu_speedup"] = (
        summary["local_cpu_mean_ms"] / summary["local_gpu_mean_ms"]
    )
    summary["orca_gpu_speedup"] = (
        summary["orca_cpu_mean_ms"] / summary["orca_gpu_mean_ms"]
    )

    return summary.reset_index()


def save_summary(summary: pd.DataFrame, output_path: Path) -> None:
    """Save a summary table as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Saved summary table to: {output_path}")


def plot_matmul_runtime(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot matrix multiplication runtime versus matrix size."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    # Color encodes environment.
    # Line style encodes device: dashed = CPU, solid = GPU.
    ax.plot(
        summary["n"],
        summary["local_cpu_mean_ms"],
        color=LOCAL_COLOR,
        linestyle=CPU_LINESTYLE,
        marker=CPU_MARKER,
        linewidth=1.75,
        markersize=5,
        label="Local CPU",
    )
    ax.plot(
        summary["n"],
        summary["local_gpu_mean_ms"],
        color=LOCAL_COLOR,
        linestyle=GPU_LINESTYLE,
        marker=GPU_MARKER,
        linewidth=1.75,
        markersize=5,
        label="Local GPU",
    )
    ax.plot(
        summary["n"],
        summary["orca_cpu_mean_ms"],
        color=ORCA_COLOR,
        linestyle=CPU_LINESTYLE,
        marker=CPU_MARKER,
        linewidth=1.75,
        markersize=5,
        label="ORCA CPU",
    )
    ax.plot(
        summary["n"],
        summary["orca_gpu_mean_ms"],
        color=ORCA_COLOR,
        linestyle=GPU_LINESTYLE,
        marker=GPU_MARKER,
        linewidth=1.75,
        markersize=5,
        label="ORCA GPU",
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    ax.set_title(
        "Matrix Multiplication Runtime",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(
        r"Matrix size $n$ for $A, B \in \mathbb{R}^{n \times n}$",
        fontsize=11,
        fontweight="bold",
        labelpad=8,
    )
    ax.set_ylabel(
        "Mean runtime (ms, log scale)",
        fontsize=11,
        fontweight="bold",
        labelpad=8,
    )

    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.55)
    ax.legend(fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved runtime figure to: {output_path}")


def plot_matmul_speedup(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot GPU speedup versus matrix size."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    ax.plot(
        summary["n"],
        summary["local_gpu_speedup"],
        color=LOCAL_COLOR,
        linestyle=GPU_LINESTYLE,
        marker=GPU_MARKER,
        linewidth=2.0,
        markersize=5.5,
        label="Local: CPU/GPU",
    )
    ax.plot(
        summary["n"],
        summary["orca_gpu_speedup"],
        color=ORCA_COLOR,
        linestyle=GPU_LINESTYLE,
        marker=GPU_MARKER,
        linewidth=2.0,
        markersize=5.5,
        label="ORCA: CPU/GPU",
    )

    ax.axhline(
        1.0,
        color=REFERENCE_COLOR,
        linestyle="--",
        linewidth=1,
        label="Speedup = 1",
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    ax.set_title(
        "GPU Speedup for Matrix Multiplication",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(
        r"Matrix size $n$ for $A, B \in \mathbb{R}^{n \times n}$",
        fontsize=11,
        fontweight="bold",
        labelpad=8,
    )
    ax.set_ylabel(
        "Speedup = CPU runtime / GPU runtime (log scale)",
        fontsize=11,
        fontweight="bold",
        labelpad=8,
    )

    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.55)
    ax.legend(fontsize=10, framealpha=0.9)

    # Interpretation annotations.
    ax.text(
        14,
        1.25,
        "Above 1: GPU faster",
        fontsize=10,
        color="black",
    )
    ax.text(
        14,
        0.65,
        "Below 1: CPU faster",
        fontsize=10,
        color="black",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved speedup figure to: {output_path}")


def print_matmul_takeaways(summary: pd.DataFrame) -> None:
    """Print a few useful values for interpretation."""
    largest = summary.loc[summary["n"].idxmax()]

    print("\nMatrix multiplication takeaways")
    print("-" * 60)
    print(f"Largest n: {int(largest['n'])}")
    print(f"Local CPU mean: {largest['local_cpu_mean_ms']:.4f} ms")
    print(f"Local GPU mean: {largest['local_gpu_mean_ms']:.4f} ms")
    print(f"Local GPU speedup: {largest['local_gpu_speedup']:.2f}x")
    print(f"ORCA CPU mean: {largest['orca_cpu_mean_ms']:.4f} ms")
    print(f"ORCA GPU mean: {largest['orca_gpu_mean_ms']:.4f} ms")
    print(f"ORCA GPU speedup: {largest['orca_gpu_speedup']:.2f}x")

    local_crossover = summary[summary["local_gpu_speedup"] > 1.0]
    orca_crossover = summary[summary["orca_gpu_speedup"] > 1.0]

    if not local_crossover.empty:
        print(
            "First local size with GPU speedup > 1: "
            f"n = {int(local_crossover.iloc[0]['n'])}"
        )

    if not orca_crossover.empty:
        print(
            "First ORCA size with GPU speedup > 1: "
            f"n = {int(orca_crossover.iloc[0]['n'])}"
        )


def analyze_matmul(args: argparse.Namespace) -> None:
    """Run matrix multiplication analysis."""
    results = load_matmul_results(
        local_path=Path(args.local_matmul),
        orca_path=Path(args.orca_matmul),
    )

    summary = make_matmul_summary(results)

    save_summary(summary, Path(args.matmul_summary_output))
    plot_matmul_runtime(summary, Path(args.matmul_runtime_figure))
    plot_matmul_speedup(summary, Path(args.matmul_speedup_figure))
    print_matmul_takeaways(summary)


# ------------------------------------------------------------
# Experiment 2: CIFAR-10 batch-size benchmark
# ------------------------------------------------------------

def load_batch_size_file(
    path: Path,
    environment: str,
    device: str,
) -> pd.DataFrame:
    """Load one batch-size CSV and add environment/device labels."""
    data = pd.read_csv(path)
    data["environment"] = environment
    data["device"] = device
    data["environment_device"] = data.apply(
        lambda row: display_label(row["environment"], row["device"]),
        axis=1,
    )

    return data


def load_batch_size_results(
    local_cpu_path: Path,
    local_gpu_path: Path,
    orca_cpu_path: Path,
    orca_gpu_path: Path,
) -> pd.DataFrame:
    """Load all four CIFAR-10 batch-size benchmark result files."""
    frames = [
        load_batch_size_file(local_cpu_path, "local", "cpu"),
        load_batch_size_file(local_gpu_path, "local", "cuda"),
        load_batch_size_file(orca_cpu_path, "orca", "cpu"),
        load_batch_size_file(orca_gpu_path, "orca", "cuda"),
    ]

    combined = pd.concat(frames, ignore_index=True)

    required_columns = {
        "experiment",
        "batch_size",
        "device",
        "epochs",
        "learning_rate",
        "seed",
        "num_workers",
        "model",
        "parameter_count",
        "train_runtime_seconds",
        "train_examples",
        "train_examples_total",
        "train_examples_per_second",
        "train_eval_runtime_seconds",
        "train_accuracy",
        "train_eval_examples",
        "train_eval_examples_per_second",
        "test_runtime_seconds",
        "test_accuracy",
        "test_examples",
        "test_examples_per_second",
        "environment",
        "environment_device",
    }

    missing = required_columns.difference(combined.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return combined


def make_batch_size_summary(results: pd.DataFrame) -> pd.DataFrame:
    """
    Create a presentation-friendly long summary table.

    Accuracy columns are converted to percentages.
    Runtime columns remain in seconds.
    """
    summary = results.copy()

    summary["train_accuracy_percent"] = 100.0 * summary["train_accuracy"]
    summary["test_accuracy_percent"] = 100.0 * summary["test_accuracy"]

    # Useful for plotting or slide tables.
    summary["train_runtime_minutes"] = summary["train_runtime_seconds"] / 60.0

    ordered_columns = [
        "environment",
        "device",
        "environment_device",
        "batch_size",
        "epochs",
        "learning_rate",
        "seed",
        "num_workers",
        "model",
        "parameter_count",
        "train_runtime_seconds",
        "train_runtime_minutes",
        "train_examples_per_second",
        "train_eval_runtime_seconds",
        "train_accuracy",
        "train_accuracy_percent",
        "test_runtime_seconds",
        "test_examples_per_second",
        "test_accuracy",
        "test_accuracy_percent",
    ]

    return summary[ordered_columns].sort_values(
        by=["environment", "device", "batch_size"]
    )


def plot_batch_size_train_runtime(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot CIFAR-10 training runtime versus batch size."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    for environment in ["local", "orca"]:
        for device in ["cpu", "cuda"]:
            subset = summary[
                (summary["environment"] == environment)
                & (summary["device"] == device)
            ].sort_values("batch_size")

            ax.plot(
                subset["batch_size"],
                subset["train_runtime_seconds"],
                color=environment_color(environment),
                linestyle=device_linestyle(device),
                marker=device_marker(device),
                linewidth=1.8,
                markersize=5,
                label=display_label(environment, device),
            )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title(
        "CIFAR-10 Training Runtime by Batch Size",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(
        "Batch size",
        fontsize=11,
        fontweight="bold",
        labelpad=8,
    )
    ax.set_ylabel(
        "Training runtime (seconds, log scale)",
        fontsize=11,
        fontweight="bold",
        labelpad=8,
    )

    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.55)
    ax.legend(fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved batch-size train runtime figure to: {output_path}")


def plot_batch_size_train_throughput(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot CIFAR-10 training throughput versus batch size."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    for environment in ["local", "orca"]:
        for device in ["cpu", "cuda"]:
            subset = summary[
                (summary["environment"] == environment)
                & (summary["device"] == device)
            ].sort_values("batch_size")

            ax.plot(
                subset["batch_size"],
                subset["train_examples_per_second"],
                color=environment_color(environment),
                linestyle=device_linestyle(device),
                marker=device_marker(device),
                linewidth=1.8,
                markersize=5,
                label=display_label(environment, device),
            )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title(
        "CIFAR-10 Training Throughput by Batch Size",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(
        "Batch size",
        fontsize=11,
        fontweight="bold",
        labelpad=8,
    )
    ax.set_ylabel(
        "Training throughput (examples/sec, log scale)",
        fontsize=11,
        fontweight="bold",
        labelpad=8,
    )

    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.55)
    ax.legend(fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved batch-size train throughput figure to: {output_path}")


def plot_batch_size_test_accuracy(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot CIFAR-10 test accuracy versus batch size."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    for environment in ["local", "orca"]:
        for device in ["cpu", "cuda"]:
            subset = summary[
                (summary["environment"] == environment)
                & (summary["device"] == device)
            ].sort_values("batch_size")

            ax.plot(
                subset["batch_size"],
                subset["test_accuracy_percent"],
                color=environment_color(environment),
                linestyle=device_linestyle(device),
                marker=device_marker(device),
                linewidth=1.8,
                markersize=5,
                label=display_label(environment, device),
            )

    ax.set_xscale("log")

    ax.set_title(
        "CIFAR-10 Test Accuracy by Batch Size",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(
        "Batch size",
        fontsize=11,
        fontweight="bold",
        labelpad=8,
    )
    ax.set_ylabel(
        "Test accuracy (%)",
        fontsize=11,
        fontweight="bold",
        labelpad=8,
    )

    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.55)
    ax.legend(fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved batch-size test accuracy figure to: {output_path}")


def print_batch_size_takeaways(summary: pd.DataFrame) -> None:
    """Print a few useful values for interpretation."""
    print("\nCIFAR-10 batch-size takeaways")
    print("-" * 60)

    for environment in ["local", "orca"]:
        for device in ["cpu", "cuda"]:
            subset = summary[
                (summary["environment"] == environment)
                & (summary["device"] == device)
            ].sort_values("batch_size")

            smallest = subset.iloc[0]
            largest = subset.iloc[-1]

            label = display_label(environment, device)
            print(f"{label}:")
            print(
                f"  batch {int(smallest['batch_size'])}: "
                f"train runtime = {smallest['train_runtime_seconds']:.2f} sec, "
                f"test acc = {smallest['test_accuracy_percent']:.2f}%"
            )
            print(
                f"  batch {int(largest['batch_size'])}: "
                f"train runtime = {largest['train_runtime_seconds']:.2f} sec, "
                f"test acc = {largest['test_accuracy_percent']:.2f}%"
            )


def analyze_batch_size(args: argparse.Namespace) -> None:
    """Run CIFAR-10 batch-size analysis."""
    results = load_batch_size_results(
        local_cpu_path=Path(args.local_batch_cpu),
        local_gpu_path=Path(args.local_batch_gpu),
        orca_cpu_path=Path(args.orca_batch_cpu),
        orca_gpu_path=Path(args.orca_batch_gpu),
    )

    summary = make_batch_size_summary(results)

    save_summary(summary, Path(args.batch_summary_output))
    plot_batch_size_train_runtime(summary, Path(args.batch_train_runtime_figure))
    plot_batch_size_train_throughput(summary, Path(args.batch_train_throughput_figure))
    plot_batch_size_test_accuracy(summary, Path(args.batch_test_accuracy_figure))
    print_batch_size_takeaways(summary)


# ------------------------------------------------------------
# Experiment 3: CPU-GPU transfer overhead
# ------------------------------------------------------------

def load_transfer_results(local_path: Path, orca_path: Path) -> pd.DataFrame:
    """Load local and ORCA CPU-GPU transfer-overhead benchmark results."""
    local = pd.read_csv(local_path)
    local["environment"] = "local"

    orca = pd.read_csv(orca_path)
    orca["environment"] = "orca"

    combined = pd.concat([local, orca], ignore_index=True)

    required_columns = {
        "experiment",
        "n",
        "tensor_size_mb",
        "device",
        "dtype",
        "case",
        "repeats",
        "total_seconds",
        "seconds_per_repeat",
        "environment",
    }

    missing = required_columns.difference(combined.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    combined["milliseconds_per_repeat"] = 1000.0 * combined["seconds_per_repeat"]

    return combined


def make_transfer_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Create a summary table for CPU-GPU transfer-overhead results."""
    summary = results.copy()

    bad = summary[summary["case"] == "bad_repeated_transfer"][
        ["environment", "n", "milliseconds_per_repeat"]
    ].rename(columns={"milliseconds_per_repeat": "bad_ms_per_repeat"})

    good = summary[summary["case"] == "good_keep_on_gpu"][
        ["environment", "n", "milliseconds_per_repeat"]
    ].rename(columns={"milliseconds_per_repeat": "good_ms_per_repeat"})

    ratio = pd.merge(bad, good, on=["environment", "n"])
    ratio["bad_over_good_ratio"] = (
        ratio["bad_ms_per_repeat"] / ratio["good_ms_per_repeat"]
    )

    summary = pd.merge(
        summary,
        ratio[["environment", "n", "bad_over_good_ratio"]],
        on=["environment", "n"],
        how="left",
    )

    return summary.sort_values(by=["environment", "n", "case"])


def plot_transfer_bad_vs_good(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot bad repeated transfer versus good keep-on-GPU pattern."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_cases = ["bad_repeated_transfer", "good_keep_on_gpu"]

    case_labels = {
        "bad_repeated_transfer": "Bad - Transfer every iteration",
        "good_keep_on_gpu": "Good - Keep data on GPU",
    }

    case_markers = {
        "bad_repeated_transfer": "s",
        "good_keep_on_gpu": "o",
    }

    case_linestyles = {
        "bad_repeated_transfer": "--",
        "good_keep_on_gpu": "-",
    }

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    for environment in ["local", "orca"]:
        for case in plot_cases:
            subset = summary[
                (summary["environment"] == environment)
                & (summary["case"] == case)
            ].sort_values("tensor_size_mb")

            label_env = "Local" if environment == "local" else "ORCA"
            label = f"{label_env}: {case_labels[case]}"

            ax.plot(
                subset["tensor_size_mb"],
                subset["milliseconds_per_repeat"],
                color=environment_color(environment),
                linestyle=case_linestyles[case],
                marker=case_markers[case],
                linewidth=1.8,
                markersize=5,
                label=label,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title(
        "CPU-GPU Transfer Overhead",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(
        "Tensor size (MB, log scale)",
        fontsize=11,
        fontweight="bold",
        labelpad=8,
    )
    ax.set_ylabel(
        "Time per repeat (ms, log scale)",
        fontsize=11,
        fontweight="bold",
        labelpad=8,
    )

    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.55)
    ax.legend(fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved transfer bad-vs-good figure to: {output_path}")


def print_transfer_takeaways(summary: pd.DataFrame) -> None:
    """Print transfer-overhead interpretation values."""
    print("\nCPU-GPU transfer-overhead takeaways")
    print("-" * 60)

    for environment in ["local", "orca"]:
        subset = summary[
            (summary["environment"] == environment)
            & (summary["case"] == "bad_repeated_transfer")
        ].sort_values("tensor_size_mb")

        largest = subset.iloc[-1]
        env_label = "Local" if environment == "local" else "ORCA"

        print(
            f"{env_label}, largest tensor "
            f"({largest['tensor_size_mb']:.2f} MB):"
        )
        print(
            f"  bad/good ratio = {largest['bad_over_good_ratio']:.2f}x"
        )


def analyze_transfer(args: argparse.Namespace) -> None:
    """Run CPU-GPU transfer-overhead analysis."""
    results = load_transfer_results(
        local_path=Path(args.local_transfer),
        orca_path=Path(args.orca_transfer),
    )

    summary = make_transfer_summary(results)

    save_summary(summary, Path(args.transfer_summary_output))
    plot_transfer_bad_vs_good(summary, Path(args.transfer_bad_good_figure))
    print_transfer_takeaways(summary)


# ------------------------------------------------------------
# Experiment 4: Vectorization versus Python loops
# ------------------------------------------------------------

def load_vectorization_file(
    path: Path,
    environment: str,
    device: str,
) -> pd.DataFrame:
    """Load one vectorization CSV and add environment/device labels."""
    data = pd.read_csv(path)
    data["environment"] = environment
    data["device"] = device
    data["environment_device"] = data.apply(
        lambda row: display_label(row["environment"], row["device"]),
        axis=1,
    )

    return data


def load_vectorization_results(
    local_cpu_path: Path,
    local_gpu_path: Path,
    orca_cpu_path: Path,
    orca_gpu_path: Path,
) -> pd.DataFrame:
    """Load all four vectorization benchmark result files."""
    frames = [
        load_vectorization_file(local_cpu_path, "local", "cpu"),
        load_vectorization_file(local_gpu_path, "local", "cuda"),
        load_vectorization_file(orca_cpu_path, "orca", "cpu"),
        load_vectorization_file(orca_gpu_path, "orca", "cuda"),
    ]

    combined = pd.concat(frames, ignore_index=True)

    required_columns = {
        "experiment",
        "n",
        "tensor_size_mb",
        "device",
        "dtype",
        "case",
        "repeats",
        "total_seconds",
        "seconds_per_repeat",
        "skipped",
        "environment",
        "environment_device",
    }

    missing = required_columns.difference(combined.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    combined["milliseconds_per_repeat"] = (
        1000.0 * combined["seconds_per_repeat"]
    )

    return combined


def make_vectorization_summary(results: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table for vectorization results.

    Adds a loop/vectorized ratio for each environment/device/tensor size.
    """
    summary = results.copy()

    vectorized = summary[summary["case"] == "vectorized_tensor_op"][
        ["environment", "device", "n", "milliseconds_per_repeat"]
    ].rename(columns={"milliseconds_per_repeat": "vectorized_ms_per_repeat"})

    loop = summary[summary["case"] == "python_loop_scalar"][
        ["environment", "device", "n", "milliseconds_per_repeat"]
    ].rename(columns={"milliseconds_per_repeat": "loop_ms_per_repeat"})

    ratio = pd.merge(vectorized, loop, on=["environment", "device", "n"])
    ratio["loop_over_vectorized_ratio"] = (
        ratio["loop_ms_per_repeat"] / ratio["vectorized_ms_per_repeat"]
    )

    summary = pd.merge(
        summary,
        ratio[
            [
                "environment",
                "device",
                "n",
                "vectorized_ms_per_repeat",
                "loop_ms_per_repeat",
                "loop_over_vectorized_ratio",
            ]
        ],
        on=["environment", "device", "n"],
        how="left",
    )

    return summary.sort_values(by=["environment", "device", "n", "case"])


def vectorization_color(environment: str, case: str) -> str:
    """Return color for vectorization plot."""
    if environment == "local" and case == "vectorized_tensor_op":
        return LOCAL_VECTOR_COLOR

    if environment == "local" and case == "python_loop_scalar":
        return LOCAL_LOOP_COLOR

    if environment == "orca" and case == "vectorized_tensor_op":
        return ORCA_VECTOR_COLOR

    if environment == "orca" and case == "python_loop_scalar":
        return ORCA_LOOP_COLOR

    return "black"


def plot_vectorization_runtime(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot vectorized tensor operations versus Python scalar loops."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    short_case_labels = {
        "vectorized_tensor_op": "Vec",
        "python_loop_scalar": "Loop",
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.25))

    for environment in ["local", "orca"]:
        for device in ["cpu", "cuda"]:
            for case in ["vectorized_tensor_op", "python_loop_scalar"]:
                subset = summary[
                    (summary["environment"] == environment)
                    & (summary["device"] == device)
                    & (summary["case"] == case)
                ].sort_values("n")

                label = (
                    f"{display_label(environment, device)}: "
                    f"{short_case_labels[case]}"
                )

                ax.plot(
                    subset["n"],
                    subset["milliseconds_per_repeat"],
                    color=vectorization_color(environment, case),
                    linestyle=device_linestyle(device),
                    marker=device_marker(device),
                    linewidth=2.0 if case == "python_loop_scalar" else 1.6,
                    markersize=5.5,
                    label=label,
                    alpha=0.95,
                )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title(
        "Vectorized Tensor Operations vs. Python Loops",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(
        r"Tensor length $n$",
        fontsize=11,
        fontweight="bold",
        labelpad=8,
    )
    ax.set_ylabel(
        "Time per repeat (ms, log scale)",
        fontsize=11,
        fontweight="bold",
        labelpad=8,
    )

    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.55)
    ax.legend(
        fontsize=7.5,
        framealpha=0.9,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved vectorization runtime figure to: {output_path}")


def plot_vectorization_ratio(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot Python-loop runtime divided by vectorized-runtime."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ratio_data = summary[
        summary["case"] == "vectorized_tensor_op"
    ].copy()

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    for environment in ["local", "orca"]:
        for device in ["cpu", "cuda"]:
            subset = ratio_data[
                (ratio_data["environment"] == environment)
                & (ratio_data["device"] == device)
            ].sort_values("n")

            ax.plot(
                subset["n"],
                subset["loop_over_vectorized_ratio"],
                color=environment_color(environment),
                linestyle=device_linestyle(device),
                marker=device_marker(device),
                linewidth=1.8,
                markersize=5,
                label=display_label(environment, device),
            )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title(
        "Cost of Python Loops Compared with Vectorized Tensor Operations",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(
        r"Tensor length $n$",
        fontsize=11,
        fontweight="bold",
        labelpad=8,
    )
    ax.set_ylabel(
        "Python loop time / vectorized time (log scale)",
        fontsize=11,
        fontweight="bold",
        labelpad=8,
    )

    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.55)
    ax.legend(fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved vectorization ratio figure to: {output_path}")


def print_vectorization_takeaways(summary: pd.DataFrame) -> None:
    """Print vectorization interpretation values."""
    print("\nVectorization takeaways")
    print("-" * 60)

    ratio_data = summary[
        summary["case"] == "vectorized_tensor_op"
    ].copy()

    for environment in ["local", "orca"]:
        for device in ["cpu", "cuda"]:
            subset = ratio_data[
                (ratio_data["environment"] == environment)
                & (ratio_data["device"] == device)
            ].sort_values("n")

            largest = subset.iloc[-1]
            label = display_label(environment, device)

            print(
                f"{label}, n = {int(largest['n']):,}: "
                f"Python loop/vectorized ratio = "
                f"{largest['loop_over_vectorized_ratio']:.2f}x"
            )


def analyze_vectorization(args: argparse.Namespace) -> None:
    """Run vectorization analysis."""
    results = load_vectorization_results(
        local_cpu_path=Path(args.local_vectorization_cpu),
        local_gpu_path=Path(args.local_vectorization_gpu),
        orca_cpu_path=Path(args.orca_vectorization_cpu),
        orca_gpu_path=Path(args.orca_vectorization_gpu),
    )

    summary = make_vectorization_summary(results)

    save_summary(summary, Path(args.vectorization_summary_output))
    plot_vectorization_runtime(summary, Path(args.vectorization_runtime_figure))
    plot_vectorization_ratio(summary, Path(args.vectorization_ratio_figure))
    print_vectorization_takeaways(summary)


# ------------------------------------------------------------
# Argument parsing and main
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze DS2 GPU acceleration benchmark results."
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["matmul", "batch_size", "transfer", "vectorization", "all"],
        help="Experiment results to analyze.",
    )


    # Experiment 1 paths.
    parser.add_argument(
        "--local-matmul",
        type=str,
        default="results/local/matmul_local.csv",
        help="Path to local matrix multiplication results.",
    )
    parser.add_argument(
        "--orca-matmul",
        type=str,
        default="results/orca/matmul_orca.csv",
        help="Path to ORCA matrix multiplication results.",
    )
    parser.add_argument(
        "--matmul-summary-output",
        type=str,
        default="results/matmul_summary.csv",
        help="Path to save matrix multiplication summary CSV.",
    )
    parser.add_argument(
        "--matmul-runtime-figure",
        type=str,
        default="figures/matmul_runtime.png",
        help="Path to save matrix multiplication runtime figure.",
    )
    parser.add_argument(
        "--matmul-speedup-figure",
        type=str,
        default="figures/matmul_speedup.png",
        help="Path to save matrix multiplication speedup figure.",
    )


    # Experiment 2 paths.
    parser.add_argument(
        "--local-batch-cpu",
        type=str,
        default="results/local/batch_size_local_cpu.csv",
        help="Path to local CPU CIFAR-10 batch-size results.",
    )
    parser.add_argument(
        "--local-batch-gpu",
        type=str,
        default="results/local/batch_size_local_gpu.csv",
        help="Path to local GPU CIFAR-10 batch-size results.",
    )
    parser.add_argument(
        "--orca-batch-cpu",
        type=str,
        default="results/orca/batch_size_orca_cpu.csv",
        help="Path to ORCA CPU CIFAR-10 batch-size results.",
    )
    parser.add_argument(
        "--orca-batch-gpu",
        type=str,
        default="results/orca/batch_size_orca_gpu.csv",
        help="Path to ORCA GPU CIFAR-10 batch-size results.",
    )
    parser.add_argument(
        "--batch-summary-output",
        type=str,
        default="results/batch_size_summary.csv",
        help="Path to save CIFAR-10 batch-size summary CSV.",
    )
    parser.add_argument(
        "--batch-train-runtime-figure",
        type=str,
        default="figures/batch_size_train_runtime.png",
        help="Path to save CIFAR-10 batch-size train runtime figure.",
    )
    parser.add_argument(
        "--batch-train-throughput-figure",
        type=str,
        default="figures/batch_size_train_throughput.png",
        help="Path to save CIFAR-10 batch-size train throughput figure.",
    )
    parser.add_argument(
        "--batch-test-accuracy-figure",
        type=str,
        default="figures/batch_size_test_accuracy.png",
        help="Path to save CIFAR-10 batch-size test accuracy figure.",
    )


    # Experiment 3 paths.
    parser.add_argument(
        "--local-transfer",
        type=str,
        default="results/local/transfer_local.csv",
        help="Path to local transfer-overhead results.",
    )
    parser.add_argument(
        "--orca-transfer",
        type=str,
        default="results/orca/transfer_orca.csv",
        help="Path to ORCA transfer-overhead results.",
    )
    parser.add_argument(
        "--transfer-summary-output",
        type=str,
        default="results/transfer_summary.csv",
        help="Path to save transfer-overhead summary CSV.",
    )
    parser.add_argument(
        "--transfer-bad-good-figure",
        type=str,
        default="figures/transfer_bad_vs_good.png",
        help="Path to save bad-vs-good transfer-overhead figure.",
    )


    # Experiment 4 paths.
    parser.add_argument(
        "--local-vectorization-cpu",
        type=str,
        default="results/local/vectorization_local_cpu.csv",
        help="Path to local CPU vectorization results.",
    )
    parser.add_argument(
        "--local-vectorization-gpu",
        type=str,
        default="results/local/vectorization_local_gpu.csv",
        help="Path to local GPU vectorization results.",
    )
    parser.add_argument(
        "--orca-vectorization-cpu",
        type=str,
        default="results/orca/vectorization_orca_cpu.csv",
        help="Path to ORCA CPU vectorization results.",
    )
    parser.add_argument(
        "--orca-vectorization-gpu",
        type=str,
        default="results/orca/vectorization_orca_gpu.csv",
        help="Path to ORCA GPU vectorization results.",
    )
    parser.add_argument(
        "--vectorization-summary-output",
        type=str,
        default="results/vectorization_summary.csv",
        help="Path to save vectorization summary CSV.",
    )
    parser.add_argument(
        "--vectorization-runtime-figure",
        type=str,
        default="figures/vectorization_runtime.png",
        help="Path to save vectorization runtime figure.",
    )
    parser.add_argument(
        "--vectorization-ratio-figure",
        type=str,
        default="figures/vectorization_loop_ratio.png",
        help="Path to save vectorization loop-ratio figure.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.experiment in {"matmul", "all"}:
        analyze_matmul(args)

    if args.experiment in {"batch_size", "all"}:
        analyze_batch_size(args)

    if args.experiment in {"transfer", "all"}:
        analyze_transfer(args)

    if args.experiment in {"vectorization", "all"}:
        analyze_vectorization(args)

if __name__ == "__main__":
    main()