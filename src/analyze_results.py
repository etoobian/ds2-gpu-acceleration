"""
Analyze benchmark results and create summary tables/figures.

Current functionality:
    - Experiment 1: matrix multiplication size sweep

This script reads local and ORCA benchmark CSV files, computes CPU/GPU
speedups, saves a summary table, and generates presentation-ready figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

# Plot colors and line/marker types
LOCAL_COLOR = "tab:purple"
ORCA_COLOR = "#7A9A22"
REFERENCE_COLOR = "coral"

CPU_LINESTYLE = "--"
GPU_LINESTYLE = "-"
CPU_MARKER = "s"
GPU_MARKER = "o"

##############################################
# EXPERIMENT 1:  Matrix Multiplication Sweep
##############################################
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

    # Convert seconds to milliseconds for easier presentation
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

    # Mean runtimes
    summary["local_cpu_mean_ms"] = pivot_mean[("local", "cpu")]
    summary["local_gpu_mean_ms"] = pivot_mean[("local", "cuda")]
    summary["orca_cpu_mean_ms"] = pivot_mean[("orca", "cpu")]
    summary["orca_gpu_mean_ms"] = pivot_mean[("orca", "cuda")]

    # Standard deviations
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

    # Color encodes environment
    # Line style encodes device: dashed = CPU, solid = GPU
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
    
    # Interpretation annotations
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze DS2 GPU acceleration benchmark results."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="matmul",
        choices=["matmul"],
        help="Experiment results to analyze.",
    )
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
        "--summary-output",
        type=str,
        default="results/matmul_summary.csv",
        help="Path to save summary CSV.",
    )
    parser.add_argument(
        "--runtime-figure",
        type=str,
        default="figures/matmul_runtime.png",
        help="Path to save runtime figure.",
    )
    parser.add_argument(
        "--speedup-figure",
        type=str,
        default="figures/matmul_speedup.png",
        help="Path to save speedup figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.experiment == "matmul":
        results = load_matmul_results(
            local_path=Path(args.local_matmul),
            orca_path=Path(args.orca_matmul),
        )

        summary = make_matmul_summary(results)

        save_summary(summary, Path(args.summary_output))
        plot_matmul_runtime(summary, Path(args.runtime_figure))
        plot_matmul_speedup(summary, Path(args.speedup_figure))
        print_matmul_takeaways(summary)


if __name__ == "__main__":
    main()