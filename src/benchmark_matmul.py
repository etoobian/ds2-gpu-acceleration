"""
Experiment 1: Matrix multiplication size sweep.

This benchmark measures the runtime of

    C = A @ B

for square matrices A, B in R^{n x n}, using CPU and CUDA devices when
available. The goal is to identify when GPU parallel throughput begins to
outweigh overhead.

Outputs are saved as CSV files for later plotting.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import torch

from timing_utils import time_function


DEFAULT_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def benchmark_one_size(
    n: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> Dict[str, object]:
    """Benchmark C = A @ B for one matrix size and one device."""
    A = torch.randn((n, n), device=device, dtype=dtype)
    B = torch.randn((n, n), device=device, dtype=dtype)

    def operation() -> torch.Tensor:
        return A @ B

    timing = time_function(
        operation,
        device=device,
        warmup=warmup,
        repeats=repeats,
    )

    return {
        "experiment": "matmul",
        "n": n,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "mean_seconds": timing["mean_seconds"],
        "std_seconds": timing["std_seconds"],
        "min_seconds": timing["min_seconds"],
        "max_seconds": timing["max_seconds"],
        "repeats": int(timing["repeats"]),
        "warmup": warmup,
    }


def benchmark_devices(
    devices: List[torch.device],
    sizes: List[int],
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> List[Dict[str, object]]:
    """Run the matrix multiplication benchmark for all sizes and devices."""
    results: List[Dict[str, object]] = []

    for device in devices:
        print(f"\nDevice: {device}")

        for n in sizes:
            print(f"  n = {n}")
            row = benchmark_one_size(
                n=n,
                device=device,
                dtype=dtype,
                warmup=warmup,
                repeats=repeats,
            )
            results.append(row)

            mean_ms = 1000 * float(row["mean_seconds"])
            std_ms = 1000 * float(row["std_seconds"])
            print(f"    mean = {mean_ms:.4f} ms, std = {std_ms:.4f} ms")

    return results


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    """Write benchmark results to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
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
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark matrix multiplication on CPU and/or CUDA."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/local/matmul_local.csv",
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help="Matrix sizes n for A, B in R^{n x n}.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["cpu", "cuda"],
        choices=["cpu", "cuda"],
        help="Devices to benchmark. CUDA is skipped if unavailable.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of untimed warmup iterations.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=50,
        help="Number of timed repeats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dtype = torch.float32
    devices: List[torch.device] = []

    for device_name in args.devices:
        if device_name == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available. Skipping CUDA.")
            continue

        devices.append(torch.device(device_name))

    if not devices:
        raise RuntimeError("No valid devices selected.")

    print("Experiment 1: Matrix multiplication size sweep")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Sizes: {args.sizes}")
    print(f"Warmup: {args.warmup}")
    print(f"Repeats: {args.repeats}")

    rows = benchmark_devices(
        devices=devices,
        sizes=args.sizes,
        dtype=dtype,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    write_csv(rows, Path(args.output))


if __name__ == "__main__":
    main()