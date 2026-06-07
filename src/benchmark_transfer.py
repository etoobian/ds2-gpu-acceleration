"""
Experiment 3: CPU-GPU transfer overhead benchmark.

This benchmark measures the cost of moving tensors between CPU and GPU and
compares two patterns:

    Bad pattern:
        repeatedly move data CPU -> GPU -> CPU inside a loop

    Good pattern:
        move data to GPU once, perform repeated computation on GPU, then
        optionally move the result back once

The goal is to show that GPU acceleration is most useful when data is kept
on the GPU and transfer overhead is avoided.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List

import torch

from timing_utils import get_device, synchronize_if_cuda


DEFAULT_SIZES = [1000, 2000, 5000, 10000]


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_size_mb(n: int, dtype: torch.dtype) -> float:
    """Return approximate size in MB for a vector of length n."""
    element_size = torch.tensor([], dtype=dtype).element_size()
    return n * element_size / (1024.0 ** 2)


def warmup_cuda(device: torch.device) -> None:
    """Run a small CUDA warmup to avoid measuring lazy initialization overhead."""
    if device.type != "cuda":
        return

    x = torch.randn(1024, device=device)
    for _ in range(10):
        x = x * 1.000001 + 0.000001

    synchronize_if_cuda(device)


def time_cpu_compute(x_cpu: torch.Tensor, repeats: int) -> float:
    """Time repeated CPU computation."""
    start = time.perf_counter()

    y = x_cpu
    for _ in range(repeats):
        y = y * 1.000001 + 0.000001

    # Use the result so the computation is not obviously unused.
    _ = float(y[0])

    end = time.perf_counter()
    return end - start


def time_gpu_compute(x_gpu: torch.Tensor, repeats: int) -> float:
    """Time repeated GPU computation when data is already on GPU."""
    synchronize_if_cuda(x_gpu.device)
    start = time.perf_counter()

    y = x_gpu
    for _ in range(repeats):
        y = y * 1.000001 + 0.000001

    synchronize_if_cuda(x_gpu.device)
    end = time.perf_counter()

    # Use a tiny synchronized result after timing.
    _ = float(y[0].detach().cpu())

    return end - start


def time_cpu_to_gpu(x_cpu: torch.Tensor, device: torch.device, repeats: int) -> float:
    """Time repeated CPU -> GPU transfers."""
    synchronize_if_cuda(device)
    start = time.perf_counter()

    for _ in range(repeats):
        y_gpu = x_cpu.to(device)

    synchronize_if_cuda(device)
    end = time.perf_counter()

    _ = float(y_gpu[0].detach().cpu())

    return end - start


def time_gpu_to_cpu(x_gpu: torch.Tensor, repeats: int) -> float:
    """Time repeated GPU -> CPU transfers."""
    device = x_gpu.device

    synchronize_if_cuda(device)
    start = time.perf_counter()

    for _ in range(repeats):
        y_cpu = x_gpu.cpu()

    synchronize_if_cuda(device)
    end = time.perf_counter()

    _ = float(y_cpu[0])

    return end - start


def time_bad_repeated_transfer(
    x_cpu: torch.Tensor,
    device: torch.device,
    repeats: int,
) -> float:
    """
    Time bad pattern:
        move CPU -> GPU, compute once, move GPU -> CPU, repeated.
    """
    synchronize_if_cuda(device)
    start = time.perf_counter()

    y_cpu = x_cpu
    for _ in range(repeats):
        y_gpu = y_cpu.to(device)
        y_gpu = y_gpu * 1.000001 + 0.000001
        y_cpu = y_gpu.cpu()

    synchronize_if_cuda(device)
    end = time.perf_counter()

    _ = float(y_cpu[0])

    return end - start


def time_good_keep_on_gpu(
    x_cpu: torch.Tensor,
    device: torch.device,
    repeats: int,
) -> float:
    """
    Time good pattern:
        move CPU -> GPU once, compute repeatedly on GPU, move GPU -> CPU once.
    """
    synchronize_if_cuda(device)
    start = time.perf_counter()

    y_gpu = x_cpu.to(device)

    for _ in range(repeats):
        y_gpu = y_gpu * 1.000001 + 0.000001

    y_cpu = y_gpu.cpu()

    synchronize_if_cuda(device)
    end = time.perf_counter()

    _ = float(y_cpu[0])

    return end - start


def benchmark_one_size(
    n: int,
    device: torch.device,
    repeats: int,
    dtype: torch.dtype,
) -> List[Dict[str, object]]:
    """Run all transfer-overhead benchmark cases for one tensor size."""
    x_cpu = torch.randn(n, dtype=dtype)
    x_gpu = x_cpu.to(device)

    mb = tensor_size_mb(n, dtype)
    rows: List[Dict[str, object]] = []

    print(f"\nTensor length n = {n:,} ({mb:.2f} MB), repeats = {repeats}")

    benchmarks = [
        ("cpu_compute_only", lambda: time_cpu_compute(x_cpu, repeats)),
        ("gpu_compute_only", lambda: time_gpu_compute(x_gpu, repeats)),
        ("cpu_to_gpu_transfer", lambda: time_cpu_to_gpu(x_cpu, device, repeats)),
        ("gpu_to_cpu_transfer", lambda: time_gpu_to_cpu(x_gpu, repeats)),
        (
            "bad_repeated_transfer",
            lambda: time_bad_repeated_transfer(x_cpu, device, repeats),
        ),
        (
            "good_keep_on_gpu",
            lambda: time_good_keep_on_gpu(x_cpu, device, repeats),
        ),
    ]

    for case, fn in benchmarks:
        seconds = fn()
        seconds_per_repeat = seconds / repeats

        print(
            f"  {case:24s}: "
            f"{seconds:.4f} sec total, "
            f"{1000 * seconds_per_repeat:.4f} ms/repeat"
        )

        rows.append(
            {
                "experiment": "transfer_overhead",
                "n": n,
                "tensor_size_mb": mb,
                "device": str(device),
                "dtype": str(dtype).replace("torch.", ""),
                "case": case,
                "repeats": repeats,
                "total_seconds": seconds,
                "seconds_per_repeat": seconds_per_repeat,
            }
        )

    return rows


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    """Write benchmark rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment",
        "n",
        "tensor_size_mb",
        "device",
        "dtype",
        "case",
        "repeats",
        "total_seconds",
        "seconds_per_repeat",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CPU-GPU transfer overhead."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "auto"],
        help="CUDA device to use. CPU-only run is not meaningful for this benchmark.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help="Tensor lengths to benchmark.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
        help="Number of repeated operations/transfers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = get_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Experiment 3 requires a CUDA device.")

    dtype = torch.float32
    set_seed(args.seed)
    warmup_cuda(device)

    print("Experiment 3: CPU-GPU transfer overhead")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Selected device: {device}")
    print(f"Sizes: {args.sizes}")
    print(f"Repeats: {args.repeats}")
    print(f"Seed: {args.seed}")

    rows: List[Dict[str, object]] = []

    for n in args.sizes:
        rows.extend(
            benchmark_one_size(
                n=n,
                device=device,
                repeats=args.repeats,
                dtype=dtype,
            )
        )

    write_csv(rows, Path(args.output))


if __name__ == "__main__":
    main()