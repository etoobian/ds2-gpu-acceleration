"""
Experiment 4: Vectorization versus Python loops.

This benchmark compares vectorized tensor operations with Python-loop-based
scalar operations.

The goal is to show that GPUs are effective when computation is expressed as
large tensor operations, but Python loops can block useful GPU acceleration by
forcing many small operations to be launched and controlled from Python.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List

import torch

from timing_utils import get_device, synchronize_if_cuda


DEFAULT_SIZES = [1000, 10000, 100000, 1000000]


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def warmup_cuda(device: torch.device) -> None:
    """Run a small CUDA warmup to avoid measuring lazy initialization overhead."""
    if device.type != "cuda":
        return

    x = torch.randn(1024, device=device)
    for _ in range(10):
        x = x * 1.000001 + 0.000001

    synchronize_if_cuda(device)


def tensor_size_mb(n: int, dtype: torch.dtype) -> float:
    """Return approximate size in MB for a vector of length n."""
    element_size = torch.tensor([], dtype=dtype).element_size()
    return n * element_size / (1024.0 ** 2)


def time_vectorized(
    x: torch.Tensor,
    repeats: int,
) -> float:
    """Time vectorized tensor operations."""
    synchronize_if_cuda(x.device)
    start = time.perf_counter()

    y = x
    for _ in range(repeats):
        y = y * 1.000001 + 0.000001

    synchronize_if_cuda(x.device)
    end = time.perf_counter()

    # Use the result after timing.
    _ = float(y[0].detach().cpu())

    return end - start


def time_python_loop_cpu(
    x: torch.Tensor,
    repeats: int,
) -> float:
    """
    Time a Python loop over CPU tensor elements.

    This intentionally uses scalar indexing in Python. It is not meant to be
    an efficient CPU implementation; it is meant to demonstrate the cost of
    leaving vectorized tensor operations.
    """
    start = time.perf_counter()

    y = torch.empty_like(x)

    for _ in range(repeats):
        for i in range(x.numel()):
            y[i] = x[i] * 1.000001 + 0.000001

    end = time.perf_counter()

    _ = float(y[0])

    return end - start


def time_python_loop_cuda(
    x: torch.Tensor,
    repeats: int,
) -> float:
    """
    Time a Python loop over CUDA tensor elements.

    This is intentionally bad GPU code. Each scalar operation is controlled
    from Python and can cause many tiny GPU operations/synchronization costs.
    """
    device = x.device
    synchronize_if_cuda(device)
    start = time.perf_counter()

    y = torch.empty_like(x)

    for _ in range(repeats):
        for i in range(x.numel()):
            y[i] = x[i] * 1.000001 + 0.000001

    synchronize_if_cuda(device)
    end = time.perf_counter()

    _ = float(y[0].detach().cpu())

    return end - start


def benchmark_one_size(
    n: int,
    device: torch.device,
    repeats: int,
    loop_repeats: int,
    dtype: torch.dtype,
    max_loop_n: int,
) -> List[Dict[str, object]]:
    """Run vectorization benchmark cases for one tensor size and one device."""
    x = torch.randn(n, dtype=dtype, device=device)

    mb = tensor_size_mb(n, dtype)
    rows: List[Dict[str, object]] = []

    print(f"\nTensor length n = {n:,} ({mb:.2f} MB), device = {device}")

    # Vectorized case.
    vectorized_seconds = time_vectorized(x, repeats)
    print(
        f"  vectorized_tensor_op: "
        f"{vectorized_seconds:.4f} sec total, "
        f"{1000 * vectorized_seconds / repeats:.4f} ms/repeat"
    )

    rows.append(
        {
            "experiment": "vectorization",
            "n": n,
            "tensor_size_mb": mb,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "case": "vectorized_tensor_op",
            "repeats": repeats,
            "total_seconds": vectorized_seconds,
            "seconds_per_repeat": vectorized_seconds / repeats,
            "skipped": False,
        }
    )

    # Python-loop case. Skip very large loops so the experiment finishes.
    if n > max_loop_n:
        print(
            f"  python_loop_scalar: skipped "
            f"(n={n:,} > max_loop_n={max_loop_n:,})"
        )

        rows.append(
            {
                "experiment": "vectorization",
                "n": n,
                "tensor_size_mb": mb,
                "device": str(device),
                "dtype": str(dtype).replace("torch.", ""),
                "case": "python_loop_scalar",
                "repeats": loop_repeats,
                "total_seconds": None,
                "seconds_per_repeat": None,
                "skipped": True,
            }
        )

        return rows

    if device.type == "cuda":
        loop_seconds = time_python_loop_cuda(x, loop_repeats)
    else:
        loop_seconds = time_python_loop_cpu(x, loop_repeats)

    print(
        f"  python_loop_scalar: "
        f"{loop_seconds:.4f} sec total, "
        f"{1000 * loop_seconds / loop_repeats:.4f} ms/repeat"
    )

    rows.append(
        {
            "experiment": "vectorization",
            "n": n,
            "tensor_size_mb": mb,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "case": "python_loop_scalar",
            "repeats": loop_repeats,
            "total_seconds": loop_seconds,
            "seconds_per_repeat": loop_seconds / loop_repeats,
            "skipped": False,
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
        "skipped",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark vectorized tensor operations versus Python loops."
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
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device to use for this run.",
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
        help="Repeats for vectorized tensor operations.",
    )
    parser.add_argument(
        "--loop-repeats",
        type=int,
        default=1,
        help="Repeats for Python-loop scalar operations.",
    )
    parser.add_argument(
        "--max-loop-n",
        type=int,
        default=100000,
        help="Skip Python-loop scalar case when n is larger than this.",
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
    dtype = torch.float32

    set_seed(args.seed)
    warmup_cuda(device)

    print("Experiment 4: Vectorization versus Python loops")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Selected device: {device}")
    print(f"Sizes: {args.sizes}")
    print(f"Vectorized repeats: {args.repeats}")
    print(f"Loop repeats: {args.loop_repeats}")
    print(f"Max loop n: {args.max_loop_n}")
    print(f"Seed: {args.seed}")

    rows: List[Dict[str, object]] = []

    for n in args.sizes:
        rows.extend(
            benchmark_one_size(
                n=n,
                device=device,
                repeats=args.repeats,
                loop_repeats=args.loop_repeats,
                dtype=dtype,
                max_loop_n=args.max_loop_n,
            )
        )

    write_csv(rows, Path(args.output))


if __name__ == "__main__":
    main()