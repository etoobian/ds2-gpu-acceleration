"""
Experiment 5: Single-GPU versus multi-GPU DataParallel benchmark.

This experiment follows Fleuret's discussion of nn.DataParallel:
    - split the mini-batch along the first dimension,
    - send each piece to a model replica on a GPU,
    - concatenate the outputs.

The benchmark uses synthetic CIFAR-shaped tensors so that the experiment
isolates multi-GPU computation and coordination rather than dataset loading.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from models import ProjectCIFAR10CNN, count_parameters
from timing_utils import synchronize_if_cuda


DEFAULT_BATCH_SIZES = [128, 256, 512, 1024, 2048, 4096]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize_all_cuda() -> None:
    """Synchronize all visible CUDA devices."""
    if not torch.cuda.is_available():
        return

    for device_id in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_id)


class Dummy(nn.Module):
    """
    Small module used to show how DataParallel splits the input batch.

    This mirrors the spirit of Fleuret's DataParallel demonstration.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("Dummy.forward", tuple(x.size()), x.device)
        return self.module(x)


def run_split_demo(batch_size: int, input_dim: int) -> None:
    """Print a small DataParallel split demonstration."""
    device_count = torch.cuda.device_count()

    print("\nDataParallel split demo")
    print("-" * 60)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Visible CUDA device count: {device_count}")

    if device_count == 0:
        print("No CUDA devices available. Skipping DataParallel demo.")
        return

    x_cpu = torch.randn(batch_size, input_dim)
    model = Dummy(nn.Linear(input_dim, 5))

    print("\nOn CPU")
    _ = model(x_cpu)

    x_gpu = x_cpu.to("cuda:0")
    model = model.to("cuda:0")

    print("\nOn GPU without nn.DataParallel")
    _ = model(x_gpu)

    if device_count < 2:
        print("\nOnly one CUDA device is visible. Skipping multi-GPU DataParallel demo.")
        return

    print("\nOn GPU with nn.DataParallel")
    parallel_model = nn.DataParallel(model, device_ids=list(range(device_count)))
    _ = parallel_model(x_gpu)

    synchronize_all_cuda()


def make_synthetic_batch(
    batch_size: int,
    device: torch.device,
    num_classes: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create one synthetic CIFAR-shaped batch."""
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    return x, y


def make_model(
    mode: str,
    device_ids: List[int],
) -> nn.Module:
    """Create either a single-GPU model or a DataParallel model."""
    model = ProjectCIFAR10CNN()

    if mode == "single_gpu":
        return model.to("cuda:0")

    if mode == "dataparallel":
        model = model.to("cuda:0")
        return nn.DataParallel(model, device_ids=device_ids)

    raise ValueError(f"Unknown mode: {mode}")


def time_forward_only(
    model: nn.Module,
    x: torch.Tensor,
    warmup: int,
    repeats: int,
) -> float:
    """Time forward-only inference."""
    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

        synchronize_all_cuda()
        start = time.perf_counter()

        for _ in range(repeats):
            _ = model(x)

        synchronize_all_cuda()
        end = time.perf_counter()

    return end - start


def time_forward_backward(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    warmup: int,
    repeats: int,
) -> float:
    """Time forward + loss + backward."""
    criterion = nn.CrossEntropyLoss()
    model.train()

    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()

    synchronize_all_cuda()
    start = time.perf_counter()

    for _ in range(repeats):
        model.zero_grad(set_to_none=True)
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()

    synchronize_all_cuda()
    end = time.perf_counter()

    return end - start


def benchmark_one(
    batch_size: int,
    mode: str,
    timing_case: str,
    device_ids: List[int],
    warmup: int,
    repeats: int,
) -> Dict[str, object]:
    """Benchmark one batch size, mode, and timing case."""
    primary_device = torch.device("cuda:0")
    x, y = make_synthetic_batch(batch_size=batch_size, device=primary_device)

    model = make_model(mode=mode, device_ids=device_ids)
    parameter_count = count_parameters(
        model.module if isinstance(model, nn.DataParallel) else model
    )

    if timing_case == "forward_only":
        total_seconds = time_forward_only(
            model=model,
            x=x,
            warmup=warmup,
            repeats=repeats,
        )
    elif timing_case == "forward_backward":
        total_seconds = time_forward_backward(
            model=model,
            x=x,
            y=y,
            warmup=warmup,
            repeats=repeats,
        )
    else:
        raise ValueError(f"Unknown timing case: {timing_case}")

    seconds_per_repeat = total_seconds / repeats
    examples_per_second = (batch_size * repeats) / total_seconds

    print(
        f"  {mode:12s} | {timing_case:16s} | "
        f"batch={batch_size:5d} | "
        f"{1000 * seconds_per_repeat:9.4f} ms/repeat | "
        f"{examples_per_second:10.2f} examples/sec"
    )

    return {
        "experiment": "dataparallel",
        "batch_size": batch_size,
        "mode": mode,
        "timing_case": timing_case,
        "visible_cuda_devices": torch.cuda.device_count(),
        "used_cuda_devices": 1 if mode == "single_gpu" else len(device_ids),
        "device_ids": " ".join(str(i) for i in device_ids)
        if mode == "dataparallel"
        else "0",
        "model": "ProjectCIFAR10CNN",
        "parameter_count": parameter_count,
        "warmup": warmup,
        "repeats": repeats,
        "total_seconds": total_seconds,
        "seconds_per_repeat": seconds_per_repeat,
        "milliseconds_per_repeat": 1000.0 * seconds_per_repeat,
        "examples_per_second": examples_per_second,
    }


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    """Write benchmark rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment",
        "batch_size",
        "mode",
        "timing_case",
        "visible_cuda_devices",
        "used_cuda_devices",
        "device_ids",
        "model",
        "parameter_count",
        "warmup",
        "repeats",
        "total_seconds",
        "seconds_per_repeat",
        "milliseconds_per_repeat",
        "examples_per_second",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark single GPU versus nn.DataParallel."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help="Batch sizes to benchmark.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup repeats.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Timed repeats.",
    )
    parser.add_argument(
        "--timing-cases",
        type=str,
        nargs="+",
        default=["forward_only", "forward_backward"],
        choices=["forward_only", "forward_backward"],
        help="Timing cases to run.",
    )
    parser.add_argument(
        "--max-gpus",
        type=int,
        default=4,
        help="Maximum number of GPUs to use for DataParallel.",
    )
    parser.add_argument(
        "--run-demo",
        action="store_true",
        help="Run the Fleuret-style DataParallel split demo before timing.",
    )
    parser.add_argument(
        "--demo-batch-size",
        type=int,
        default=50,
        help="Batch size for the split demo.",
    )
    parser.add_argument(
        "--demo-input-dim",
        type=int,
        default=10,
        help="Input dimension for the split demo.",
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

    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 5 requires CUDA.")

    visible_devices = torch.cuda.device_count()
    if visible_devices < 1:
        raise RuntimeError("No CUDA devices visible.")

    device_ids = list(range(min(args.max_gpus, visible_devices)))

    set_seed(args.seed)

    print("Experiment 5: Single GPU vs nn.DataParallel")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Visible CUDA devices: {visible_devices}")
    print(f"DataParallel device IDs: {device_ids}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Timing cases: {args.timing_cases}")
    print(f"Warmup: {args.warmup}")
    print(f"Repeats: {args.repeats}")
    print(f"Seed: {args.seed}")

    for device_id in range(visible_devices):
        props = torch.cuda.get_device_properties(device_id)
        print(
            f"  cuda:{device_id}: {props.name}, "
            f"{props.total_memory / (1024 ** 3):.2f} GB"
        )

    if args.run_demo:
        run_split_demo(
            batch_size=args.demo_batch_size,
            input_dim=args.demo_input_dim,
        )

    rows: List[Dict[str, object]] = []

    print("\nTiming results")
    print("-" * 80)

    for batch_size in args.batch_sizes:
        for timing_case in args.timing_cases:
            rows.append(
                benchmark_one(
                    batch_size=batch_size,
                    mode="single_gpu",
                    timing_case=timing_case,
                    device_ids=[0],
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
            )

            if len(device_ids) >= 2:
                rows.append(
                    benchmark_one(
                        batch_size=batch_size,
                        mode="dataparallel",
                        timing_case=timing_case,
                        device_ids=device_ids,
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                )
            else:
                print("  Skipping DataParallel timing: Fewer than 2 GPUs visible.")

    write_csv(rows, Path(args.output))


if __name__ == "__main__":
    main()