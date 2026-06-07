"""
Experiment 2: CIFAR-10 batch-size runtime and accuracy benchmark.

This experiment trains a small CIFAR-10 CNN for several batch sizes and 
records runtime and accuracy.

The goal is not to find the best CIFAR-10 model. The goal is to study how
batch size affects training runtime, testing runtime, throughput, and accuracy
across CPU/GPU and local/ORCA environments.

Timing convention:
    - Training runtime measures the actual training loop only:
      data loading, device transfer, forward pass, loss, backward pass,
      and optimizer step.
    - Final train accuracy is evaluated after training and is not included
      in training runtime.
    - Final test accuracy is also evaluated separately.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import ProjectCIFAR10CNN, count_parameters
from timing_utils import get_device, synchronize_if_cuda


DEFAULT_BATCH_SIZES = [20, 50, 100, 200, 500, 1000]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cifar10_loaders(
    data_dir: Path,
    batch_size: int,
    download: bool,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 train and test DataLoaders."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=download,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=download,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> Dict[str, float]:
    """
    Train the model and record total training runtime.

    The training runtime measures the actual training loop only:
        data loading, device transfer, forward pass, loss,
        backward pass, and optimizer step.

    Final training accuracy is evaluated separately after training so that
    accuracy calculation is not mixed into the training runtime.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_examples_per_epoch = len(train_loader.dataset)
    train_examples_total = train_examples_per_epoch * epochs

    synchronize_if_cuda(device)
    start = time.perf_counter()

    for epoch in range(epochs):
        model.train()

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f"    epoch {epoch + 1:02d}/{epochs}: completed")

    synchronize_if_cuda(device)
    end = time.perf_counter()

    train_runtime = end - start

    return {
        "train_runtime_seconds": train_runtime,
        "train_examples": float(train_examples_per_epoch),
        "train_examples_total": float(train_examples_total),
        "train_examples_per_second": float(train_examples_total) / train_runtime,
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate the model and record evaluation runtime and accuracy.

    Gradients are disabled during evaluation because no training update is
    performed. This reduces memory use and avoids unnecessary autograd work.
    """
    model.eval()

    synchronize_if_cuda(device)
    start = time.perf_counter()

    correct = 0
    total = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        _, predicted = outputs.max(dim=1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    synchronize_if_cuda(device)
    end = time.perf_counter()

    eval_runtime = end - start
    accuracy = correct / total

    return {
        "eval_runtime_seconds": eval_runtime,
        "accuracy": accuracy,
        "examples": float(total),
        "examples_per_second": float(total) / eval_runtime,
    }


def benchmark_batch_size(
    batch_size: int,
    device: torch.device,
    data_dir: Path,
    download: bool,
    num_workers: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> Dict[str, object]:
    """Train/evaluate one fresh model for one batch size."""
    print(f"\nBatch size: {batch_size}")
    print(f"  device: {device}")

    set_seed(seed)

    pin_memory = device.type == "cuda"

    train_loader, test_loader = get_cifar10_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        download=download,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = ProjectCIFAR10CNN().to(device)

    parameter_count = count_parameters(model)
    print(f"  model parameters: {parameter_count:,}")

    train_metrics = train_model(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    train_eval_metrics = evaluate_model(
        model=model,
        data_loader=train_loader,
        device=device,
    )

    test_metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    print(
        f"  train runtime      = "
        f"{train_metrics['train_runtime_seconds']:.4f} sec"
    )
    print(
        f"  train eval runtime = "
        f"{train_eval_metrics['eval_runtime_seconds']:.4f} sec, "
        f"train accuracy = {100 * train_eval_metrics['accuracy']:.2f}%"
    )
    print(
        f"  test runtime       = "
        f"{test_metrics['eval_runtime_seconds']:.4f} sec, "
        f"test accuracy = {100 * test_metrics['accuracy']:.2f}%"
    )

    row: Dict[str, object] = {
        "experiment": "batch_size_cifar10",
        "batch_size": batch_size,
        "device": str(device),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "seed": seed,
        "num_workers": num_workers,
        "model": model.__class__.__name__,
        "parameter_count": parameter_count,
        "train_runtime_seconds": train_metrics["train_runtime_seconds"],
        "train_examples": train_metrics["train_examples"],
        "train_examples_total": train_metrics["train_examples_total"],
        "train_examples_per_second": train_metrics["train_examples_per_second"],
        "train_eval_runtime_seconds": train_eval_metrics["eval_runtime_seconds"],
        "train_accuracy": train_eval_metrics["accuracy"],
        "train_eval_examples": train_eval_metrics["examples"],
        "train_eval_examples_per_second": train_eval_metrics[
            "examples_per_second"
        ],
        "test_runtime_seconds": test_metrics["eval_runtime_seconds"],
        "test_accuracy": test_metrics["accuracy"],
        "test_examples": test_metrics["examples"],
        "test_examples_per_second": test_metrics["examples_per_second"],
    }

    return row


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    """Write benchmark results to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
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
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CIFAR-10 CNN batch-size runtime and accuracy."
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
        "--data-dir",
        type=str,
        default="data",
        help="Directory for CIFAR-10 data.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CIFAR-10 if not already present.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help="Batch sizes to test.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs per batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="SGD learning rate.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers.",
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
    data_dir = Path(args.data_dir)

    print("Experiment 2: CIFAR-10 batch-size benchmark")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Selected device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Num workers: {args.num_workers}")
    print(f"Seed: {args.seed}")

    rows: List[Dict[str, object]] = []

    for batch_size in args.batch_sizes:
        row = benchmark_batch_size(
            batch_size=batch_size,
            device=device,
            data_dir=data_dir,
            download=args.download,
            num_workers=args.num_workers,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        rows.append(row)

    write_csv(rows, Path(args.output))


if __name__ == "__main__":
    main()