"""
Small neural-network models for the DS2 GPU acceleration project.

These models are intentionally simple. The goal is not to maximize accuracy,
but to provide reusable workloads for timing CPU/GPU behavior.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SmallMLP(nn.Module):
    """
    Small fully connected network for vector inputs.

    This model is useful for simple batch-size and forward/backward timing
    experiments where the input is already flattened.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 512,
        output_dim: int = 10,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallCNN(nn.Module):
    """
    Small convolutional network for image-like inputs.

    Default input shape is intended for CIFAR-10-like tensors:

        N x 3 x 32 x 32

    This model is useful for demonstrating that convolutions and batched
    tensor computations are GPU-friendly workloads.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe_model(model: nn.Module) -> str:
    """Return a short model summary string."""
    return (
        f"{model.__class__.__name__}("
        f"trainable_parameters={count_parameters(model):,}"
        f")"
    )