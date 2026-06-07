"""
Neural-network models for the DS2 GPU acceleration project.

The main model is a small CNN for CIFAR-10. 

The goal is not to maximize CIFAR-10 accuracy. The goal is to provide a
recognizable, course-connected deep-learning workload for comparing
batch-size behavior across CPU, local GPU, and ORCA GPU environments.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectCIFAR10CNN(nn.Module):
    """
    Small convolutional neural network for CIFAR-10.

    Expected input shape:

        N x 3 x 32 x 32

    Architecture:

        Conv2d(3 -> 32)  + ReLU
        Conv2d(32 -> 64) + ReLU
        MaxPool2d
        Conv2d(64 -> 128) + ReLU
        MaxPool2d
        Flatten
        Linear(128 * 8 * 8 -> 256) + ReLU
        Linear(256 -> 10)

    This model is intentionally simple and course-connected. It gives a
    convolutional workload large enough to make GPU behavior visible, while
    remaining small enough to run repeatedly across several batch sizes and
    computing environments.
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