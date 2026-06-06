"""
Timing utilities for the DS2 GPU acceleration project.

The key issue is that CUDA operations are asynchronous. When timing GPU code,
we need to synchronize before stopping the timer so that the measured time
actually includes the GPU work.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import torch


def synchronize_if_cuda(device: torch.device | str | None = None) -> None:
    """
    Synchronize CUDA operations if CUDA is being used.

    Parameters
    ----------
    device:
        Device being timed. If None, synchronize whenever CUDA is available.
    """
    if not torch.cuda.is_available():
        return

    if device is None:
        torch.cuda.synchronize()
        return

    device = torch.device(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_function(
    fn: Callable[[], Any],
    *,
    device: torch.device | str | None = None,
    warmup: int = 10,
    repeats: int = 50,
) -> dict[str, float]:
    """
    Time a function using warmup iterations and repeated measurements.

    Parameters
    ----------
    fn:
        Function with no required arguments. The function should perform the
        operation being timed.

    device:
        Device used by the function. If this is a CUDA device, timing will
        synchronize before and after each measured call.

    warmup:
        Number of untimed warmup calls.

    repeats:
        Number of timed calls.

    Returns
    -------
    dict[str, float]
        Dictionary containing mean, standard deviation, min, max, and repeats.
        Times are reported in seconds.
    """
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    for _ in range(warmup):
        fn()

    synchronize_if_cuda(device)

    times: list[float] = []

    for _ in range(repeats):
        synchronize_if_cuda(device)
        start = time.perf_counter()
        fn()
        synchronize_if_cuda(device)
        end = time.perf_counter()

        times.append(end - start)

    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0

    return {
        "mean_seconds": mean,
        "std_seconds": std,
        "min_seconds": min(times),
        "max_seconds": max(times),
        "repeats": float(repeats),
    }


def format_seconds(seconds: float) -> str:
    """Format seconds using convenient units."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.3f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.3f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.3f} ms"
    return f"{seconds:.3f} s"


def get_device(device_name: str) -> torch.device:
    """
    Return a torch.device from a string.

    Parameters
    ----------
    device_name:
        One of "cpu", "cuda", or "auto".

    Returns
    -------
    torch.device
        Selected device.
    """
    device_name = device_name.lower()

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device_name must be one of: 'cpu', 'cuda', or 'auto'.")

    return torch.device(device_name)