"""
Environment check for the DS2 GPU acceleration project.

This script records Python, PyTorch, CUDA, and GPU information for either
a local machine or the PSU ORCA cluster. It is intended to make benchmark
results easier to interpret and reproduce.

The default mode is "brief", which avoids unnecessary process-level details
from nvidia-smi. Use "--mode full" for the full nvidia-smi diagnostic output.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

import torch


def run_command(command: list[str]) -> str:
    """Run a shell command and return its output, or an error message."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout.strip()
        error = result.stderr.strip()

        if result.returncode == 0:
            return output if output else "(command produced no output)"

        return (
            f"(command exited with return code {result.returncode})\n"
            f"STDOUT:\n{output}\n\nSTDERR:\n{error}"
        )
    except FileNotFoundError:
        return f"Command not found: {' '.join(command)}"
    except Exception as exc:
        return f"Error running command {' '.join(command)}: {exc}"


def collect_nvidia_smi_info(mode: str) -> str:
    """
    Collect NVIDIA driver/GPU information.

    Parameters
    ----------
    mode:
        "brief" returns a compact GPU summary.
        "full" returns the standard full nvidia-smi output.
    """
    if mode == "brief":
        return run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,temperature.gpu,power.limit",
                "--format=csv",
            ]
        )

    if mode == "full":
        return run_command(["nvidia-smi"])

    raise ValueError("mode must be either 'brief' or 'full'.")


def collect_environment_info(mode: str = "brief") -> str:
    """Collect environment information as a formatted string."""
    lines: list[str] = []

    lines.append("=" * 80)
    lines.append("DS2 GPU Acceleration Project: Environment Check")
    lines.append("=" * 80)
    lines.append(f"Report mode: {mode}")
    lines.append("")

    lines.append("System Information")
    lines.append("-" * 80)
    if mode == "brief":
        lines.append(f"Python executable: {Path(sys.executable).name}")
    else:
        lines.append(f"Python executable: {sys.executable}")
    lines.append(f"Python version:    {sys.version.replace(chr(10), ' ')}")
    lines.append(f"Platform:          {platform.platform()}")
    lines.append(f"Processor:         {platform.processor()}")
    lines.append("")

    lines.append("PyTorch Information")
    lines.append("-" * 80)
    lines.append(f"PyTorch version:          {torch.__version__}")
    lines.append(f"CUDA available:           {torch.cuda.is_available()}")
    lines.append(f"CUDA version in PyTorch:  {torch.version.cuda}")
    lines.append(f"cuDNN version:            {torch.backends.cudnn.version()}")
    lines.append("")

    lines.append("CUDA Device Information")
    lines.append("-" * 80)

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        lines.append(f"CUDA device count: {device_count}")

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            total_memory_gb = props.total_memory / (1024**3)

            lines.append("")
            lines.append(f"Device {i}")
            lines.append(f"  Name:                     {props.name}")
            lines.append(f"  Total memory:             {total_memory_gb:.2f} GB")
            lines.append(f"  Multiprocessor count:     {props.multi_processor_count}")
            lines.append(f"  Compute capability:       {props.major}.{props.minor}")
    else:
        lines.append("No CUDA device available to PyTorch.")

    lines.append("")
    lines.append("nvidia-smi")
    lines.append("-" * 80)
    lines.append(collect_nvidia_smi_info(mode))
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record Python, PyTorch, CUDA, and GPU environment information."
    )
    parser.add_argument(
        "--mode",
        choices=["brief", "full"],
        default="brief",
        help="Environment report mode. Use 'brief' for clean public output or 'full' for full nvidia-smi diagnostics.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the environment report.",
    )
    args = parser.parse_args()

    report = collect_environment_info(mode=args.mode)
    print(report)

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"\nSaved environment report to: {output_path}")


if __name__ == "__main__":
    main()