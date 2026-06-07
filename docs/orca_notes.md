# ORCA Notes

This document records the PSU ORCA cluster component of the project.

## Purpose

The ORCA component extends the project beyond local CPU and local GPU timing by adding a scheduler-based research-computing environment.

The purpose is to document:

* How ORCA was used
* Which Slurm jobs were run
* Which output files were produced
* How ORCA results should be interpreted

The goal is not to provide a full ORCA tutorial. The goal is to record the project-specific cluster workflow and the information needed to reproduce or interpret the benchmark results.

## Key Point

A Slurm job can allocate GPU hardware, but PyTorch code must still explicitly use that GPU.

Requesting a GPU from the scheduler is not the same as moving tensors and models to CUDA.

For example:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
x = x.to(device)
y = y.to(device)
```

The ORCA runs reinforce the same device-placement rule used locally. Tensors and models must be on the device where the computation should occur.

## Login Node Versus GPU Job

The ORCA login node can run the project Python environment and import PyTorch, but it does not expose a GPU allocation for computation.

In the project environment, the login-node check showed:

```text
CUDA available to PyTorch: False
```

Inside a Slurm job requesting a GPU, PyTorch detected the allocated GPU:

```text
CUDA available to PyTorch: True
GPU: NVIDIA L40S
```

This distinction is important for cluster computing:

* The login node is for setup, editing, and job submission
* Compute jobs are submitted through Slurm
* GPU work should be done inside an allocated GPU job

## ORCA Environment Recorded

The ORCA GPU environment check recorded:

```text
Python: 3.9.25
PyTorch: 2.5.1+cu121
CUDA available to PyTorch: True
PyTorch CUDA version: 12.1
cuDNN version: 90100
GPU: NVIDIA L40S
GPU memory: 44.64 GB
Compute capability: 8.9
NVIDIA driver: 610.43.02
```

The ORCA login-node environment check recorded:

```text
Python: 3.9.25
PyTorch: 2.5.1+cu121
CUDA available to PyTorch: False
```

The local and ORCA environments are not identical. Therefore, timing results are interpreted as hardware/software case studies rather than universal benchmark claims.

## ORCA Workflow

ORCA benchmark jobs were submitted through Slurm using the scripts in `scripts/`. Each job writes its CSV output to `results/orca/`.

The ORCA results were copied back to the local repository, where `src/analyze_results.py` was used to generate the summary tables and figures.

Slurm `.out` and `.err` files were used for checking job output during development, but they are ignored by Git and are not part of the committed benchmark results.

## ORCA Slurm Scripts

The ORCA Slurm scripts are stored in `scripts/`.

```text
scripts/orca_environment_job.sh
scripts/orca_matmul_job.sh
scripts/orca_batch_size_gpu_job.sh
scripts/orca_batch_size_cpu_job.sh
scripts/orca_transfer_job.sh
scripts/orca_vectorization_gpu_job.sh
scripts/orca_vectorization_cpu_job.sh
scripts/orca_dataparallel_job.sh
```

Each script requests the needed ORCA resources and runs the corresponding Python benchmark script from `src/`.

## ORCA Output Files

ORCA benchmark outputs are stored in `results/orca/`.

```text
results/orca/environment_orca.txt
results/orca/environment_orca_login.txt
results/orca/matmul_orca.csv
results/orca/batch_size_orca_gpu.csv
results/orca/batch_size_orca_cpu.csv
results/orca/transfer_orca.csv
results/orca/vectorization_orca_gpu.csv
results/orca/vectorization_orca_cpu.csv
results/orca/dataparallel_orca.csv
```

Slurm `.out` and `.err` files were useful during development, especially for checking job status and DataParallel split-demo output, but they are ignored by Git and are not part of the committed results.

## ORCA Benchmarks Run

The ORCA component includes:

| Experiment               | ORCA result file                          |
| ------------------------ | ----------------------------------------- |
| Environment check        | `results/orca/environment_orca.txt`       |
| Matrix multiplication    | `results/orca/matmul_orca.csv`            |
| CIFAR-10 batch size, GPU | `results/orca/batch_size_orca_gpu.csv`    |
| CIFAR-10 batch size, CPU | `results/orca/batch_size_orca_cpu.csv`    |
| Transfer overhead        | `results/orca/transfer_orca.csv`          |
| Vectorization, GPU       | `results/orca/vectorization_orca_gpu.csv` |
| Vectorization, CPU       | `results/orca/vectorization_orca_cpu.csv` |
| DataParallel             | `results/orca/dataparallel_orca.csv`      |

## DataParallel ORCA Run

The multi-GPU extension was run on ORCA using four NVIDIA L40S GPUs.

The DataParallel split demo showed that a batch of 50 was split across four GPUs:

```text
Dummy.forward (13, 10) cuda:0
Dummy.forward (13, 10) cuda:1
Dummy.forward (13, 10) cuda:2
Dummy.forward (11, 10) cuda:3
```

Thus,

```text
50 = 13 + 13 + 13 + 11
```

This output is useful for the presentation because it gives a concrete example of how `nn.DataParallel` splits the mini-batch along dimension 0.

## Result Interpretation Notes

ORCA timing results depend on:

* GPU model
* CPU allocation
* Memory allocation
* PyTorch version
* CUDA runtime and driver
* Scheduler allocation
* Tensor size
* Batch size
* Timing methodology
* CPU-GPU transfer behavior
* Use of `torch.cuda.synchronize()`

The ORCA results should be interpreted as project-specific benchmark case studies, not universal hardware claims.

The main teaching value of the ORCA component is that it shows how the same PyTorch ideas behave in a cluster setting:

* Scheduler allocation matters.
* Device placement still matters.
* GPU memory movement still matters.
* Batch size and vectorization still matter.
* Multi-GPU execution has overhead.