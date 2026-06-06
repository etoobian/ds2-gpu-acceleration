# ORCA Notes

This document records the PSU ORCA cluster component of the project.

## Purpose of the ORCA Component

The ORCA component extends the project beyond local CPU and local GPU timing by adding a research-computing GPU environment.

The cluster component supports the original project by adding:

- Cluster GPU comparison
- Practical example of GPU access through a scheduler
- Discussion of research-computing workflow
- Explanation that requesting GPU hardware is separate from writing PyTorch code that correctly uses the GPU

The goal is not to turn this project into a full ORCA tutorial. The goal is to document the ORCA details that are needed to reproduce and interpret the benchmark results.

## ORCA Hardware and Environment Context

ORCA provides a scheduler-based cluster environment with GPU-enabled compute resources. For this project, the relevant ORCA details are the hardware and software details assigned to each benchmark job.

The project records:

- GPU model assigned to the job
- GPU memory
- CPU resources assigned to the job
- System memory assigned to the job
- Python version
- PyTorch version
- CUDA availability inside PyTorch
- CUDA version reported by PyTorch
- CUDA version shown by `nvidia-smi`, when available
- Slurm job settings

These details matter because cluster benchmark results depend not only on the Python code, but also on the allocated hardware, software environment, scheduler configuration, and current cluster conditions.

## ORCA Benchmarks

The ORCA component includes:

- Environment check
- Matrix multiplication benchmark
- Batch-size throughput benchmark
- Transfer-overhead benchmark
- Vectorization benchmark
- Multi-GPU or `nn.DataParallel` extension

The multi-GPU component is treated as an extension. If full multi-GPU benchmark results are not available, the presentation explains the batch-splitting idea conceptually and discusses communication and synchronization overhead.

## ORCA Workflow Used

The ORCA workflow for this project is:

1. Log in to ORCA
2. Set up or activate a Python environment
3. Confirm PyTorch and CUDA availability
4. Submit Slurm jobs requesting GPU resources
5. Save benchmark outputs to `results/orca/`
6. Copy ORCA results back into the project repository
7. Generate or update plots
8. Compare ORCA results with local CPU and local GPU results

## Expected ORCA Output Files

The ORCA benchmark outputs are saved in `results/orca/`.

Expected ORCA output files include:

- `results/orca/environment_orca.txt`
- `results/orca/matmul_orca.csv`
- `results/orca/batch_size_orca.csv`
- `results/orca/transfer_orca.csv`
- `results/orca/vectorization_orca.csv`
- `results/orca/dataparallel_orca.csv`

If the multi-GPU benchmark is handled conceptually rather than empirically, the DataParallel output file may be omitted or replaced by notes in the presentation materials.

## ORCA Job Scripts

The ORCA Slurm scripts are stored in `scripts/`.

Expected ORCA job scripts include:

- `scripts/orca_environment_job.sh`
- `scripts/orca_matmul_job.sh`
- `scripts/orca_batch_size_job.sh`
- `scripts/orca_transfer_job.sh`
- `scripts/orca_vectorization_job.sh`
- `scripts/orca_dataparallel_job.sh`

The job scripts request the needed compute resources and run the matching Python benchmark scripts from `src/`.

## Run Records

For each ORCA run, the project records:

- Date of run
- Slurm job script used
- Slurm job ID, when available
- Number of GPUs requested
- GPU type requested or assigned
- Number of CPUs requested
- Memory requested
- Runtime limit
- Python benchmark script used
- Output file name
- Error/output log file name

These run records are used to interpret differences between local CPU, local GPU, and ORCA GPU results.

## Important Teaching Point

A cluster scheduler can allocate GPU hardware, but PyTorch code must still be written correctly to use that GPU. Requesting a GPU from Slurm is not the same thing as moving a model and tensors to `cuda`.

For this reason, the ORCA component reinforces the same PyTorch device model used in the local experiments:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

The benchmark code must still place tensors and models on the selected device.

## Result Interpretation Notes

The ORCA timings are interpreted as hardware-specific case studies, not universal benchmark claims.

Important factors include:

- GPU model
- GPU memory
- CPU resources
- Software environment
- Job allocation
- Batch size
- Tensor dtype
- Timing methodology
- Whether CPU-GPU transfers are included in the timed region
- Whether `torch.cuda.synchronize()` was used before stopping GPU timers

The main purpose of the ORCA comparison is to show how the same PyTorch tensor computations behave in a research-computing GPU environment compared with the local CPU and local laptop GPU.
