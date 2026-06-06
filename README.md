# GPU Acceleration for Deep Neural Networks

This repository contains the code, benchmark results, figures, notes, and presentation materials for a DS2 final teaching project on GPU acceleration for deep learning.

## Project Title

**GPU Acceleration for Deep Neural Networks: Tensor Parallelism, Memory Movement, and Hardware Scale**

## Overview

This project studies when GPUs accelerate deep-learning computations and when the expected speedup may fail. The project connects mathematical deep-learning operations to practical PyTorch performance across local CPU, local GPU, and PSU ORCA cluster GPU settings.

The central question is:

**How do deep-learning tensor computations behave across local CPU, local GPU, and cluster GPU settings?**

The project is designed for a graduate-level deep learning course and builds on course topics such as tensors, batched matrix multiplication, convolutional layers, computational graphs, backpropagation, autograd, PyTorch modules, mini-batch training, and GPU use in deep learning.

The goal is not to give a broad survey of GPU architecture or CUDA programming. Instead, the project focuses on how the structure of tensor computations, device placement, batching, memory movement, and hardware scale affect the performance of deep-learning workloads.

## Central Claim

GPUs are most useful for deep-learning computations when the computation is:

- Large
- Parallel
- Vectorized
- Batched
- Kept on the GPU

GPU acceleration may be reduced or eliminated when overhead dominates, especially for:

- Small tensors
- Python loops
- Frequent CPU-GPU transfers
- Insufficient batch sizes
- Memory bottlenecks
- Device-placement mistakes
- Communication/synchronization overhead in multi-GPU settings

## Course Connections

This project connects GPU acceleration to several core ideas from the course:

- Tensor operations and tensor shapes
- Matrix multiplication
- Batched linear layers
- Convolutional layers
- Mini-batch training
- Computational graphs
- Forward and backward passes
- Backpropagation and autograd
- PyTorch modules
- PyTorch device placement
- GPU use in modern deep learning

A central theme is that deep learning computations are not only mathematical formulas. They are implemented as tensor operations, and the structure of those tensor operations strongly affects whether GPU acceleration is effective.

## Experiments

The project includes five main experiments.

1. **Matrix multiplication size sweep**  
   Measures how runtime changes as matrix size increases for local CPU, local GPU, and ORCA GPU settings. This experiment illustrates the crossover point where GPU parallel throughput begins to outweigh overhead.

2. **Batch size and neural-network throughput experiment**  
   Measures how batch size affects runtime, throughput, and possibly accuracy for a small neural network. The main focus is computational performance, but accuracy may be included as a secondary observation because batch size can also affect optimization behavior.

3. **CPU-GPU transfer overhead experiment**  
   Compares strategies such as moving data to the GPU once versus repeatedly transferring data inside a loop. This experiment demonstrates why memory movement can dominate runtime.

4. **Vectorized tensor operations versus Python loops**  
   Compares vectorized PyTorch tensor operations with explicit Python loops. This experiment illustrates why GPU acceleration depends on expressing computations as tensor operations that can reach optimized backend kernels.

5. **Multi-GPU or `nn.DataParallel` extension**  
   Studies or demonstrates the basic idea of splitting a batch across multiple GPUs. This extension is included as the multi-GPU component of the project. If full benchmark results are not available, the presentation explains the computation conceptually and discusses the expected communication and synchronization overhead.

The main comparison across the project is:

- Local CPU
- Local laptop GPU
- PSU ORCA cluster GPU

## ORCA Cluster Component

This project includes a PSU ORCA cluster component. ORCA is used to compare local GPU performance with cluster GPU performance and to introduce the research-computing workflow needed to access GPU resources through a scheduler.

The ORCA component is part of the project. It supports the original GPU-acceleration teaching goal by adding:

- Cluster GPU benchmarking
- Hardware comparison between local and research-computing environments
- Practical exposure to scheduler-based GPU access
- Discussion of the difference between requesting GPU hardware and writing PyTorch code that correctly uses the GPU

The multi-GPU extension is included as a project extension. If full multi-GPU benchmark results are not available, the extension is handled conceptually using the same device-placement and batching framework as the other experiments.

## Expected Outputs

The final project should include:

- Python benchmark scripts
- Local benchmark results
- ORCA benchmark results
- Generated figures
- A final presentation
- Speaker notes or a presentation script
- Documentation explaining the project structure, experiments, and cluster workflow

Expected figures include:

1. Matrix multiplication runtime versus matrix size
2. Matrix multiplication speedup versus matrix size
3. Batch size versus examples per second
4. Transfer-overhead comparison
5. Vectorized tensor operation versus Python-loop runtime comparison
6. Multi-GPU or DataParallel result or conceptual comparison

Expected tables include:

1. Hardware/software environment summary
2. Experiment summary table
3. Timing result summary table

## Repository Structure

```text
ds2-gpu-acceleration/
├── README.md
├── requirements.txt
├── .gitignore
│
├── docs/
│   ├── project_overview.md
│   ├── experiment_plan.md
│   ├── orca_notes.md
│   └── proposal/
│       ├── Toobian_DS2_ProjectProposal.pdf
│       └── Toobian_DS2_Supplemental_Info_ProjectProposal.pdf
│
├── src/
│   ├── check_environment.py
│   ├── timing_utils.py
│   ├── benchmark_matmul.py
│   ├── benchmark_batch_size.py
│   ├── benchmark_transfer.py
│   ├── benchmark_vectorization.py
│   ├── benchmark_dataparallel.py
│   ├── models.py
│   └── plot_results.py
│
├── notebooks/
│   ├── 00_environment_check.ipynb
│   ├── 01_bonus_batch_size_reference.ipynb
│   └── 02_results_preview.ipynb
│
├── scripts/
│   ├── run_local_all.ps1
│   ├── orca_environment_job.sh
│   ├── orca_matmul_job.sh
│   ├── orca_batch_size_job.sh
│   ├── orca_transfer_job.sh
│   ├── orca_vectorization_job.sh
│   └── orca_dataparallel_job.sh
│
├── results/
│   ├── local/
│   └── orca/
│
├── figures/
│
└── presentation/
    ├── presentation_outline.md
    ├── speaker_notes.md
    └── final_slides.pptx
```

## Documentation

Additional project details are organized in the `docs/` folder:

- `docs/project_overview.md`: Project motivation, course connection, mathematical background, PyTorch device model, and broader computational context
- `docs/experiment_plan.md`: Detailed experiment descriptions, timing methodology, expected figures, and expected tables
- `docs/orca_notes.md`: ORCA cluster notes, environment setup notes, job workflow, and run records
- `docs/proposal/`: Accepted proposal documents

Presentation materials are organized in the `presentation/` folder:

- `presentation/presentation_outline.md`: Working outline for the 30-minute teaching presentation
- `presentation/speaker_notes.md`: Speaker notes or presentation script
- `presentation/final_slides.pptx`: Final slide deck
## Running the Project

The benchmark scripts are stored in `src/`.

The local Windows workflow uses:

```powershell
scripts\run_local_all.ps1
```

The ORCA workflow uses Slurm job scripts in `scripts/`, such as:

```bash
sbatch scripts/orca_environment_job.sh
sbatch scripts/orca_matmul_job.sh
sbatch scripts/orca_batch_size_job.sh
sbatch scripts/orca_transfer_job.sh
sbatch scripts/orca_vectorization_job.sh
sbatch scripts/orca_dataparallel_job.sh
```

The exact ORCA environment setup and job settings are documented in `docs/orca_notes.md`.

## Project Workflow

The project workflow is:

1. Record local hardware/software environment information
2. Run local CPU and local GPU benchmarks
3. Run ORCA cluster GPU benchmarks
4. Save benchmark outputs in `results/`
5. Generate figures in `figures/`
6. Compare local and ORCA results
7. Build the final presentation and speaker notes in `presentation/`

The repository is organized so that code, outputs, figures, documentation, and presentation materials can be reviewed together.