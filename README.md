# GPU Acceleration for Deep Neural Networks

This repository contains the code, benchmark results, generated figures, documentation, and presentation materials for a DS2 final project on GPU acceleration for deep learning.

## Project Title

**GPU Acceleration for Deep Neural Networks: Tensor Parallelism, Memory Movement, and Hardware Scale**

## Overview

This project studies when GPUs accelerate deep-learning computations and when the expected speedup may fail. The project connects mathematical deep-learning operations to practical PyTorch performance across:

- Local CPU
- Local laptop GPU
- PSU ORCA CPU/GPU resources
- ORCA multi-GPU `nn.DataParallel`

The central question is:

**How do deep-learning tensor computations behave across local CPU, local GPU, and cluster GPU settings?**

The project goal is not to give a broad survey of GPU architecture or CUDA programming. Instead, the project focuses on how tensor structure, device placement, batching, memory movement, vectorization, and hardware scale affect the performance of PyTorch deep-learning workloads.

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

## Experiments

The project includes five main experiments.

1. **Matrix multiplication size sweep**  
   Measures CPU and GPU runtime as matrix size increases. This experiment illustrates the crossover point where GPU parallel throughput begins to outweigh overhead.

2. **CIFAR-10 batch size and neural-network throughput**  
   Trains `ProjectCIFAR10CNN` on CIFAR-10 using several batch sizes. The experiment measures training runtime, throughput, final train accuracy, test runtime, and test accuracy across local and ORCA environments.

3. **CPU-GPU transfer overhead**  
   Compares repeated CPU-GPU transfers against keeping data on the GPU. This experiment shows how memory movement can dominate runtime.

4. **Vectorized tensor operations versus Python loops**  
   Compares vectorized PyTorch tensor operations with explicit Python scalar loops. This experiment illustrates why GPU acceleration depends on expressing computations as tensor operations that can reach optimized backend kernels.

5. **Single GPU versus `nn.DataParallel`**  
   Demonstrates how `nn.DataParallel` splits a mini-batch across multiple GPUs and benchmarks synthetic CIFAR-shaped forward and forward+backward workloads on one ORCA GPU versus four ORCA GPUs.

## Key Outputs

The project includes:

- Python benchmark scripts in `src/`
- ORCA Slurm job scripts in `scripts/`
- Local benchmark results in `results/local/`
- ORCA benchmark results in `results/orca/`
- Summary CSV files in `results/`
- Generated figures in `figures/`
- Project documentation in `docs/`
- Final presentation materials in `presentation/`

Generated figures include:

1. `figures/matmul_runtime.png`
2. `figures/matmul_speedup.png`
3. `figures/batch_size_train_runtime.png`
4. `figures/batch_size_train_throughput.png`
5. `figures/batch_size_test_accuracy.png`
6. `figures/transfer_bad_vs_good.png`
7. `figures/vectorization_runtime.png`
8. `figures/vectorization_loop_ratio.png`
9. `figures/dataparallel_runtime.png`
10. `figures/dataparallel_speedup.png`

Summary CSV files include:

1. `results/matmul_summary.csv`
2. `results/batch_size_summary.csv`
3. `results/transfer_summary.csv`
4. `results/vectorization_summary.csv`
5. `results/dataparallel_summary.csv`

## Repository Structure

```text
ds2-gpu-acceleration/
├── README.md
├── requirements.txt
├── .gitignore
│
├── docs/
│   ├── project_overview.md              # Project motivation and technical background
│   ├── experiment_plan.md               # Experiment design and timing methodology
│   ├── orca_notes.md                    # PSU ORCA cluster workflow and run notes
│   └── proposal/
│       ├── Toobian_DS2_ProjectProposal.pdf
│       └── Toobian_DS2_Supplemental_Info_ProjectProposal.pdf
│
├── src/
│   ├── check_environment.py             # Local/ORCA environment and GPU checks
│   ├── timing_utils.py                  # Shared timing helpers
│   ├── models.py                        # Small neural-network model used in benchmarks
│   ├── benchmark_matmul.py              # Experiment 1: Matrix multiplication sweep
│   ├── benchmark_batch_size.py          # Experiment 2: Batch size and throughput
│   ├── benchmark_transfer.py            # Experiment 3: CPU-GPU transfer overhead
│   ├── benchmark_vectorization.py       # Experiment 4: Vectorized tensors vs Python loops
│   ├── benchmark_dataparallel.py        # Experiment 5: Multi-GPU/DataParallel extension
│   └── analyze_results.py               # Generate figures / tables from benchmark CSV files
│
├── scripts/
│   ├── orca_environment_job.sh          # ORCA environment-check Slurm job
│   ├── orca_matmul_job.sh               # ORCA job for Experiment 1
│   ├── orca_batch_size_cpu_job.sh       # ORCA job for Experiment 2 on CPU
│   ├── orca_batch_size_gpu_job.sh       # ORCA job for Experiment 2 on GPU
│   ├── orca_transfer_job.sh             # ORCA job for Experiment 3
│   ├── orca_vectorization_cpu_job.sh    # ORCA job for Experiment 4 on CPU
│   ├── orca_vectorization_gpu_job.sh    # ORCA job for Experiment 4 on GPU
│   └── orca_dataparallel_job.sh         # ORCA job for Experiment 5
│
├── results/
│   ├── matmul_summary.csv               # Combined results for Experiment 1
│   ├── batch_size_summary.csv           # Combined results for Experiment 2
│   ├── transfer_summary.csv             # Combined results for Experiment 3
│   ├── vectorization_summary.csv        # Combined results for Experiment 4
│   ├── dataparallel_summary.csv         # Combined results for Experiment 5
│   ├── local/                           # Local benchmark outputs
│   └── orca/                            # ORCA benchmark outputs
│
├── figures/
│   └── *.png                            # Generated plots for the presentation
│
└── presentation/
    └── final_slides.pptx                # Final slide deck
```

## Documentation

Additional documentation is stored in `docs/`:

- `docs/project_overview.md`: Project motivation, course connection, mathematical background, and PyTorch GPU concepts
- `docs/experiment_plan.md`: Final experiment settings, outputs, and main takeaways.
- `docs/orca_notes.md`: ORCA workflow, SLURM scripts, cluster environment notes, and output file records
- `docs/proposal/`: Accepted proposal documents


## Running Locally

From the repository root, activate the local Python environment and run scripts directly. 

Examples:

```powershell
python src\check_environment.py --output results\local\environment_local.txt
python src\benchmark_matmul.py --output results\local\matmul_local.csv
python src\benchmark_batch_size.py --device cuda --download --output results\local\batch_size_local_gpu.csv
python src\benchmark_transfer.py --output results\local\transfer_local.csv
python src\benchmark_vectorization.py --device cuda --output results\local\vectorization_local_gpu.csv
python src\analyze_results.py --experiment all
```

### Running on ORCA

The ORCA workflow uses Slurm job scripts in `scripts/`.

Examples:
```bash
sbatch scripts/orca_environment_job.sh
sbatch scripts/orca_matmul_job.sh
sbatch scripts/orca_batch_size_gpu_job.sh
sbatch scripts/orca_batch_size_cpu_job.sh
sbatch scripts/orca_transfer_job.sh
sbatch scripts/orca_vectorization_gpu_job.sh
sbatch scripts/orca_vectorization_cpu_job.sh
sbatch scripts/orca_dataparallel_job.sh
```

The ORCA setup, output files, and interpretation notes are documented in `docs/orca_notes.md`.

## Regenerating Figures

After benchmark CSV files are present, regenerate all summary tables and figures with:

```powershell
python src\analyze_results.py --experiment all
```

Individual experiment figures can also be regenerated:
```powershell
python src\analyze_results.py --experiment matmul
python src\analyze_results.py --experiment batch_size
python src\analyze_results.py --experiment transfer
python src\analyze_results.py --experiment vectorization
python src\analyze_results.py --experiment dataparallel
```