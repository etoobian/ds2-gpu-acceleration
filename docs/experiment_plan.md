# Experiment Plan

The experiments demonstrate principles rather than universal hardware benchmarks. Timing results are treated as hardware-specific case studies.

The project compares:

- Local CPU
- Local laptop GPU
- PSU ORCA cluster GPU

## Experiment 1: Matrix Multiplication Size Sweep

### Purpose

This experiment tests when GPU parallel throughput begins to outweigh overhead.

### Computation

For increasing matrix sizes,

$$A, B \in \mathbb{R}^{n \times n}$$

and

$$C = AB,$$

the experiment measures the runtime of matrix multiplication on different devices.

### Matrix sizes

```python
sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
```

### Expected behavior

For small matrices, the CPU may be competitive or faster because GPU overhead dominates. For larger matrices, the GPU should become faster because the computation exposes enough parallel work.

### Expected outputs

- Runtime versus matrix size
- GPU speedup versus matrix size
- Comparison of local CPU, local GPU, and ORCA GPU results

## Experiment 2: Batch Size and Neural-Network Throughput

### Purpose

This experiment tests how batch size affects runtime, throughput, and GPU utilization for neural-network computation.

### Computation

A small neural network is run using several batch sizes. The experiment records:

- Forward pass runtime
- Forward plus backward pass runtime
- Examples per second
- Final training accuracy
- Final test accuracy
- GPU memory use, when available

### Batch sizes

```python
batch_sizes = [1, 8, 32, 128, 512, 1024, 2048]
```

### Accuracy note

The primary purpose of this experiment is computational, to study throughput and GPU utilization. Accuracy may be included as a secondary measurement because batch size can also affect optimization and generalization. However, this project does not treat batch size primarily as a hyperparameter-tuning study.

### Expected outputs

- Batch size versus runtime
- Batch size versus examples per second
- Batch size versus final accuracy, when training accuracy is included
- Local CPU/local GPU comparison
- ORCA GPU comparison

## Experiment 3: CPU-GPU Transfer Overhead

### Purpose

This experiment demonstrates why memory movement matters.

### Comparison

The experiment compares strategies such as:

1. Move data to the GPU once and perform many operations.
2. Move data from CPU to GPU inside the loop.
3. Move results from GPU to CPU inside the loop.

### Expected behavior

Moving data once and reusing it should be much faster than repeatedly transferring data inside a timed loop. Repeated CPU-GPU transfers can dominate runtime and eliminate the expected GPU speedup.

### Expected outputs

- Runtime by transfer strategy
- Explanation of why CPU-GPU memory movement can be a bottleneck

## Experiment 4: Vectorized Tensor Operations Versus Python Loops

### Purpose

This experiment shows why expressing computation as tensor operations matters.

### Comparison

The experiment compares vectorized tensor code against explicit Python loops.

For example, a vectorized operation such as

```python
C = A + B
```

can be compared with a loop-based implementation.

### Expected behavior

Vectorized tensor operations can use optimized backend kernels. Python loops often prevent efficient use of GPU parallelism and can make GPU execution ineffective.

### Expected outputs

- Vectorized runtime versus loop-based runtime
- Explanation of why tensor-level programming is important for GPU acceleration

## Experiment 5: Multi-GPU or DataParallel Extension

### Purpose

This experiment demonstrates the basic idea of splitting a batch across multiple GPUs using `nn.DataParallel`. The purpose is to show that multi-GPU computation can increase available parallel throughput, but that scaling is not automatic because communication, synchronization, and data movement introduce overhead.

### Scope

This is an extension experiment. The project includes the multi-GPU concept, but the main empirical comparison does not depend on successful multi-GPU benchmark results.

### Conceptual point

Multi-GPU computation can split a batch across multiple GPUs, but scaling is not automatic. Communication, synchronization, and data movement can reduce or eliminate the expected speedup.

## Timing Methodology

GPU timing requires care because CUDA operations can be asynchronous. Timing code should include:

- Warmup iterations
- Multiple repeats
- Mean and standard deviation
- Consistent dtype
- Controlled device placement
- No unnecessary printing inside timed loops
- Synchronization when timing GPU operations

In PyTorch, GPU timing should use `torch.cuda.synchronize()` before stopping the timer.

## Expected Figures

Expected figures include:

1. Matrix multiplication runtime versus matrix size
2. Matrix multiplication speedup versus matrix size
3. Batch size versus examples per second
4. Transfer-overhead comparison
5. Vectorized versus loop-based runtime comparison
6. Multi-GPU or DataParallel result or conceptual comparison

## Expected Tables

Expected tables include:

1. Local and ORCA hardware/software environment table
2. Experiment summary table
3. Timing result summary table