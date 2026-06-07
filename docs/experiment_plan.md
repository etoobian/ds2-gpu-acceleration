# Experiment Plan

The experiments demonstrate principles rather than universal hardware benchmarks. Timing results are treated as hardware-specific case studies.

The project compares:

* Local CPU
* Local laptop GPU
* PSU ORCA CPU/GPU resources
* ORCA multi-GPU `nn.DataParallel`

## Experiment Summary

| Experiment                            | Main question                                                                             |
| ------------------------------------- | ----------------------------------------------------------------------------------------- |
| Matrix multiplication size sweep      | At what problem size does GPU parallel throughput outweigh overhead?                      |
| CIFAR-10 batch size and throughput    | How does batch size affect runtime, throughput, and accuracy in a real CNN training loop? |
| CPU-GPU transfer overhead             | How much does repeated CPU-GPU memory movement cost?                                      |
| Vectorized operations vs Python loops | Why does tensor-level code matter for GPU acceleration?                                   |
| Single GPU vs `nn.DataParallel`       | When does splitting a batch across multiple GPUs help?                                    |

## Experiment 1: Matrix Multiplication Size Sweep

### Purpose

Show when GPU parallel throughput begins to outweigh overhead.

### Computation

For increasing matrix sizes,

$$
A, B \in \mathbb{R}^{n \times n},
\qquad
C = AB.
$$

The experiment measures matrix multiplication runtime on CPU and CUDA devices.

### Final settings

```python
sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
warmup = 10
repeats = 30
dtype = "float32"
```

### Devices/environments

* Local CPU
* Local GPU
* ORCA CPU
* ORCA GPU

### Output files

```text
results/local/matmul_local.csv
results/orca/matmul_orca.csv
results/matmul_summary.csv

figures/matmul_runtime.png
figures/matmul_speedup.png
```

### Main takeaway

For small matrices, CPU runtime can be competitive because overhead dominates. For large matrices, GPU runtime becomes much faster because the computation exposes enough parallel work.

## Experiment 2: CIFAR-10 Batch Size and Neural-Network Throughput

### Purpose

Show how batch size affects training runtime, training throughput, and accuracy in a recognizable deep-learning workload.

### Computation

The experiment trains `ProjectCIFAR10CNN` on CIFAR-10 using several batch sizes. The model has 2,193,226 trainable parameters.

### Final settings

```python
batch_sizes = [20, 50, 100, 200, 500, 1000]
epochs = 10
learning_rate = 0.01
optimizer = "SGD"
seed = 0
num_workers = 0
```

### Devices/environments

* Local CPU
* Local GPU
* ORCA CPU
* ORCA GPU

### Timing definition

Training runtime measures the actual training loop:

* Data loading
* Device transfer
* Forward pass
* Loss computation
* Backward pass
* Optimizer step

Final train accuracy and final test accuracy are evaluated separately after training, so accuracy evaluation is not included in training runtime.

### Output files

```text
results/local/batch_size_local_cpu.csv
results/local/batch_size_local_gpu.csv
results/orca/batch_size_orca_cpu.csv
results/orca/batch_size_orca_gpu.csv
results/batch_size_summary.csv

figures/batch_size_train_runtime.png
figures/batch_size_train_throughput.png
figures/batch_size_test_accuracy.png
```

### Main takeaway

Larger batch sizes improved training runtime and throughput, especially on GPUs. However, accuracy decreased for larger batches in this fixed-epoch, fixed-learning-rate setup because larger batches receive fewer optimizer updates per epoch.

## Experiment 3: CPU-GPU Transfer Overhead

### Purpose

Show that GPU acceleration can be reduced or eliminated when data is moved repeatedly between CPU and GPU.

### Computation

The experiment compares several cases, with the main contrast between:

```python
# Bad pattern
for _ in range(repeats):
    y_gpu = x_cpu.to("cuda")
    y_gpu = y_gpu * 1.000001 + 0.000001
    y_cpu = y_gpu.cpu()
```

and

```python
# Good pattern
y_gpu = x_cpu.to("cuda")

for _ in range(repeats):
    y_gpu = y_gpu * 1.000001 + 0.000001

y_cpu = y_gpu.cpu()
```

### Final settings

```python
sizes = [10000, 100000, 1000000, 10000000]
repeats = 50
dtype = "float32"
```

### Devices/environments

* Local GPU
* ORCA GPU

### Output files

```text
results/local/transfer_local.csv
results/orca/transfer_orca.csv
results/transfer_summary.csv

figures/transfer_bad_vs_good.png
```

### Main takeaway

Repeated CPU-GPU transfers were much slower than moving data to the GPU once and keeping it there. For the largest tensor tested, the repeated-transfer pattern was about $26\times$ slower locally and about $31\times$ slower on ORCA.

## Experiment 4: Vectorized Tensor Operations Versus Python Loops

### Purpose

Show why expressing computation as tensor operations matters.

### Computation

The experiment compares two ways of computing the same elementwise operation:

$$y_i = 1.000001x_i + 0.000001$$

Vectorized tensor operation:

```python
y = x * 1.000001 + 0.000001
```

Python scalar loop:

```python
for i in range(x.numel()):
    y[i] = x[i] * 1.000001 + 0.000001
```

### Final settings

```python
sizes = [1000, 10000, 100000, 1000000]
vectorized_repeats = 100
loop_repeats = 1
dtype = "float32"
```

### Devices/environments

* Local CPU
* Local GPU
* ORCA CPU
* ORCA GPU

### Output files

```text
results/local/vectorization_local_cpu.csv
results/local/vectorization_local_gpu.csv
results/orca/vectorization_orca_cpu.csv
results/orca/vectorization_orca_gpu.csv
results/vectorization_summary.csv

figures/vectorization_runtime.png
figures/vectorization_loop_ratio.png
```

### Main takeaway

The Python scalar loop was thousands to over a million times slower than the vectorized tensor operation, depending on device and tensor size. The arithmetic is essentially the same, but the execution model is very different.

## Experiment 5: Single GPU Versus `nn.DataParallel`

### Purpose

Show how `nn.DataParallel` splits a mini-batch across multiple GPUs and when multi-GPU execution becomes faster than single-GPU execution.

### Computation

The experiment uses synthetic CIFAR-shaped input batches with `ProjectCIFAR10CNN`.

Input shape:

$$\texttt{batch\_size} \times 3 \times 32 \times 32$$


The timing benchmark compares:

* One ORCA GPU
* Four ORCA GPUs using `nn.DataParallel`

### Final settings

```python
batch_sizes = [128, 256, 512, 1024, 2048, 4096]
warmup = 5
repeats = 20
timing_cases = ["forward_only", "forward_backward"]
max_gpus = 4
```

### Devices/environments

* ORCA single GPU
* ORCA four-GPU `nn.DataParallel`

### Output files

```text
results/orca/dataparallel_orca.csv
results/dataparallel_summary.csv

figures/dataparallel_runtime.png
figures/dataparallel_speedup.png
```

### Main takeaway

For small batches, `nn.DataParallel` was slower because splitting, gathering, and synchronization overhead dominated. For larger batches, `nn.DataParallel` became faster. At batch size 4096, the forward-only speedup was about $2.95\times$ and the forward+backward speedup was about $3.50\times$ on four GPUs.

## Timing Methodology

GPU timing requires care because CUDA operations can be asynchronous. Timing code used:

* Warmup iterations
* Repeated timed runs
* Consistent dtype
* Controlled device placement
* No unnecessary printing inside timed loops
* `torch.cuda.synchronize()` before and after timed CUDA regions

A typical CUDA timing pattern is:

```python
torch.cuda.synchronize()
start = time.perf_counter()

# CUDA work being timed
torch.cuda.synchronize()
end = time.perf_counter()

elapsed = end - start
```

## Interpretation Notes

These experiments are case studies, not universal benchmark claims. Results depend on hardware, software versions, driver/CUDA environment, PyTorch version, power settings, scheduler allocation, tensor sizes, batch sizes, and timing methodology.

The consistent teaching conclusion is:

> GPU acceleration works best when computations are large, parallel, vectorized, batched, and kept on the GPU.