# Presentation Outline

Working outline for the 30-minute DS2 final teaching presentation.

## Formal Title

**GPU Acceleration for Deep Neural Networks: Tensor Parallelism, Memory Movement, and Hardware Scale**

## Guiding Question

When does GPU acceleration actually improve deep-learning computation?

## Working Slide Structure

1. Title and central question
2. Course connection
3. Fleuret motivation for GPUs
4. CPU versus GPU mental model
5. Matrix multiplication parallelism
6. Linear layers and convolution
7. PyTorch device model
8. Backend libraries
9. When GPUs do not help
10. Experiment design
11. Results: matrix multiplication
12. Results: batch size and transfer overhead
13. ORCA cluster component and optional multi-GPU extension
14. Practical GPU tips
15. Summary and Q&A

## Final Message

GPU acceleration depends on tensor parallelism, batching, memory movement, and device placement.
