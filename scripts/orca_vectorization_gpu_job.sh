#!/usr/bin/env bash
#SBATCH --job-name=ds2-vector-gpu
#SBATCH --partition=short
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=results/orca/vectorization_gpu_%j.out
#SBATCH --error=results/orca/vectorization_gpu_%j.err

echo "Job started on: $(date)"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

source ~/ds2-gpu-env/bin/activate

python src/benchmark_vectorization.py \
    --device cuda \
    --output results/orca/vectorization_orca_gpu.csv \
    --sizes 1000 10000 100000 1000000 \
    --repeats 100 \
    --loop-repeats 1 \
    --max-loop-n 1000000

echo "Job finished on: $(date)"