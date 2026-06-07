#!/usr/bin/env bash
#SBATCH --job-name=ds2-cifar-cpu
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=results/orca/batch_size_cpu_%j.out
#SBATCH --error=results/orca/batch_size_cpu_%j.err

echo "Job started on: $(date)"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

source ~/ds2-gpu-env/bin/activate

python src/benchmark_batch_size.py \
    --device cpu \
    --data-dir data \
    --batch-sizes 20 50 100 200 500 1000 \
    --epochs 10 \
    --learning-rate 0.01 \
    --num-workers 0 \
    --seed 0 \
    --output results/orca/batch_size_orca_cpu.csv

echo "Job finished on: $(date)"