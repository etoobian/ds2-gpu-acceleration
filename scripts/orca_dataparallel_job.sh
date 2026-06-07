#!/usr/bin/env bash
#SBATCH --job-name=ds2-dp
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:4
#SBATCH --output=results/orca/dataparallel_%j.out
#SBATCH --error=results/orca/dataparallel_%j.err

echo "Job started on: $(date)"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

source ~/ds2-gpu-env/bin/activate

python src/benchmark_dataparallel.py \
    --output results/orca/dataparallel_orca.csv \
    --batch-sizes 128 256 512 1024 2048 4096 \
    --warmup 5 \
    --repeats 20 \
    --timing-cases forward_only forward_backward \
    --max-gpus 4 \
    --run-demo \
    --demo-batch-size 50 \
    --demo-input-dim 10 \
    --seed 0

echo "Job finished on: $(date)"