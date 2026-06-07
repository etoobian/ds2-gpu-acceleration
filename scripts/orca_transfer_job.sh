#!/usr/bin/env bash
#SBATCH --job-name=ds2-transfer
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=results/orca/transfer_%j.out
#SBATCH --error=results/orca/transfer_%j.err

echo "Job started on: $(date)"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

source ~/ds2-gpu-env/bin/activate

python src/benchmark_transfer.py \
    --output results/orca/transfer_orca.csv \
    --sizes 10000 100000 1000000 10000000 \
    --repeats 50

echo "Job finished on: $(date)"