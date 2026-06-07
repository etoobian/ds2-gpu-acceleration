#!/usr/bin/env bash
#SBATCH --job-name=ds2-env-check
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=results/orca/env_check_%j.out
#SBATCH --error=results/orca/env_check_%j.err

echo "Job started on: $(date)"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

source ~/ds2-gpu-env/bin/activate

python src/check_environment.py --output results/orca/environment_orca.txt

echo "Job finished on: $(date)"