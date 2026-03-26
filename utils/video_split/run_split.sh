#!/bin/bash
#SBATCH --job-name=split          # Job name
#SBATCH --output=logs/split_%A_%a.out   # Standard output and error log
#SBATCH --error=logs/split_%A_%a.err
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=16                  # Number of CPU cores per task
#SBATCH --mem=8G                          # Total memory
#SBATCH --time=4:00:00                    # Time limit hrs:min:sec
#SBATCH --array=0-499                       # Array range (e.g., 100 jobs)

# Create logs directory if not exists
mkdir -p logs

python split_slurm.py
