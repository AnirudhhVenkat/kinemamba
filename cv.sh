#!/bin/bash
#SBATCH --job-name=cv_kinemamba
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.venkat@ufl.edu
#SBATCH --array=1-10
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 # 1 GPU per task
#SBATCH --mem=64000
#SBATCH --time=48:00:00
#SBATCH --output=logs/cv_%a.out
#SBATCH --error=logs/cv_%a.err
#SBATCH --partition=hpg-b200   

module load conda
conda activate kinemamba


export TRITON_CACHE_DIR=/blue/iruchkin/a.venkat/triton_cache/$SLURM_ARRAY_TASK_ID
mkdir -p "$TRITON_CACHE_DIR"

MY_ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" cv_params.txt)

echo "Starting task $SLURM_ARRAY_TASK_ID with args: $MY_ARGS"

# 2. Run Python with those arguments
python cross_validate.py $MY_ARGS
