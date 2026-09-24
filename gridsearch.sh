#!/bin/bash
#SBATCH --job-name=kinemamba_gridsearch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.venkat@ufl.edu
#SBATCH --array=1-2       
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                # 1 GPU per task
#SBATCH --time=48:00:00
#SBATCH --output=logs/gridsearch_%a.out
#SBATCH --error=logs/gridsearch_%a.err
#SBATCH --mem=32000
#SBATCH --partition=hpg-b200   

module load conda
conda activate kinemamba


export TRITON_CACHE_DIR=/blue/iruchkin/a.venkat/triton_cache/$SLURM_ARRAY_TASK_ID
mkdir -p "$TRITON_CACHE_DIR"

MY_ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" gridsearch_params.txt)

echo "Starting task $SLURM_ARRAY_TASK_ID with args: $MY_ARGS"

# 2. Run Python with those arguments
python train_kinemamba.py $MY_ARGS
