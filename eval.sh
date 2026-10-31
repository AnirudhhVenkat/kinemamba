#!/bin/bash
#SBATCH --job-name=kinemamba_eval
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.venkat@ufl.edu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --mem=32000
#SBATCH --partition=hpg-turin

module load conda
conda activate kinemamba

mkdir -p logs


python eval_rollout.py --checkpoint checkpoints_mamba2_cv/a1_b1_c1/best.pt --horizon 20 --context-steps 20 --num-samples 0 --batch-size 1
python eval_rollout.py --checkpoint checkpoints_mamba2_cv/a1_b1_c1_m10/best.pt --horizon 20 --context-steps 20 --num-samples 0 --batch-size 1
python eval_rollout.py --checkpoint checkpoints_mamba2_cv/a1_b1_c1_m20/best.pt --horizon 20 --context-steps 20 --num-samples 0 --batch-size 1
python eval_rollout.py --checkpoint checkpoints_mamba2_cv/a1_b1_c1_m30/best.pt --horizon 20 --context-steps 20 --num-samples 0 --batch-size 1
python eval_rollout.py --checkpoint checkpoints_mamba2_cv/a1_b1_c1_m40/best.pt --horizon 20 --context-steps 20 --num-samples 0 --batch-size 1
python eval_rollout.py --checkpoint checkpoints_mamba2_cv/a1_b1_c1_m50/best.pt --horizon 20 --context-steps 20 --num-samples 0 --batch-size 1


