#!/bin/bash
#SBATCH --job-name=prune_model
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=0:20:00
#SBATCH --array=0-239
#SBATCH --output=logs_training/prune_model_%a.out

module load StdEnv/2023
module load python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r $HOME/requirements_main.txt

MODEL_VALUES=(0 1 2)
PRUNE_VALUES=(10 20 30 40 50 60 70 80)
SEED_VALUES=(10 20 30 40 50 60 70 80 90 100)

TASK_ID=${SLURM_ARRAY_TASK_ID}

MODEL=${MODEL_VALUES[$((TASK_ID / 80))]}
PRUNE=${PRUNE_VALUES[$(((TASK_ID / 10) % 8))]}
SEED=${SEED_VALUES[$((TASK_ID % 10))]}

echo "Prune model with model=$MODEL, seed=$SEED, prune ratio=$PRUNE %"

srun python code/ml_training/prune_model.py --model $MODEL --seed $SEED --prune $PRUNE