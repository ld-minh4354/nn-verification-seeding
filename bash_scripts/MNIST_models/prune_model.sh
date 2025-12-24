#!/bin/bash
#SBATCH --job-name=MNIST_prune_model
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:20:00
#SBATCH --array=0-79
#SBATCH --output=logs_training/MNIST_prune_model_%a.out

module load StdEnv/2023
module load python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r $HOME/requirements_main.txt

PRUNE_VALUES=(10 20 30 40 50 60 70 80)
SEED_VALUES=(10 20 30 40 50 60 70 80 90 100)

TASK_ID=${SLURM_ARRAY_TASK_ID}

PRUNE=${PRUNE_VALUES[$((TASK_ID / 10))]}
SEED=${SEED_VALUES[$((TASK_ID % 10))]}

echo "Prune MNIST model with seed=$SEED, prune ratio=$PRUNE %"

srun python code/MNIST/prune_model.py --seed $SEED --prune $PRUNE