#!/bin/bash
#SBATCH --job-name=prune_CIFAR10
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=1:00:00
#SBATCH --array=70-79
#SBATCH --output=logs_training/prune_CIFAR10_%a.out

module load StdEnv/2023
module load python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r $HOME/requirements_main.txt

PRUNE_VALUES=(10 20 30 40 50 60 70 80)
SEED_VALUES=(10 20 30 40 50 60 70 80 90 100)

NUM_J=${#SEED_VALUES[@]}
prune_index=$(( SLURM_ARRAY_TASK_ID / NUM_J ))
seed_index=$(( SLURM_ARRAY_TASK_ID % NUM_J ))

prune=${PRUNE_VALUES[$prune_index]}
seed=${SEED_VALUES[$seed_index]}

echo "Prune model for CIFAR10 with seed=$seed, prune ratio=$prune %"

srun python code/ml_training/prune_CIFAR10.py --seed $seed --prune $prune