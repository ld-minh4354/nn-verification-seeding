#!/bin/bash
#SBATCH --job-name=JPL_train_baseline_model
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=01:00:00
#SBATCH --array=0-9
#SBATCH --output=logs_training/JPL_train_baseline_model_%a.out

module load StdEnv/2023
module load python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r $HOME/requirements_main.txt

SEED_VALUES=(10 20 30 40 50 60 70 80 90 100)

TASK_ID=${SLURM_ARRAY_TASK_ID}

SEED=${SEED_VALUES[$((TASK_ID))]}

echo "Train JPL baseline model with seed=$SEED"

srun python code/JPL/train_baseline_model.py --seed $SEED
