#!/bin/bash
#SBATCH --job-name=verify_JPL_0.001
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=61G
#SBATCH --time=3:00:00
#SBATCH --array=0-449
#SBATCH --output=logs/verify_JPL_0.001_%a.out

module load StdEnv/2023
module load python/3.11
source $HOME/eml-verification/.venv_abc/bin/activate

export OMP_NUM_THREADS=1

EPSILON=0.001

for (( X=0; X<20; X++ )); do
    ID=$((SLURM_ARRAY_TASK_ID * 20 + X))
    LOGFILE="logs_verification/JPL_${EPSILON}_${ID}.out"

    {
        srun python code/property_gen/generate_property_JPL.py \
        --epsilon $EPSILON --index $ID --job $SLURM_ARRAY_TASK_ID

        start_time=$(date +%s)

        timeout 5m srun python $HOME/eml-verification/alpha-beta-CROWN/complete_verifier/abcrown.py \
        --config $HOME/eml-verification/properties/JPL_${EPSILON}_${SLURM_ARRAY_TASK_ID}.yaml

        end_time=$(date +%s)
        elapsed=$((end_time - start_time))

        echo
        echo "RUNTIME: $elapsed"

    } &> "$LOGFILE"
done