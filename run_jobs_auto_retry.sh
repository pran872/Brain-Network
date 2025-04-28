#!/bin/bash
#PBS -N brainnet_exp1
#PBS -l select=1:ncpus=4:mem=8gb:ngpus=1
#PBS -l walltime=03:00:00 
#PBS -o job_logs/train.out
#PBS -e job_logs/train.err
#PBS -j oe

cd $PBS_O_WORKDIR
mkdir -p job_logs
mv job_logs/train.out job_logs/train_${PBS_JOBID}.out
mv job_logs/train.err job_logs/train_${PBS_JOBID}.err

eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
conda activate brain-network-env

export OUTPUT_DIR=/rds/general/user/psp20/home/Brain-Network/runs/stanford_dogs
CONFIG=${CONFIG:-configs/config_template.json}
TRAIN_CMD="python source/simple_cnn.py -c $CONFIG"

MAX_RETRIES=3
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]
do
    echo "Starting training attempt $((RETRY+1)) on $(hostname) at $(date)"
    echo "Using config: $CONFIG"

    $TRAIN_CMD

    EXIT_CODE=$?

    if grep -q "uncorrectable ECC error" job_logs/train_${PBS_JOBID}.err; then
        echo "Detected ECC Error - retrying job (attempt $((RETRY+2)))"
        RETRY=$((RETRY+1))
        sleep 10
    elif [ $EXIT_CODE -eq 0 ]; then
        echo "Completed successfully at $(date)"
        break
    else
        echo "Training failed with unexpected error. Check logs."
        break
    fi
done    

if [ $RETRY -eq $MAX_RETRIES ]; then
    echo "Reached maximum retry limit. Job failed."
fi
