#!/bin/bash
#PBS -N brainnet_exp1
#PBS -l select=1:ncpus=4:mem=8gb:ngpus=1
#PBS -l walltime=05:00:00 
#PBS -j oe
#PBS -o job_logs/train.out
#PBS -e job_logs/train.err

cd $PBS_O_WORKDIR
mkdir -p job_logs

CONFIG=${CONFIG:-configs/config_template.json}
CONFIG_NAME=$(basename "$CONFIG" .json)

eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
conda activate brain-network-env

export OUTPUT_DIR=/rds/general/user/psp20/home/Brain-Network/runs/stanford_dogs/batch_9
TRAIN_CMD="python source/simple_cnn.py -c $CONFIG"

MAX_RETRIES=3
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]; do
    ATTEMPT_LOG="job_logs/attempt_${PBS_JOBID}_${RETRY}_${CONFIG_NAME}"

    echo "Starting training attempt $((RETRY+1)) on $(hostname) at $(date)"
    echo "Using config: $CONFIG"

    $TRAIN_CMD >> "${ATTEMPT_LOG}.out" 2>> "${ATTEMPT_LOG}.err"
    EXIT_CODE=$?

    if grep -qi "uncorrectable ECC error" "${ATTEMPT_LOG}.err"; then
        echo "Detected ECC Error - retrying (attempt $((RETRY+2)))" | tee -a "${ATTEMPT_LOG}.out"
        RETRY=$((RETRY+1))
        sleep 10
    elif [ $EXIT_CODE -eq 0 ]; then
        echo "Completed successfully at $(date)" | tee -a "${ATTEMPT_LOG}.out"
        break
    else
        echo "Training failed with unexpected error (code $EXIT_CODE). Check logs." | tee -a "${ATTEMPT_LOG}.out"
        break
    fi
done    

if [ $RETRY -ge $MAX_RETRIES ]; then
    echo "Reached maximum retry limit. Job failed with ECC errors. Attempting requeue..." | tee -a "${ATTEMPT_LOG}.out"

    # Resubmit itself
    qsub $0 -v CONFIG=$CONFIG
    echo "Resubmitted job at $(date)" | tee -a "${ATTEMPT_LOG}.out"
fi

FINAL_LOG_DIR="job_logs/job_${PBS_JOBID}_${CONFIG_NAME}"
mkdir -p "$FINAL_LOG_DIR"
mv job_logs/attempt_${PBS_JOBID}_*_${CONFIG_NAME}.out "$FINAL_LOG_DIR"/
mv job_logs/attempt_${PBS_JOBID}_*_${CONFIG_NAME}.err "$FINAL_LOG_DIR"/

mv job_logs/train_${PBS_JOBID}.out "$FINAL_LOG_DIR/"
if [ -f job_logs/train_${PBS_JOBID}.err ]; then
    mv job_logs/train_${PBS_JOBID}.err "$FINAL_LOG_DIR/"
fi

echo "Job cleanup done. Logs stored in $FINAL_LOG_DIR"
