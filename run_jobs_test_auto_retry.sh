#!/bin/bash
#PBS -N brainnet_exp1
#PBS -l select=1:ncpus=4:mem=8gb:ngpus=1
#PBS -l walltime=00:02:00 
#PBS -j oe
#PBS -o test_job_logs/test.out
#PBS -e test_job_logs/test.err

cd $PBS_O_WORKDIR
mkdir -p test_job_logs

RUNFOLDER=${RUNFOLDER:-}
RUNFOLDER_NAME=$(basename "$RUNFOLDER" )

eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
conda activate brain-network-env

TRAIN_CMD="python source/test.py --run_default_attacks --debug --run_folder $RUNFOLDER"

MAX_RETRIES=3
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]; do
    ATTEMPT_LOG="test_job_logs/attempt_${PBS_JOBID}_${RETRY}_${RUNFOLDER}"

    echo "Starting training attempt $((RETRY+1)) on $(hostname) at $(date)"
    echo "Using run folder: $RUNFOLDER"

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
        echo "Testing failed with unexpected error (code $EXIT_CODE). Check logs." | tee -a "${ATTEMPT_LOG}.out"
        break
    fi
done    

if [ $RETRY -ge $MAX_RETRIES ]; then
    echo "Reached maximum retry limit. Job failed with ECC errors. Attempting requeue..." | tee -a "${ATTEMPT_LOG}.out"

    # Resubmit itself
    qsub $0 -v RUNFOLDER=$RUNFOLDER
    echo "Resubmitted job at $(date)" | tee -a "${ATTEMPT_LOG}.out"
fi

FINAL_LOG_DIR="test_job_logs/job_${PBS_JOBID}_${RUNFOLDER_NAME}"
mkdir -p "$FINAL_LOG_DIR"
mv test_job_logs/attempt_${PBS_JOBID}_*_${RUNFOLDER_NAME}.out "$FINAL_LOG_DIR"/
mv test_job_logs/attempt_${PBS_JOBID}_*_${RUNFOLDER_NAME}.err "$FINAL_LOG_DIR"/

mv test_job_logs/train_${PBS_JOBID}.out "$FINAL_LOG_DIR/"
if [ -f test_job_logs/train_${PBS_JOBID}.err ]; then
    mv test_job_logs/train_${PBS_JOBID}.err "$FINAL_LOG_DIR/"
fi

echo "Job cleanup done. Logs stored in $FINAL_LOG_DIR"
