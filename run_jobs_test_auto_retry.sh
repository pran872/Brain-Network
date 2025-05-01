#!/bin/bash
#PBS -N brainnet_exp1
#PBS -l select=1:ncpus=4:mem=8gb:ngpus=1
#PBS -l walltime=02:00:00 
#PBS -j oe
#PBS -o test_job_logs/test.out
#PBS -e test_job_logs/test.err

cd "$PBS_O_WORKDIR"
mkdir -p test_job_logs

RUNFOLDER=${RUNFOLDER:-}
RUNFOLDER_NAME=$(basename "$RUNFOLDER")
CLEAN_JOBID=${PBS_JOBID%%.*}
ECC_RETRIES=${ECC_RETRIES:-0}

eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
conda activate brain-network-env

TRAIN_CMD="python source/test.py --run_default_attacks --run_folder $RUNFOLDER"
ATTEMPT_LOG="test_job_logs/attempt_${CLEAN_JOBID}_ecc${ECC_RETRIES}_${RUNFOLDER_NAME}"

echo "Starting test on $(hostname) at $(date)"
echo "Using run folder: $RUNFOLDER"

$TRAIN_CMD >> "${ATTEMPT_LOG}.out" 2>> "${ATTEMPT_LOG}.err"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Test completed successfully at $(date)" | tee -a "${ATTEMPT_LOG}.out"
else
    if grep -qi "uncorrectable ECC error" "${ATTEMPT_LOG}.err"; then
        echo "Detected ECC Error." | tee -a "${ATTEMPT_LOG}.out"

        if [ "$ECC_RETRIES" -lt 2 ]; then
            NEXT_ECC_RETRIES=$((ECC_RETRIES + 1))
            LOGFILE="test_job_logs/resub_${CLEAN_JOBID}_${NEXT_ECC_RETRIES}_${RUNFOLDER_NAME}.out"

            RESUBMIT_OUTPUT=$(qsub -o "$LOGFILE" -j oe "$0" -v RUNFOLDER=\"$RUNFOLDER\",ECC_RETRIES=$NEXT_ECC_RETRIES 2>&1)
            QSUB_EXIT_CODE=$?

            if [ $QSUB_EXIT_CODE -eq 0 ]; then
                echo "Resubmitted job due to ECC error (ECC_RETRIES=$NEXT_ECC_RETRIES): $RESUBMIT_OUTPUT" | tee -a "${ATTEMPT_LOG}.out"
            else
                echo "Resubmission failed: $RESUBMIT_OUTPUT" | tee -a "${ATTEMPT_LOG}.out"
            fi
        else
            echo "Maximum ECC resubmissions reached. No further resubmits." | tee -a "${ATTEMPT_LOG}.out"
        fi
    else
        echo "Test failed with non-ECC error (code $EXIT_CODE). Not resubmitting." | tee -a "${ATTEMPT_LOG}.out"
    fi
fi

# Organize logs
FINAL_LOG_DIR="test_job_logs/job_${CLEAN_JOBID}_${RUNFOLDER_NAME}"
mkdir -p "$FINAL_LOG_DIR"
mv "${ATTEMPT_LOG}.out" "$FINAL_LOG_DIR/"
mv "${ATTEMPT_LOG}.err" "$FINAL_LOG_DIR/"
mv test_job_logs/test_${CLEAN_JOBID}.out "$FINAL_LOG_DIR/" 2>/dev/null
mv test_job_logs/test_${CLEAN_JOBID}.err "$FINAL_LOG_DIR/" 2>/dev/null

echo "Job cleanup done. Logs stored in $FINAL_LOG_DIR"
