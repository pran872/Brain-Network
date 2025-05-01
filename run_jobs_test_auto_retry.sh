#!/bin/bash
#PBS -N brainnet_exp1
#PBS -l select=1:ncpus=4:mem=8gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o test_job_logs/test_$(date +%Y%m%d_%H%M%S).out
#PBS -e test_job_logs/test_$(date +%Y%m%d_%H%M%S).err

cd "$PBS_O_WORKDIR"
mkdir -p test_job_logs

RUNFOLDER=${RUNFOLDER:-}
RUNFOLDER_NAME=$(basename "$RUNFOLDER")
CLEAN_JOBID=${PBS_JOBID%%.*}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
conda activate brain-network-env

TRAIN_CMD="python source/test.py --run_default_attacks --run_folder $RUNFOLDER"
ATTEMPT_LOG="test_job_logs/attempt_${CLEAN_JOBID}_${TIMESTAMP}_${RUNFOLDER_NAME}"

echo "Starting test on $(hostname) at $(date)"
echo "Using run folder: $RUNFOLDER"

$TRAIN_CMD >> "${ATTEMPT_LOG}.out" 2>> "${ATTEMPT_LOG}.err"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Test completed successfully at $(date)" | tee -a "${ATTEMPT_LOG}.out"
else
    echo "Test failed with exit code $EXIT_CODE" | tee -a "${ATTEMPT_LOG}.out"
fi

FINAL_LOG_DIR="test_job_logs/job_${CLEAN_JOBID}_${TIMESTAMP}_${RUNFOLDER_NAME}"
mkdir -p "$FINAL_LOG_DIR"
mv "${ATTEMPT_LOG}.out" "$FINAL_LOG_DIR/"
mv "${ATTEMPT_LOG}.err" "$FINAL_LOG_DIR/"
mv test_job_logs/test_${CLEAN_JOBID}.out "$FINAL_LOG_DIR/" 2>/dev/null
mv test_job_logs/test_${CLEAN_JOBID}.err "$FINAL_LOG_DIR/" 2>/dev/null

echo "Job cleanup done. Logs stored in $FINAL_LOG_DIR"
