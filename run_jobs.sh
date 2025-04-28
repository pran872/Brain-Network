#!/bin/bash
#PBS -N brainnet_exp1
#PBS -l select=1:ncpus=4:mem=8gb:ngpus=1
#PBS -l walltime=02:15:00 
#PBS -o job_logs/train.out
#PBS -e job_logs/train.err
#PBS -j oe

cd $PBS_O_WORKDIR
mkdir -p job_logs

CONFIG=${CONFIG:-configs/config_template.json}
CONFIG_NAME=$(basename "$CONFIG" .json)

cleanup(){
    echo "Clean up"
    mv job_logs/train.out "job_logs/train_${PBS_JOBID}_${CONFIG_NAME}.out"
    mv job_logs/train.err "job_logs/train_${PBS_JOBID}_${CONFIG_NAME}.err"
}
trap cleanup EXIT

eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
conda activate brain-network-env

export OUTPUT_DIR=/rds/general/user/psp20/home/Brain-Network/runs/stanford_dogs/batch_7

echo "Starting training on $(hostname) at $(date)"
echo "Using config: $CONFIG"
python source/simple_cnn.py -c $CONFIG
echo "Finished at $(date)"
