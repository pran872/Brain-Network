#!/bin/bash
#PBS -N brainnet_exp1
#PBS -l select=1:ncpus=4:mem=8gb:ngpus=0
#PBS -l walltime=00:15:00 
#PBS -o logs/train_$PBS_JOBID.out
#PBS -e logs/train_$PBS_JOBID.err
#PBS -j oe


cd $PBS_O_WORKDIR
mkdir -p job_logs

eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
conda activate brain-network-env

export OUTPUT_DIR=/rds/general/user/psp20/home/Brain-Network/runs

python source/simple_cnn.py -c config_template.json -v

echo "Starting training on $(hostname) at $(date)"
python level4_run.py
echo "Finished at $(date)"
