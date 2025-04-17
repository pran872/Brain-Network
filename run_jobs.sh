#!/bin/bash
#PBS -N brainnet_exp1
#PBS -l select=1:ncpus=4:mem=8gb:ngpus=1
#PBS -l walltime=45:00:00 
#PBS -o job_logs/train.out
#PBS -e job_logs/train.err
#PBS -j oe

cd $PBS_O_WORKDIR
mkdir -p job_logs
mv job_logs/train.out job_logs/train_${PBS_JOBID}.out
mv job_logs/train.err job_logs/train_${PBS_JOBID}.err

eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
conda activate brain-network-env

export OUTPUT_DIR=/rds/general/user/psp20/home/Brain-Network/runs

echo "Starting training on $(hostname) at $(date)"
python source/simple_cnn.py -c config_template.json -v
echo "Finished at $(date)"
