#!/bin/bash
#PBS -N flex_trial1
#PBS -l select=1:ncpus=2:mem=4gb:ngpus=1
#PBS -l walltime=00:10:00 
#PBS -o logs/testing_$PBS_JOBID.out
#PBS -e logs/testing_$PBS_JOBID.err
#PBS -j oe


cd $PBS_O_WORKDIR
mkdir -p logs

module load tools/prod
module load PyTorch/2.1.2-foss-2023a

export PYTHONUSERBASE=$HOME/.local
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH

python3 -c "import torchvision; print('Torchvision version:', torchvision.__version__)"

echo "Starting training on $(hostname) at $(date)"
python level4_run.py
echo "Finished at $(date)"
