#!/bin/sh

# Walltime limit
#SBATCH -t 1-00:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-task=4
#SBATCH --constraint=a100

# Job name
#SBATCH -J aitp_train

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err


module --force purge; module load modules/2.1.1-20230405
module load slurm cuda/11.8.0 cudnn/8.4.0.27-11.6 nccl/2.14.3-1 openmpi/4.0.7 gcc cmake

# Add jobscript to job output
echo "#################### Job submission script. #############################"
cat $0
echo "################# End of job submission script. #########################"

source /mnt/ceph/users/ewulff/mambaforge/bin/activate aitp
which python
python --version

python check_cuda.py
python -m scripts.train -c $1 -p $2
