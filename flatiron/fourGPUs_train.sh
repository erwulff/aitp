#!/bin/sh

# Walltime limit
#SBATCH -t 1-00:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-task=4
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=64

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

num_gpus=$SLURM_GPUS_PER_TASK
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK))  # OpenMPI threads per process

source /mnt/ceph/users/ewulff/mambaforge/bin/activate aitp
echo "Python in use:"
which python
python --version

python check_cuda.py
torchrun --standalone --nproc_per_node=$num_gpus -m scripts.train -c $1 -p $2
