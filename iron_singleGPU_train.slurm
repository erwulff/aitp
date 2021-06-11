#!/bin/sh

# Walltime limit
#SBATCH -t 23:59:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH --constraint=v100
#SBATCH -c 2
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16000

# Job name
#SBATCH -J gpu_train

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Add jobscript to job output
cat $0

source ~/miniconda3/bin/activate torch
which python
python --version

mkdir $TMPDIR/aitp_wip
rsync -ar --exclude=".git" . $TMPDIR/aitp_wip
cd $TMPDIR/aitp_wip
if [ $? -eq 0 ]
then
  echo "Successfully changed directory"
else
  echo "Could not change directory" >&2
  exit 1
fi

python check_cuda.py
python -m scripts.train -c configs/default_config.yml -p default_
rsync -a training_sessions/* $SLURM_SUBMIT_DIR/training_sessions
rm -r training_sessions
