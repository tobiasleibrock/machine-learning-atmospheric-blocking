#!/bin/bash
# FILEPATH: blocking/jobs/model.sh

#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --job-name=blocking-model
#SBATCH --partition=normal
#SBATCH --gres=gpu:full:1
#SBATCH --output="blocking/jobs/model-<additional-informations>.out"
#SBATCH --error="blocking/jobs/model-<additional-informations>.error"

cd blocking

poetry shell

mpiexec -n 1 /home/scc/mw8007/blocking/.venv/bin/python model/model.py