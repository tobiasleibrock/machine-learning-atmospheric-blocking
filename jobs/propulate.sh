#!/bin/bash
# FILEPATH: blocking/jobs/propulate.sh

#SBATCH --ntasks=8
#SBATCH --time=8:00:00
#SBATCH --job-name=blocking-propulate
#SBATCH --partition=normal
#SBATCH --gres=gpu:full:4
#SBATCH --output="blocking/jobs/propulate-<additional-informations>.out"
#SBATCH --error="blocking/jobs/propulate-<additional-informations>.error"

cd blocking

poetry shell

mpiexec -n 8 ./.venv/bin/python model/propulate_search.py