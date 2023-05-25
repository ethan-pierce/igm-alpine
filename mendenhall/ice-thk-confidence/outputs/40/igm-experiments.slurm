#!/bin/bash

#SBATCH --partition=ami100
#SBATCH --job-name=igm40
#SBATCH --output=IGM.%j.out
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu 

module purge
module load anaconda
conda activate igm

python igm-experiments.py
