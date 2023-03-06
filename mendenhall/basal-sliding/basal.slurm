#!/bin/bash

#SBATCH --partition=ami100
#SBATCH --job-name=basal
#SBATCH --output=basal.%j.out
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu 

module purge
module load anaconda
conda activate igm

python basal.py