#!/bin/bash

#SBATCH --partition=aa100
#SBATCH --job-name=igm
#SBATCH --output=igm.%j.out
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu 

module purge
module load anaconda
conda activate igm

python basal.py
