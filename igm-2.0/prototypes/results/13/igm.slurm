#!/bin/bash

#SBATCH --qos=blanca-csdms
#SBATCH --job-name=igm
#SBATCH --output=igm.%j.out
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge
module load anaconda
conda activate igm2

python igm-run.py
