#!/bin/bash

#SBATCH --partition=ami100
#SBATCH --job-name=basal
#SBATCH --output=basal.%j.out
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu 

module purge
module load anaconda
conda activate igm

for exp in 100 80 60 40 20
do
    cp ./inputs/observation_"${exp}"_pct_conf.nc ./inputs/observation.nc
    python basal.py
    mv *.out convergence.png costs.dat ex.nc geology-optimized.nc igm-run-parameters.txt rms_std.dat strflowctrl.dat tcompoptimize.dat ts.nc volume.dat ./outputs/"${exp}"
    cp ./basal-sliding/outputs/low_thk_confidence/explore-output.ipynb ./outputs/"${exp}"
    rm ./inputs/observation.nc clean.sh
done

