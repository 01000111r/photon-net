#!/bin/bash

#SBATCH --job-name=test-9
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --account=phys036784
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


module --force purge
module load languages/python/3.12.3

cd $HOME/photon-net


source venv2/bin/activate

python reup_python.py