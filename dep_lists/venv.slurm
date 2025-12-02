#!/bin/bash

#SBATCH --job-name=venv
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=16G
#SBATCH --account=phys036784
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=test

module --force purge
module load languages/python/3.12.3


cd $HOME/photon-net

python3 -m venv venv2

source venv2/bin/activate

pip3 install -r reqs_no_version.txt