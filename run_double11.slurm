#!/bin/bash

#SBATCH --job-name=r_d_11
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=16G
#SBATCH --account=phys036784
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=short
module --force purge


export CACHE_BASE="$HOME/work/CACHE"
export JOB_CACHE_DIR="${CACHE_BASE}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "${JOB_CACHE_DIR}"
rm -rf    "${JOB_CACHE_DIR}"/*
export PYTHONPYCACHEPREFIX="${JOB_CACHE_DIR}"

cd $HOME/photon-net

source venv2/bin/activate

python run_double11.py 
