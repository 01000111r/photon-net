#!/bin/bash

#SBATCH --job-name=init-phases-vals
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --account=phys036784
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


module --force purge
module load languages/python/3.12.3

export CACHE_BASE="$HOME/work/CACHE"
export JOB_CACHE_DIR="${CACHE_BASE}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "${JOB_CACHE_DIR}"
rm -rf    "${JOB_CACHE_DIR}"/*
export PYTHONPYCACHEPREFIX="${JOB_CACHE_DIR}"

cd $HOME/photon-net

source venv2/bin/activate

python init_phases.py