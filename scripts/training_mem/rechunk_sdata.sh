#! /usr/bin/bash

#SBATCH -o /cellar/users/dlaub/projects/ML4GLand/EUGENe_paper/scripts/out/%A_%a_%x.out
#SBATCH -e /cellar/users/dlaub/projects/ML4GLand/EUGENe_paper/scripts/err/%A_%a_%x.err
#SBATCH -p carter-compute

#SBATCH -J rechunk
#SBATCH -a 0-37%8
#SBATCH -c 2
#SBATCH --mem 48G

wdir=/cellar/users/dlaub/projects/ML4GLand/EUGENe_paper/scripts/training_mem

stores=( ${wdir}/data/*.zarr )
store=${stores[$SLURM_ARRAY_TASK_ID]}

python ${wdir}/rechunk_sdata.py $store
