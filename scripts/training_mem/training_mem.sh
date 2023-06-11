#! /usr/bin/bash

#SBATCH -o /cellar/users/dlaub/projects/ML4GLand/EUGENe_paper/scripts/out/%A_%a_%x.out
#SBATCH -e /cellar/users/dlaub/projects/ML4GLand/EUGENe_paper/scripts/err/%A_%a_%x.err
#SBATCH -p carter-compute

#SBATCH -c 1

wdir=/cellar/users/dlaub/projects/ML4GLand/EUGENe_paper/scripts/training_mem
data=${wdir}/data/sdata-$1-$2.zarr

if []

memray run \
    --follow-fork \
    --native \
    --output ${wdir}/memray-$1-$2.bin \
    training_mem.py \
    $1 $2 \
    ${wdir}/sdata-$1-$2.zarr \
    $SLURM_MEM_PER_CPU