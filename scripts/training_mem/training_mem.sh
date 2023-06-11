#! /usr/bin/bash

#SBATCH -o /cellar/users/dlaub/projects/ML4GLand/EUGENe_paper/scripts/out/%A_%a_%x.out
#SBATCH -e /cellar/users/dlaub/projects/ML4GLand/EUGENe_paper/scripts/err/%A_%a_%x.err
#SBATCH -p carter-compute

#SBATCH -a 0-37%8
#SBATCH -c 2
#SBATCH --mem-per-cpu 8G

echo Starting
date

wdir=/cellar/users/dlaub/projects/ML4GLand/EUGENe_paper/scripts/training_mem

stores=( ${wdir}/data/*.zarr )
store=${stores[$SLURM_ARRAY_TASK_ID]}

echo Processing $store

name=$(basename $store)
stem=${name%.*}
arr=(${stem//_/ })
n_seq=${arr[1]}
n_seq=${n_seq:2}
seq_length=${arr[2]}
seq_length=${seq_length:2}

memray run \
    --follow-fork \
    --native \
    --output ${wdir}/memray_n=${n_seq}_l=${seq_length}.bin \
    training_mem.py \
    $store

if [[ $((n_seq * seq_length )) -le 10000000 ]]; then
    echo "Including run with data loaded into memory"
    memray run \
        --follow-fork \
        --native \
        --output ${wdir}/memray_n=${n_seq}_l=${seq_length}_load.bin \
        training_mem.py \
        --load \
        $store
fi

echo Finished
date