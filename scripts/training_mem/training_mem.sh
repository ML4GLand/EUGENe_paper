#! /usr/bin/bash

#SBATCH -o /cellar/users/dlaub/projects/ML4GLand/EUGENe_paper/scripts/out/%A_%a_%x.out
#SBATCH -e /cellar/users/dlaub/projects/ML4GLand/EUGENe_paper/scripts/err/%A_%a_%x.err
#SBATCH -p carter-compute

#SBATCH -J profile_mem
#SBATCH -a 0-37%8
#SBATCH -c 2
#SBATCH --mem 32G

echo Starting
date

wdir=/cellar/users/dlaub/projects/ML4GLand/EUGENe_paper/scripts/training_mem
memray_dir=${wdir}/memray

stores=( ${wdir}/data/rechunked/*.zarr )
store=${stores[$SLURM_ARRAY_TASK_ID]}

echo Processing $store

name=$(basename $store)
stem=${name%.*}
arr=(${stem//_/ })
n_seq=${arr[1]}
n_seq=${n_seq:2}
seq_length=${arr[2]}
seq_length=${seq_length:2}

echo n_seq=$n_seq
echo seq_length=$seq_length

memray run \
    --native \
    --output ${memray_dir}/memray_n=${n_seq}_l=${seq_length}.bin \
    --force \
    training_mem.py \
    $store

memray stats \
    --json \
    --force \
    --output ${memray_dir}/memray_stats_n=${n_seq}_l=${seq_length}.json \
    ${memray_dir}/memray_n=${n_seq}_l=${seq_length}.bin

if [[ $((n_seq * seq_length )) -le 10000000 ]]; then
    echo "Including run with data loaded into memory"
    memray run \
        --native \
        --output ${memray_dir}/memray_n=${n_seq}_l=${seq_length}_load.bin \
        --force \
        training_mem.py \
        --load \
        $store
    
    memray stats \
        --json \
        --force \
        --output ${memray_dir}/memray_stats_n=${n_seq}_l=${seq_length}_load.json \
        ${memray_dir}/memray_n=${n_seq}_l=${seq_length}_load.bin
fi

echo Finished
date