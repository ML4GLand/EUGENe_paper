#! /usr/bin/bash

#SBATCH -o /cellar/users/dlaub/projects/ML4GLand/EUGENe_paper/scripts/out/%A_%a_%x.out
#SBATCH -e /cellar/users/dlaub/projects/ML4GLand/EUGENe_paper/scripts/err/%A_%a_%x.err
#SBATCH -p carter-compute

#SBATCH -c 2
#SBATCH --mem 8G

wdir=/cellar/users/dlaub/projects/ML4GLand/EUGENe_paper/scripts/training_mem
data_dir=${wdir}/data

n_seqs=( 100 1000 10000 100000 1000000 1000000 1000000 )
seq_lengths=( 100 1000 5000 10000 100000 1000000 10000000 )

for n_seq in $n_seqs; do
    for seq_length in $seq_lengths; do
        output=${data_dir}/sdata_n=${n_seq}_l=${seq_length}.zarr
        python ${wdir}/generate_sdata.py $n_seq $seq_length $output $SLURM_MEM_PER_CPU