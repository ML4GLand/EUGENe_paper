#!/bin/bash
#SBATCH --partition=carter-compute
#SBATCH --job-name=generate_random_seqs_reasonable_lengths
#SBATCH --output=generate_random_seqs_reasonable_lengths_%j.out
#SBATCH --error=generate_random_seqs_reasonable_lengths_%j.err
#SBATCH --time=14-00:00:00
#SBATCH --mem=700G
#SBATCH --cpus-per-task=4

# Load the required environment
source activate /cellar/users/aklie/opt/miniconda3/envs/ml4gland

# Define the input and output directories
output_dir="/cellar/users/aklie/data/eugene/revision/memory_analysis"

# Define the sequence lengths
num_seqs="10000"
seq_lens="10000000"

# Run the Python script
python generate_random_seqs.py $output_dir --seq_lens $seq_lens --num_seqs $num_seqs
