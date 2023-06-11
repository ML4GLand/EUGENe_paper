#!/bin/bash
#SBATCH --partition=carter-compute
#SBATCH --job-name=load_and_loop_sdata
#SBATCH --output=load_and_loop_sdata_%j.out
#SBATCH --error=load_and_loop_sdata_%j.err
#SBATCH --time=14-00:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=4

# Load the required environment
source activate /cellar/users/aklie/opt/miniconda3/envs/ml4gland

# Define the input and output directories
input_file=$1

# Run the Python script with memray
python -m memray run load_and_loop_sdata.py --zarr_file $input_file --load
