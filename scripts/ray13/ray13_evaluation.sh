#!/bin/bash
#SBATCH --partition=carter-compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00
#SBATCH -o ./%x.%A.%a.out
#SBATCH -e ./%x.%A.%a.err
#SBATCH --array=1-1%1
# Usage: sbatch --job-name=ray13_evaluation ray13_evaluation.sh

# Define the ...
models=(setA ST MT Kipoi)
model=${models[$SLURM_ARRAY_TASK_ID-1]}

source activate /cellar/users/aklie/opt/miniconda3/envs/eugene_dev
echo -e "python ray13_evaluation_$model.py"
python ray13_evaluation_$model.py
echo -e "\n\n"