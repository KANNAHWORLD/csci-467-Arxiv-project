#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=03:15:00
#SBATCH --account=robinjia_1265
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-12

module purge
module load gcc/11.3.0
module load python

eval "$(conda shell.bash hook)"
deactivate
source csci467/bin/activate
python3 bert_baseline_lime_args.py --input=input/input_${SLURM_ARRAY_TASK_ID}
