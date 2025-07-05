#!/bin/bash
#SBATCH --qos=normal
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gpus=1
#SBATCH --constraint=GPUMEM80GB
module load mamba
source activate torch
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/acraev/data/conda/envs/torch/pkgs/cuda-toolkit
python run.py --config_file config_run.json --train-only