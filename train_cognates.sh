#!/bin/bash
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --constraint=GPUMEM80GB
module load mamba
source activate torch
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/acraev/data/conda/envs/torch/pkgs/cuda-toolkit
python lexibank_train.py
