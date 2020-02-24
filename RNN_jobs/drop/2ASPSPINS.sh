#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=bumblebee
#SBATCH --gres=gpu:1
source activate tensorflow
module load cudnn/7.0-9.0
python RNN_MODELS.py drop/2ASP SPIRAL SPINS1 hidden_size=378 DSIZE=512 RUNS=0 EPOCHS=30 seed=69 device=/gpu:0 keep_prob=0.9