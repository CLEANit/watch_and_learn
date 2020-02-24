#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=bumblebee
#SBATCH --gres=gpu:1
source activate tensorflow
module load cudnn/7.0-9.0
python RNN_MODELS.py drop/2SP SPIRAL POTTS1 DSIZE=512 RUNS=0 EPOCHS=15 keep_prob=0.9