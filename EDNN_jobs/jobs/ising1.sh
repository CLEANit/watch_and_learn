#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=bumblebee
#SBATCH --gres=gpu:1
source activate tensorflow
module load cudnn/7.0-9.0
python EDNN.py ISING1 15 30 -l