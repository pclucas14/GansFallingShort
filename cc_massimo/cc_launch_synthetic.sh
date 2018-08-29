#!/bin/bash
#SBATCH --gres=gpu:1 # request GPU "generic resource" 
#SBATCH --cpus-per-task=2 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham. 
#SBATCH --mem=10000M # memory per
#SBATCH --time=0-03:00 # time (DD-HH:MM) 
#SBATCH --output=logs/%N-%j.out
#SBATCH --account=rrg-dprecup

source ~/pytorch/bin/activate 
cd ~/OnExposureBias/
python "$@"
