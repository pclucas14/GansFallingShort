#!/bin/bash
#SBATCH --gres=gpu:1 # request GPU "generic resource" 
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham. 
#SBATCH --mem=50000M # memory per
#SBATCH --time=0-23:00 # time (DD-HH:MM) 
#SBATCH --output=logs/%N-%j.out
#SBATCH --account=rpp-bengioy

source ~/torch/bin/activate 
cd ~/GansFallingShort/real_data_experiments


#python "$@"
python score_models.py --model_path trained_models/news/word/best_CoT_bnll/
