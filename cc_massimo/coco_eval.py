import numpy as np
import pdb
import os
import time
import sys


DIR = 'trained_models/coco/word/'

oracle_path = DIR + 'best_mle'

MODELS = ['best_mle','gan_lm_beta0_mti0', 'gan_lm_beta0_mti1', 'gan_lm_beta1_mti0', 'gan_mix_beta0_mti0',
          'gan_mix_beta1_mti0','gan_mix_beta0_mti1']

MODELS = ['gan_lm_beta0_mti0', 'gan_lm_beta0_mti1', 'gan_lm_beta1_mti0', 'gan_mix_beta0_mti0',
          'gan_mix_beta1_mti0','gan_mix_beta0_mti1']



for model in MODELS:

    
    model_path = DIR + model

    command="score_models.py --model_path {} --lm_path {} --data_dir data/coco".format(model_path, oracle_path)
    print(command)
    
    command = "{} cc_launch_eval.sh {}".format(sys.argv[1], command) 


    os.system(command)
    time.sleep(2)












