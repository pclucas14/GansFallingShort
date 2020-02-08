import numpy as np
import pdb
import os
import time
import sys


BASE_DIR='/home/optimass/scratch/GansFallingShort/best_mle'
#BASE_DIR='/home/optimass/GansFallingShort/real_data_experiments/trained_models/news/word/best_mle'

DECODERS=['temp', 'beam', 'topk', 'weighted_topk','gen_ll']
DECODERS=['disc_ll']
DECODERS=['gen_ll', 'temp']

for decoder in DECODERS:

    command="score_models.py \
        --decoder %(decoder)s \
        --model_path %(BASE_DIR)s " % locals()
    
    print(command)
    
    command = "{} cc_launch_long.sh {}".format(sys.argv[1], command) 

    os.system(command)
    time.sleep(2)












