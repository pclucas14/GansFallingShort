import numpy as np
import pdb
import os
import time
import sys



BASE_DIR='/home/optimass/scratch/GansFallingShort'
LOAD_GEN_PATH = 'trained_models/news/word/best_mle'

runs = 1

for _ in range(runs):


    ## layers
    num_layers = [1,2]
    p = [0.5, 0.5]
    num_layers = np.random.choice(num_layers, 1, p=p)[0]

    ## dim
    hd = [128, 256, 512, 1024]
    p = [0.15, 0.3, 0.4, 0.15]
    hd  = np.random.choice(hd, 1, p=p)[0]
 
    ## batch_size
    bs = [64, 128, 256, 512, 1024]
    p = [0.2]*5
    bs = np.random.choice(bs, 1, p=p)[0]

    ## seq_len
    seq_len=51

    ## dropout
    disc_vdp = [0.6, 0.5, 0.4, 0.3]
    p = [0.25]*4
    disc_vdp = np.random.choice(disc_vdp, 1, p=p)[0]

    # dlr
    disc_lr = [1e-2, 1e-3, 1e-4, 5e-4]
    p = [0.25]*4
    disc_lr = np.random.choice(disc_lr, 1, p=p)[0]
    
    # fixed 
    mle_epochs = 0
    adv_epochs = disc_epochs = 100
    dti=1
    gti=0
    mti=0
    transfer_weights = 1
        
    
    #_____________________________________________________
    # launch model:
  
    base_dir="%(BASE_DIR)s/news/disc_LY%(num_layers)s_VDDISC%(disc_vdp)s_BS%(bs)s_DLR%(disc_lr)s_HD%(hd)s" % locals() 
    
    
    print(base_dir)

    command="main.py \
        --disc_lr %(disc_lr)s \
        --var_dropout_p_gen %(disc_vdp)s \
        --var_dropout_p_disc %(disc_vdp)s \
        --batch_size %(bs)s \
        --base_dir %(base_dir)s \
        --save_every 0 \
        --load_gen_path %(LOAD_GEN_PATH)s \
        --mle_epochs %(mle_epochs)s \
        --disc_pretrain_epochs %(disc_epochs)s \
        --adv_epochs %(disc_epochs)s \
        -dti %(dti)s \
        -gti %(gti)s \
        -mti %(mti)s \
        --num_layers_gen %(num_layers)s \
        --num_layers_disc %(num_layers)s \
        --hidden_dim_gen %(hd)s \
        --hidden_dim_disc %(hd)s \
        --max_seq_len %(seq_len)s \
        --transfer_weights_after_pretraining %(transfer_weights)s \
        " % locals()
    
    print(command)
    
    command = "{} cc_launch_news.sh {}".format(sys.argv[1], command) 

    os.system(command)
    time.sleep(2)












