import numpy as np
import pdb
import os
import time
import sys



BASE_DIR='/home/optimass/scratch/OnExposureBias/char/'

runs = 40

for _ in range(runs):

    ## MLE or GAN
    # for now just MLE
    loss = ['mle','gan']
    p = [1.0, 0.0]
    loss = np.random.choice(loss, 1, p=p)[0]

    ## layers
    num_layers = [1,2,3,4]
    p = [0.25]*4
    num_layers = np.random.choice(num_layers, 1, p=p)[0]

    ## dim
    hd = [32, 64, 128, 256, 512]
    p = [0.2]*5
    hd  = np.random.choice(hd, 1, p=p)[0]
    
    ## batch_size
    bs = [256, 512, 1024]
    p = [0.33, 0.33, 0.34]
    bs = np.random.choice(bs, 1, p=p)[0]

    ## glr
    gen_lr = [1e-3, 5e-4, 1e-4]
    p = [0.25, 0.5, 0.25]
    gen_lr = np.random.choice(gen_lr, 1, p=p)[0]

    ## seq_len
    #seq_len = [20, 25, 30, 35]
    #p = [0.25]*4
    #seq_len = np.random.choice(seq_len, 1, p=p)[0]
    #seq_len=51
    seq_len=300

    ## dropout
    gen_vdp = [0.6, 0.5, 0.4, 0.3]
    p = [0.25]*4
    gen_vdp = np.random.choice(gen_vdp, 1, p=p)[0]

    # fixed if MLE:
    mle_epochs = 300
    adv_epochs = 0
    disc_epochs= 0
    dti=0
    gti=0
    mti=0
    disc_lr=0
    disc_vdp=0
    beta=0
    ats=1

    ## Gan related stuff:
    # TODO(MASSIMO):
    if loss=='gan':

        # mle pretrain
        mle_epochs = [0, 10, 40, 80]
        p = [0.4, 0.2, 0.2, 0.2]
        mle_epochs = np.random.choice(mle_epochs, 1, p=p)[0]
            
        # disc pretrain
        if mle_epochs > 0:
            disc_epochs = [ 1, mle_epochs/5, mle_epochs/2, mle_epochs ]
            p = [0.4, 0.2, 0.2, 0.2]
            disc_epochs = int(np.random.choice(disc_epochs, 1, p=p)[0])
        else:
            disc_epochs = 0

        adv_epochs=int(300-mle_epochs)

        # dti
        dti = [1,  5, 10, 20]
        p = [0.25] *4
        dti = np.random.choice(dti, 1, p=p)[0]

        # gti
        gti=1

        # mti 
        mti = [0, 1]
        p = [0.6, 0.4]
        mti = np.random.choice(mti, 1, p=p)[0]

        # dlr
        disc_lr = [5e-4, 1e-4, 5e-5]
        p = [0.25, 0.5, 0.25]
        disc_lr = np.random.choice(disc_lr, 1, p=p)[0]
        
        # dvd
        disc_vdp = [0.5, 0.4, 0.3, 0.2]
        p = [0.25]*4
        disc_vdp = np.random.choice(disc_vdp, 1, p=p)[0]
     
        # betas
        beta = [0.0, 0.1, 1.0, 2.0]
        p = [0.25]*4
        beta = np.random.choice(beta, 1, p=p)[0]
        beta=0

        # alpha test
        ats = [1.0, 1.2, 1.4, 1.6] 
        p = [0.55, 0.15, 0.15, 0.15 ]
        ats = np.random.choice(ats, 1, p=p)[0]
        ats=1

        # use conv disc
        #mode = 0 
    
    #_____________________________________________________
    # launch model:
  
    if loss=='mle':
        base_dir="%(BASE_DIR)s/news/mle_LY%(num_layers)s_VDGEN%(gen_vdp)s_BS%(bs)s_GLR%(gen_lr)s_HD%(hd)s_SQ%(seq_len)s" % locals() 
    
    elif loss=='gan':
        base_dir="%(BASE_DIR)s/news/gan_VDGEN%(gen_vdp)s_VDDISC%(disc_vdp)s_BS%(bs)s_GLR%(gen_lr)s_DLR%(disc_lr)s_MLE%(mle_epochs)s_DE%(disc_epochs)s_DTI%(dti)s_GTI%(gti)s_MTI%(mti)s_HD%(hd)s_SQ%(seq_len)s_ats%(ats)s_beta%(beta)s" % locals() 
    
    print(base_dir)

    command="main.py \
        --gen_lr %(gen_lr)s \
        --disc_lr %(disc_lr)s \
        --var_dropout_p_gen %(gen_vdp)s \
        --var_dropout_p_disc %(disc_vdp)s \
        --batch_size %(bs)s \
        --base_dir %(base_dir)s \
        --mle_epochs %(mle_epochs)s \
        --disc_pretrain_epochs %(disc_epochs)s \
        --adv_epochs %(adv_epochs)s \
        -dti %(dti)s \
        -gti %(gti)s \
        -mti %(mti)s \
        --num_layers_gen %(num_layers)s \
        --num_layers_disc %(num_layers)s \
        --hidden_dim_gen %(hd)s \
        --hidden_dim_disc %(hd)s \
        --max_seq_len %(seq_len)s \
        --alpha_test %(ats)s \
        --beta %(beta)s \
        --character_level " % locals()
    print(command)
    
    command = "{} cc_launch_news.sh {}".format(sys.argv[1], command) 


    os.system(command)
    time.sleep(2)












