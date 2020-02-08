import argparse
import pdb
import numpy as np
import torch
import torch.utils.data
import tensorboardX
from collections import OrderedDict as OD
from PIL import Image
import matplotlib; matplotlib.use('Agg')

import __init__

from common.utils       import * 
from common.data        import * 
from common.models      import * 
from common.losses      import * 
from common.args        import * 
from common.eval_decode import * 
from main   import main

args  = get_test_args()

# reproducibility
torch.manual_seed(2)
np.random.seed(2)

# dataset creation
dataset_train, word_dict = tokenize(os.path.join(args.data_dir, 'train.txt'), \
        train=True, char_level=args.character_level)
dataset_test,  word_dict = tokenize(os.path.join(args.data_dir, 'valid.txt'), \
        train=False, word_dict=word_dict, char_level=args.character_level)

# fetch one minibatch of data
train_batch = next(minibatch_generator(dataset_train, args, shuffle=False))
test_batch  = next(minibatch_generator(dataset_test,  args, shuffle=False))

# load model that will be evaluated
gen, loaded_epoch = load_model_from_file(args.model_path, epoch=args.model_epoch)
gen.eval()

# Logging
writer = tensorboardX.SummaryWriter(log_dir=os.path.join(args.model_path, \
        'TB'))
writes = 0

# create dir:
#if not os.path.exists(args.model_path+'/samples'): os.makedirs(args.base_dir+'/samples')

if args.lm_path: 
    oracle_lm = load_model_from_file(args.lm_path, epoch=args.lm_epoch)[0]
    oracle_lm.eval()

if args.cuda: 
    gen  = gen.cuda()
    if args.lm_path: oracle_lm = oracle_lm.cuda()

MODE = [('free_running', test_batch, OD(), [], [])]

TEMPERATURES = [0.9, 0.95, 1.0, 1.05, 1.1,  1.15, 1.20,
                1.25, 1.30, 1.35, 1.40, 1.50, 1.60, 1.70, 1.8, 1.9, 2.0, 3.0, 4.0 ]
BEAM_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
TOP_K = [10, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 500] 
WTOP_K = [20, 50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000, 5500] 
GEN_THRES = [ 2.5, 2.75, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5, 5.25, 5.50,
                 5.75, 6.0, 6.5, 7, 8, 9, 10 ] 
DISC_THRES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.48, 0.5, 0.52, 0.55, 
                   0.6, 0.7, 0.8, 0.9 ] 

#######
GEN_THRES = [ 2., 2.25, 3., 3.62, 3.87 ] 
TEMPERATURES = [1.01, 1.02, 1.03, 1.04, 1.06, 1.07, 1.08, 1.09, 2.25, 2.5, 2.75, 3.25, 3.5, 3.75]
DISC_THRES = [0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98] 


if   args.decoder == 'temp':          PARAMS = TEMPERATURES
elif args.decoder == 'beam':          PARAMS = BEAM_SIZES
elif args.decoder == 'topk':          PARAMS = TOP_K
elif args.decoder == 'weighted_topk': PARAMS = WTOP_K
elif args.decoder == 'gen_ll':        PARAMS = GEN_THRES
elif args.decoder == 'disc_ll':       PARAMS = DISC_THRES

_, word_dict = tokenize('{}/train.txt'.format(args.data_dir), train=True)


for param in PARAMS:
            
    if   args.decoder=='temp':          kwargs = {'alpha':param}
    elif args.decoder=='beam':          kwargs = {'beam_size':param}
    elif args.decoder=='topk':          kwargs = {'k':param}
    elif args.decoder=='weighted_topk': kwargs = {'k':param}
    elif args.decoder=='gen_ll':        kwargs = {'threshold':param}
    elif args.decoder=='disc_ll':       kwargs = {'threshold':param}

    ## trying this:
    kwargs['remove_duplicates'] = True
    
    kwargs['model_path'] = args.model_path
           
    sentences = sample_from_model(gen, args.decoder, args.num_samples, **kwargs)
   
    ### LM Score:
    lm_score = []
    chunks = 10.
    for i in range(int(chunks)):
        idx = range(int(args.num_samples/chunks*(i)), int(args.num_samples/chunks*(i+1)))  
        lm_score += [compute_lm_score(sentences[idx], 
                                      oracle_lm, 
                                      verbose=(i==chunks-1),
                                      word_dict=word_dict,
                                      args=args)]
    lm_score = np.mean(lm_score)
     
    if args.decoder=='temp': param = int(param*100)
    if args.decoder=='disc_ll': param = int(param*100)
    if args.decoder=='gen_ll': param = int(param*100)
    
    print_and_log_scalar(writer, 'eval/{}/lm_score'.format(args.decoder), lm_score, param)
    
    ### RLM SCORE:
    
    # save the generated sequences somewhere 
    rlm_dir = os.path.join(args.model_path, "{}_rlm_alpha{}".format(args.decoder, param))
    print_and_save_samples(sentences, 
           word_dict, rlm_dir, for_rlm=True, split='train', breakdown=10)
    
    rlm_score = main(rlm=True, rlm_dir=rlm_dir)
    
    print_and_log_scalar(writer, 'eval/{}/rlm_score'.format(args.decoder), rlm_score, param)
    
    # delete the dataset
    command="rm -rf {}".format(rlm_dir)
    print(command)
    os.system(command) 


