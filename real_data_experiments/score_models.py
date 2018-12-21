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
if not os.path.exists(args.base_dir+'/samples'): os.makedirs(args.base_dir+'/samples')

if args.lm_path: 
    oracle_lm = load_model_from_file(args.lm_path, epoch=args.lm_epoch)[0]
    oracle_lm.eval()

if args.cuda: 
    gen  = gen.cuda()
    if args.lm_path: oracle_lm = oracle_lm.cuda()

MODE = [('free_running', test_batch, OD(), [], [])]


TEMPERATURES = [0.9, 0.95, 1.0, 1.03, 1.06, 1.09, 1.12, 1.15, 1.20,
                1.25, 1.30, 1.35, 1.40, 1.50, 1.60, 1.70, 1.8, 1.9, 2.0, 3.0, 4.0 ]

BEAM_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 

if args.decoder == 'temp': PARAMS = TEMPERATURES
elif args.decoder == 'beam': PARAMS = BEAM_SIZE


_, word_dict = tokenize('{}/train.txt'.format(args.data_dir), train=True)

for param in PARAMS:

    hidden_state_oracle = None
    oracle_nlls = []

    with torch.no_grad():
                
        if args.decoder=='temp':
            sentences = sample_from_model(gen, args.decoder, args.num_samples, alpha=param)

        if args.lm_path:
            for t in range(-1,args.max_seq_len): 
                if t > -1: 
                    # query the oracle for NLL of the next word (i.e. use x_t to index p(x_t | x_{i<t})
                    # oracle_nlls += [-1. * oracle_dist.log_prob(input_idx.squeeze()).mean(dim=0).item()]
                    oracle_nll_t = -1. * oracle_dist.log_prob(sentences[:,[t]].squeeze())
                    oracle_nlls += [remove_pad_tokens(oracle_nll_t, sentences[:,[t]].squeeze()).item()]
                    full_oracle_nll = oracle_nll_t.view(-1,1) if t==0 \
                        else torch.cat((full_oracle_nll,oracle_nll_t.view(-1,1)),1)

                # feed the current word to the model.
                if t==-1: 
                    # first feed SOS_TOKEN=2 
                    input_oracle = oracle_lm.embedding(torch.ones_like(sentences[:,[t]])*2)
                else:
                    input_oracle = oracle_lm.embedding(sentences[:,[t]])
                output_oracle, hidden_state_oracle = oracle_lm.step(input_oracle, \
                        hidden_state_oracle, t)
                oracle_dist = oracle_lm.output_layer(output_oracle)
                oracle_dist = Categorical(logits=oracle_dist.squeeze(1))

    # print most/less likely sequences
    seq_len = (sentences != 0).sum(1)
    tot_oracle_nll = full_oracle_nll.sum(1)
    avg_oracle_nll = tot_oracle_nll.cpu().numpy() / seq_len.cpu().numpy()

    sentences_ = id_to_words(sentences.cpu().data.numpy(), word_dict)
    sorted_idx = np.argsort(avg_oracle_nll)

    if args.character_level: sentences_ = remove_sep_spaces(sentences_)
    
    print("most likely sentences under oracle: \n")
    for i in range(3):
        print(sentences_[sorted_idx[i]])
        print("nll oracle: {:.4f}".format(avg_oracle_nll[sorted_idx[i]]))
    
    print("least likely sentences under oracle: \n ")
    for i in range(1,3):
        print(sentences_[sorted_idx[-i]])
        print("nll oracle: {:.4f}".format(avg_oracle_nll[sorted_idx[-i]]))

    print('some samples \n')
    for i in range(1,3):
        print(sentences_[-i])
        print("nll oracle: {:.4f}".format(avg_oracle_nll[-i]))

    if args.decoder=='temp': param = int(param*100)
    
    ######  LM score   ######
    lm_score = np.mean(avg_oracle_nll)
    print_and_log_scalar(writer, 'eval/{}/lm_score'.format(args.decoder), lm_score, param)

    ##### RLM SCORE ######

    # save the generated sequences somewhere 
    rlm_dir = os.path.join(args.model_path, "{}_rlm_alpha{}".format(args.decoder, param))
    print_and_save_samples(sentences, 
           word_dict, rlm_dir, for_rlm=True, split='train', breakdown=10)
    
    rlm_score = main(rlm=True, rlm_dir=rlm_dir)
    
    print_and_log_scalar(writer, 'eval/{}/rlm_score'.format(args.decoder), rlm_score, param)
    
    # delete the dataset
    command="rm {}".format(os.path.join(rlm_dir,'train.txt'))
    print(command)
    os.system(command) 
 

