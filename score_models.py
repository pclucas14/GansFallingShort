import argparse
import pdb
import numpy as np
import torch
import torch.utils.data
import tensorboardX
from collections import OrderedDict as OD
from PIL import Image
import matplotlib; matplotlib.use('Agg')

from tsne import compute_tsne
from tsne_utils import create_matrix_for_tsne

from utils  import * 
from data   import * 
from models import * 
from losses import * 
from args   import * 
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
gen.args.alpha_test = args.alpha_test
gen.eval()
print('switching the temperature to {}'.format(gen.args.alpha_test))

# Logging
writer = tensorboardX.SummaryWriter(log_dir=os.path.join(args.model_path, \
        'TB'))
writes = 0

if args.lm_path: 
    oracle_lm = load_model_from_file(args.lm_path, epoch=args.lm_epoch)[0]
    oracle_lm.eval()

if args.cuda: 
    gen  = gen.cuda()
    if args.lm_path: oracle_lm = oracle_lm.cuda()

MODE = [('free_running', test_batch, OD(), [], [])]

TEMPERATURES = [0.9, 0.95, 1.0, 1.03, 1.06, 1.09, 1.12, 1.15, 1.20,
                1.25, 1.30, 1.35, 1.40, 1.50, 1.60, 1.70, 1.8, 1.9, 2.0]


for alpha in TEMPERATURES:

    with torch.no_grad():
        for mode, data, hs_dict, oracle_nlls, embeddings in MODE: 
            input, _, _ = data

            # here we basically expose the model's forward pass to fetch the hidden states efficiently
            teacher_force = mode != 'free_running'
            print('teacher forcing : {}'.format(teacher_force))
            hidden_state, hidden_state_oracle = None, None

            for t in range(args.tsne_max_t):
                if teacher_force or t == 0: 
                    input_idx = input[:, [t]]

                input_t = gen.embedding(input_idx)
                output, hidden_state = gen.step(input_t, hidden_state, t)
                embeddings += [input_t.cpu().data.numpy()]
                
                if args.lm_path: 
                    if t > 0: 
                        # query the oracle for NLL of the next word (i.e. use x_t to index p(x_t | x_{i<t})
                        # oracle_nlls += [-1. * oracle_dist.log_prob(input_idx.squeeze()).mean(dim=0).item()]
                        oracle_nll_t = -1. * oracle_dist.log_prob(input_idx.squeeze())
                        oracle_nlls += [remove_pad_tokens(oracle_nll_t, input_idx.squeeze()).item()]
                        full_oracle_nll = oracle_nll_t.view(-1,1) if t==1 \
                            else torch.cat((full_oracle_nll,oracle_nll_t.view(-1,1)),1)

                    # feed the current word to the model. 
                    input_oracle = oracle_lm.embedding(input_idx)
                    output_oracle, hidden_state_oracle = oracle_lm.step(input_oracle, \
                            hidden_state_oracle, t)
                    oracle_dist = oracle_lm.output_layer(output_oracle)
                    oracle_dist = Categorical(logits=oracle_dist.squeeze(1))
               
                if not teacher_force: 
                    dist = gen.output_layer(output)
                    dist *= gen.args.alpha_test
                    input_idx = Categorical(logits=dist.squeeze(1)).sample().unsqueeze(1)
                    fake_sentences = input_idx if t==0 else torch.cat((fake_sentences,input_idx), 1)

                if (t+1) % args.tsne_log_every == 0: 
                    # for lstm we take the hidden state (i.e. h_t of (h_t, c_t))
                    hs = hidden_state[0] if isinstance(hidden_state, tuple) else hidden_state
                    hs_dict[t] = hs.cpu().data.numpy()

                if (t+1) % args.oracle_nll_log_every == 0 and args.lm_path and t > 0: 
                    p_x_1t = sum(oracle_nlls)
                    p_x_t = oracle_nlls[-1]

            # print most/less likely sequences
            seq = input[:,1:] if teacher_force else fake_sentences
            seq_len = (seq != 0).sum(1)
            tot_oracle_nll = full_oracle_nll.sum(1)
            avg_oracle_nll = tot_oracle_nll.cpu().numpy() / seq_len.cpu().numpy()

            sentences = id_to_words(seq.cpu().data.numpy(), word_dict)
            sorted_idx = np.argsort(avg_oracle_nll)
        
            if args.character_level: sentences = remove_sep_spaces(sentences)
            
            print("most likely sentences under oracle: \n")
            for i in range(3):
                print(sentences[sorted_idx[i]])
                print("nll oracle: {:.4f}".format(avg_oracle_nll[sorted_idx[i]]))
            
            print("least likely sentences under oracle: \n ")
            for i in range(1,3):
                print(sentences[sorted_idx[-i]])
                print("nll oracle: {:.4f}".format(avg_oracle_nll[sorted_idx[-i]]))

            print('some samples \n')
            for i in range(1,3):
                print(sentences[-i])
                print("nll oracle: {:.4f}".format(avg_oracle_nll[-i]))

    ######  LM score   ######
    lm_score = np.mean(avg_oracle_nll)
    print_and_log_scalar(writer, 'eval/lm_score', lm_score, int(alpha*100))

    ##### RLM SCORE ######

    # save the generated sequences somewhere 
    rlm_dir = os.path.join(args.model_path,"rlm_alpha{}".format(alpha))
    print_and_save_samples(fake_sentences, 
            word_dict, rlm_dir, for_rlm=True, split='train', breakdown=10)
    
    rlm_score = main(rlm=True, rlm_dir=rlm_dir)
    
    print_and_log_scalar(writer, 'eval/rlm_score', rlm_score, int(alpha*100))
    
    # delete the dataset
    command="rm {}".format(os.path.join(rlm_dir,'train.txt'))
    print(command)
    os.system(command) 
 

# ----------------------------------------------------------------------------
# Evaluate quality/diversity tradeoff in GAN and MLE w/ Temperature Control
# ----------------------------------------------------------------------------

exit()

"""" run the Reverse LM score """
if args.run_rlm:

    # save the generated sequences somewhere 
    rlm_base_dir = os.path.join(args.model_path,"rlm_alpha{}".format(gen.args.alpha_test))
    print_and_save_samples(fake_sentences, 
            word_dict, rlm_base_dir, for_rlm=True, split='train', breakdown=10)

    rlm_log_dir = os.path.join(args.model_path,"TB_alpha{}".format(gen.args.alpha_test))
    rlm_tb = 'eval/rlm_score'

    # run main.py on the generated dataset
    command="python main.py --setup rlm   \
                            --base_dir %s \
                            --data_dir %s \
                            --rlm_log_dir %s \
                            --rlm_tb %s" % (rlm_base_dir, args.data_dir,
                                            rlm_log_dir, rlm_tb)
    print(command)
    os.system(command) 
   
    # delete the dataset
    command="rm {}".format(os.path.join(rlm_base_dir,'train.txt'))
    print(command)
    os.system(command) 
            

""" run LM score on sentences completion """
### TODO() make 100% sure there is no bug
if args.run_sc:

    with torch.no_grad():
        input, _, _ = test_batch

        for t in range(args.tsne_max_t):

            teacher_force = True if t<args.breakpoint else False

            if teacher_force or t == 0: 
                input_idx = input[:, [t]]

            input_t = gen.embedding(input_idx)
            output, hidden_state = gen.step(input_t, hidden_state, t)
            
            if t >= args.breakpoint: 
                # query the oracle for NLL of the next word (i.e. use x_t to index p(x_t | x_{i<t})
                # oracle_nlls += [-1. * oracle_dist.log_prob(input_idx.squeeze()).mean(dim=0).item()]
                oracle_nll_t = -1. * oracle_dist.log_prob(input_idx.squeeze())
                oracle_nlls += [remove_pad_tokens(oracle_nll_t, input_idx.squeeze()).item()]
                full_oracle_nll = oracle_nll_t.view(-1,1) if t==1 \
                        else torch.cat((full_oracle_nll,oracle_nll_t.view(-1,1)),1)

            # feed the current word to the model. 
            input_oracle = oracle_lm.embedding(input_idx)
            output_oracle, hidden_state_oracle = oracle_lm.step(input_oracle, \
                    hidden_state_oracle, t)
            oracle_dist = oracle_lm.output_layer(output_oracle)
            oracle_dist = Categorical(logits=oracle_dist.squeeze(1))
           

            if not teacher_force: 
                dist = gen.output_layer(output)
                dist *= gen.args.alpha_test
                input_idx = Categorical(logits=dist.squeeze(1)).sample().unsqueeze(1)
            
            # this should work but make sure it does:
            fake_sentences = input_idx if t==0 else torch.cat((fake_sentences,input_idx), 1)


        p_x_bt = sum(oracle_nlls)

        # print most/less likely sequences
        seq = fake_sentences
        seq_len = (seq != 0).sum(1)
        tot_oracle_nll = full_oracle_nll.sum(1)
        avg_oracle_nll = tot_oracle_nll.cpu().numpy() / seq_len.cpu().numpy()

        sentences = id_to_words(seq.cpu().data.numpy(), word_dict)
        sorted_idx = np.argsort(avg_oracle_nll)

        if args.character_level: sentences = remove_sep_spaces(sentences)
        
        print("most likely sentences under oracle:")
        for i in range(10):
            print(sentences[sorted_idx[i]])
            print("nll oracle: {:.4f}".format(avg_oracle_nll[sorted_idx[i]]))
        
        print("least likely sentences under oracle:")
        for i in range(1,11):
            print(sentences[sorted_idx[-i]])
            print("nll oracle: {:.4f}".format(avg_oracle_nll[sorted_idx[-i]]))

        # store LM score
        lm_score = np.mean(avg_oracle_nll)
        print_and_log_scalar(writer, 'eval/completion_lm_score_b{}'.format(args.breakpoint), lm_score, 0)


    """ run reverse LM score on the sentence completed dataset """
    # save the generated sequences somewhere 
    rlm_base_dir = os.path.join(args.model_path,"scrlm_alpha{}".format(gen.args.alpha_test))
    print_and_save_samples(fake_sentences, 
            word_dict, rlm_base_dir, for_rlm=True, split='train', breakdown=10)

    pdb.set_trace()
    # run main.py on the generated dataset
    command="python main.py --setup rlm   \
                            --base_dir %s \
                            --data_dir %s \
                            --rlm_log_dir %s \
                            --rlm_tb %s" % (rlm_base_dir, args.data_dir, rlm_log_dir, rlm_tb)

    print(command)
    os.system(command) 
   
    # delete the dataset
    command="rm {}".format(os.path.join(rlm_base_dir,'train.txt'))
    print(command)
    os.system(command) 



