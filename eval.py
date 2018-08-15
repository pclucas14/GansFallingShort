import argparse
import pdb
import numpy as np
import torch
import torch.optim as optim
import tensorboardX
from collections import OrderedDict as OD
import matplotlib; matplotlib.use('Agg')

from tsne import compute_tsne
from tsne_utils import create_matrix_for_tsne

from utils  import * 
from data   import * 
from models import * 
from losses import * 
from args   import * 

args  = get_test_args()

# reproducibility
torch.manual_seed(1)
np.random.seed(1)

# dataset creation
if args.debug: # --> allows for faster iteration when coding 
    dataset_train, word_dict = tokenize(os.path.join(args.data_dir, 'valid.txt'), train=True)
    dataset_test = dataset_train
else: 
    dataset_train, word_dict = tokenize(os.path.join(args.data_dir, 'train.txt'), train=True)
    dataset_test,  word_dict = tokenize(os.path.join(args.data_dir, 'valid.txt'), train=False, word_dict=word_dict)

# fetch one minibatch of data
train_batch = next(minibatch_generator(dataset_train, args, shuffle=False))
test_batch  = next(minibatch_generator(dataset_test,  args, shuffle=False))

# load model that will be evaluated
gen, loaded_epoch = load_model_from_file(args.model_path, epoch=args.model_epoch)
gen.eval()

# Logging
# maybe_create_dir(os.path.join(args.model_path, 'eval/%s_epoch' % loaded_epoch)) # TODO: maybe put in TB directly ?
writer = tensorboardX.SummaryWriter(log_dir=os.path.join(args.model_path, 'TB_tnse{}'.format(args.tsne_log_every)))
writes = 0

if args.lm_path: 
    oracle_lm = load_model_from_file(args.lm_path, epoch=args.lm_epoch)[0]
    oracle_lm.eval()

if args.cuda: 
    gen  = gen.cuda()
    if args.lm_path: oracle_lm = oracle_lm.cuda()

# First experiment : log hidden states for T-SNE plots
MODE = [('train', train_batch, OD(), []), ('test', test_batch, OD(), []), ('free_running', test_batch, OD(), [])]

with torch.no_grad():
    for mode, data, hs_dict, oracle_nlls in MODE: 
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
            
            if args.lm_path: 
                if t > 0: 
                    # query the oracle for NLL of the next word (i.e. use x_t to index p(x_t | x_{i<t})
                    # oracle_nlls += [-1. * oracle_dist.log_prob(input_idx.squeeze()).mean(dim=0).item()]
                    oracle_nll_t = -1. * oracle_dist.log_prob(input_idx.squeeze())
                    oracle_nlls += [remove_pad_tokens(oracle_nll_t, input_idx.squeeze()).item()]

                # feed the current word to the model. 
                input_oracle = oracle_lm.embedding(input_idx)
                output_oracle, hidden_state_oracle = oracle_lm.step(input_oracle, \
                        hidden_state_oracle, t)
                oracle_dist = oracle_lm.output_layer(output_oracle)
                oracle_dist = Categorical(logits=oracle_dist.squeeze(1))
            
            if not teacher_force: 
                dist = gen.output_layer(output)
                input_idx = Categorical(logits=dist.squeeze(1)).sample().unsqueeze(1)

            if (t+1) % args.tsne_log_every == 0: 
                # for lstm we take the hidden state (i.e. h_t of (h_t, c_t))
                hs = hidden_state[0] if isinstance(hidden_state, tuple) else hidden_state
                hs_dict[t] = hs.cpu().data.numpy()

            if (t+1) % args.oracle_nll_log_every == 0 and args.lm_path: 
                p_x_1t = sum(oracle_nlls)
                p_x_t = oracle_nlls[-1]
                #writer.add_scalar('eval/%s_oracle_nll' % mode , p_x_t, t)
                print_and_log_scalar(writer, 'eval/%s_oracle_nll' % mode, p_x_t, t) 
#______________________________________________________________________________________
# from here we should do T-SNE --> the required hidden_states are stored in MODE's `OD`.

timesteps=MODE[0][2].keys()
oracle_nll = MODE[-1][-1]

for t in timesteps:
    X, y = create_matrix_for_tsne(MODE,t)
    distances, image = compute_tsne(X, y, t, args)
    writer.add_image('eval/tsne-plot', image, t)
    for i in range(distances.shape[0]):
        for j in range(i + 1, distances.shape[1]):
            writer.add_scalar('eval/distance_centroids%d-%d' % (i, j), distances[i,j], t)


