import argparse
import pdb
import numpy as np
import torch
import torch.optim as optim
import tensorboardX
from collections import OrderedDict as OD

from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform
from matplotlib.patches import Ellipse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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
writer = tensorboardX.SummaryWriter(log_dir=os.path.join(args.model_path, 'TB'))
writes = 0

if args.cuda: 
    gen  = gen.cuda()

# First experiment : log hidden states for T-SNE plots
MODE = [('train', train_batch, OD()), ('test', test_batch, OD()), ('free_running', test_batch, OD())]

with torch.no_grad():
    for mode, data, hs_dict in MODE: 
        input, _, _ = data

        # here we basically expose the model's forward pass to fetch the hidden states efficiently
        teacher_force = not mode == 'free_running'
        hidden_state = None

        for t in range(args.tsne_max_t):
            if teacher_force or t == 0: 
                input_idx = input[:, [t]]

            input_t = gen.embedding(input_idx)
            output, hidden_state = gen.step(input_t, hidden_state, t)
            
            if not teacher_force: 
                dist = gen.output_layer(output)
                input_idx = Categorical(logits=dist.squeeze(1)).sample().unsqueeze(1)

            if (t+1) % args.tsne_log_every == 0: 
                # for lstm we take the hidden state (i.e. h_t of (h_t, c_t))
                hs = hidden_state[0] if isinstance(hidden_state, tuple) else hidden_state
                hs_dict[t] = hs.cpu().data.numpy()



#______________________________________________________________________________________
# from here we should do T-SNE --> the required hidden_states are stored in MODE's `OD`.

timesteps=MODE[0][2].keys()

for t in timesteps:

    X, y = create_matrix_for_tsne(MODE,t)

    distances = compute_tsne(X, y, t)

    print(distances)





