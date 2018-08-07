import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.autograd import Variable
from pydoc import locate
from torch.distributions import Categorical

from utils import * 

'''
General Class Wrapper around RNNs that supports variational dropout
'''
class Model(nn.Module):
    def __init__(self, num_layers, hidden_dim, args):
        super(Model, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(args.vocab_size, hidden_dim)
        rnn       = locate('torch.nn.{}.'.format(args.rnn))
        self.rnns = [ rnn(hidden_dim, hidden_dim, num_layers=1, batch_first=True) 
                        for _ in range(num_layers) ]

        self.rnns = nn.ModuleList(self.rnns)
        self.mask = None

    def step(self, x, hidden_state, step, var_drop_p=0.5):
        assert x.size(1)  == 1, 'this method is for single timestep use only'
        
        if step == 0 and self.training and var_drop_p > 0.: 
            # new sequence --> create mask
            self.mask = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - var_drop_p)
            self.mask = Variable(self.mask, requires_grad=False) / (1 - var_drop_p)

        output = x * self.mask if self.training and var_drop_p > 0. else x

        for l, rnn in enumerate(self.rnns):
            output, hidden_state = rnn(output, hidden_state)
            if self.training and var_drop_p > 0: output = output * self.mask

        return output, hidden_state 


class Generator(Model):
    def __init__(self, args, is_oracle=False):
        super(Generator, self).__init__(args.num_layers_gen, args.hidden_dim_gen, args)
        self.output_layer = nn.Linear(args.hidden_dim_gen, args.vocab_size)
        self.is_oracle = is_oracle

    def forward(self, x, hidden_state=None):
        assert len(x.size()) == 2 # bs x seq_len
        
        # if only one word is given, use it as starting token, than sample from your distribution 
        teacher_force  = x.size(1) != 1
        seq_len        = x.size(1) if teacher_force else self.args.max_seq_len
        input_idx      = x[:, [0]]
        outputs, words = [], []

        for t in range(seq_len):
            # choose first token, or overwrite sampled one
            if teacher_force or t == 0: 
                input_idx = x[:, [t]]

            input = self.embedding(input_idx)
            output, hidden_state = self.step(input, hidden_state, t, var_drop_p=self.args.var_dropout_p_gen)
            dist = self.output_layer(output)
            alpha = self.args.alpha_train if self.training  else self.args.alpha_test
            if not self.is_oracle: 
                dist = dist * alpha
   
            if not teacher_force:
                if self.training or self.is_oracle or True: 
                    input_idx = Categorical(logits=dist.squeeze(1)).sample().unsqueeze(1)
                else: 
                    input_idx = dist.squeeze(1).max(dim=1)[1].unsqueeze(1)
                words += [input_idx]

            # note : these are 1-off with input, or aligned with target
            outputs += [dist] 
        
        if not teacher_force : 
            words = torch.cat(words, dim=1)
        
        logits = torch.cat(outputs, dim=1)
        return logits, words


class Discriminator(Model):
    def __init__(self, args):
        super(Discriminator, self).__init__(args.num_layers_disc, args.hidden_dim_disc, args)
        self.output_layer = nn.Linear(args.hidden_dim_disc, 1)
        self.critic       = nn.Linear(args.hidden_dim_disc, 1)
    
    def forward(self, x, hidden_state=None):
        assert len(x.size()) == 2 # bs x seq_len
        baseline = torch.ones_like(x[:, [0]]).float() * np.log(0.5)

        emb = self.embedding(x)
        outputs  = []
        for t in range(emb.size(1)):
            output, hidden_state = self.step(emb[:, [t]], hidden_state, t, var_drop_p=self.args.var_dropout_p_disc)
            outputs += [output]

        output = torch.cat(outputs, dim=1)
        disc_logits = self.output_layer(output).squeeze(-1)
        baseline_ = self.critic(output.detach()).squeeze(-1) # critic gradient should not flow
        baseline = torch.cat([baseline, baseline_], dim=1)[:, :-1]
        return disc_logits, baseline
    

