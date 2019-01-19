import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        new_hidden_state = [] 
        # wrap hid. states in arrays even if single-layer model
        if not isinstance(hidden_state, list): 
            hidden_state = [hidden_state] * len(self.rnns)

        if step == 0 and self.training and var_drop_p > 0.: 
            # new sequence --> create mask
            self.mask = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - var_drop_p)
            self.mask = Variable(self.mask, requires_grad=False) / (1 - var_drop_p)

        output = x * self.mask if self.training and var_drop_p > 0. else x

        for l, rnn in enumerate(self.rnns):
            output, new_hs = rnn(output, hidden_state[l])
            if self.training and var_drop_p > 0: output = output * self.mask

            new_hidden_state += [new_hs]

        return output, new_hidden_state


class Generator(Model):
    def __init__(self, args, is_oracle=False):
        super(Generator, self).__init__(args.num_layers_gen, args.hidden_dim_gen, args)
        
        in_size = args.hidden_dim_gen
        if args.leak_info: 
            in_size += args.hidden_dim_disc

        self.output_layer = nn.Linear(in_size, args.vocab_size)
        self.is_oracle = is_oracle

    def forward(self, x, hidden_state=None, disc=None):
        assert len(x.size()) == 2 # bs x seq_len
        ''' note that x[:, 0] is always SOS token'''

        # if only one word is given, use it as starting token, than sample from your distribution 
        teacher_force  = x.size(1) != 1
        seq_len        = x.size(1) if teacher_force else self.args.max_seq_len
        input_idx      = x[:, [0]]
        outputs, words = [], []

        if self.args.leak_info:
            assert disc is not None
            hidden_state_disc = None

        for t in range(seq_len):
            # choose first token, or overwrite sampled one
            if teacher_force or t == 0: 
                input_idx = x[:, [t]]

            input = self.embedding(input_idx)
            output, hidden_state = self.step(input, hidden_state, t, \
                    var_drop_p=self.args.var_dropout_p_gen)
    
            if self.args.leak_info:
                output_disc, hidden_state_disc = disc.step(input, hidden_state_disc, t, \
                        var_drop_p=self.args.var_dropout_p_disc)
                output = torch.cat([output, output_disc], dim=-1)

            dist = self.output_layer(output)
            alpha = self.args.alpha_train if self.training  else self.args.alpha_test
            if not self.is_oracle: 
                dist = dist * alpha
   
            if not teacher_force:
                input_idx = Categorical(logits=dist.squeeze(1)).sample().unsqueeze(1)
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
        ''' note that x[:, 0] is NOT SOS token, but the first word of sentence '''

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
    

# ----------------------------------------------------------------------------------
# Below are models used for evaluation (in eval.py)
# ----------------------------------------------------------------------------------

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    # assumes batch_first ordering
    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


class RNNClassifier(nn.Module):
    def __init__(self, hidden_state_size):
        super(RNNClassifier, self).__init__()
        self.lstm = nn.LSTM(hidden_state_size, hidden_state_size, num_layers=2, \
           batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden_state_size * 2, 2)
        self.lockdrop = LockedDropout()

    def forward(self, x):
        x = self.lockdrop(x)
        hs = self.lstm(x)[0]
        last_hs = self.lockdrop(hs)[:, -1]
        output = self.out(last_hs)
        return output

'''
Let's try a convolutional discriminator --> maybe it can pickup a more global signal
'''
class ConvNet(nn.Module):
    def __init__(self, hidden_state_size, max_seq_len):
        super(ConvNet, self).__init__()
        convs = []

        dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20,32]
        dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160,160] 
        disc_conv_layers = zip(dis_filter_sizes, dis_num_filters)
        output_size = sum(dis_num_filters)

        for layer in disc_conv_layers: 
            kernel_size, num_filters  = layer
            convs += [nn.Sequential(
                          nn.Conv1d(hidden_state_size, num_filters, kernel_size), 
                          nn.ReLU(), 
                          nn.MaxPool1d(max_seq_len - kernel_size + 1))]
        
        self.convs = nn.ModuleList(convs)
        self.output_layer = nn.Linear(output_size, 2)
        self.drop = nn.Dropout(0.6)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        x = x.transpose(2, 1).contiguous() # bs x seq_len x h_dim --> bs x h_dim x seq_len
        x = F.pad(x, (0, self.max_seq_len - x.size(2)))
        x = self.drop(x)
        # assert x.size(1) == hidden_state_size and len(x.size()) == 3
        outputs = []
        for block in self.convs:
            output = block(x)
            outputs += [output]
      
        output = torch.cat(outputs, dim=1).squeeze(2)
        output = self.output_layer(output)
        return output 

# inspired from https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
class SelfAttn(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttn,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.register_parameter('gamma', nn.Parameter(torch.zeros(1)))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X N )
            returns :
                out : self attention value + input feature 
                attention: B X N X N 
        """
        m_batchsize, C, N = x.size()
        # proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_query  = self.query_conv(x).permute(0,2,1) # B X N X C_
        proj_key =  self.key_conv(x)  # B X C_ x N
        energy =  torch.bmm(proj_query, proj_key) # B X N X N
        attention = self.softmax(energy) # B X N X N 
        proj_value = self.value_conv(x)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        # out = out.view(m_batchsize, C, width,height)
        
        out = self.gamma * out + x
        return out # , attention 
 

class ConvNetSelfAttn(nn.Module):
    def __init__(self, hidden_state_size, channels=[100, 100, 200, 200, 300, 300]):
        super(ConvNetSelfAttn, self).__init__()
        channels = [hidden_state_size] + channels
        layers = []

        for i in range(len(channels) - 1):
            layers += [nn.Conv1d(channels[i], channels[i+1], 1),
                       nn.Dropout(),
                       nn.ReLU(True), 
                       SelfAttn(channels[i+1])]

        self.main = nn.Sequential(*layers)
        self.out = nn.Linear(channels[-1], 2)

    def forward(self, x):
        # note : we want to perform attention on the sequence axis, 
        # NOT on the channels of the hidden state. 
        x = x.transpose(2, 1) 
        out = self.main(x)
        out = out.max(dim=2)[0] # max over sequence axis
        
        return self.out(out)
