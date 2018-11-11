import pdb
import random
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.distributions import Categorical

def masked_cross_entropy(logits, target, length):
    # logits : FloatTensor bs x seq_len x vocab_size
    # target : LongTensor  bs x seq_len
    # length : LongTensor  bs 
    
    bs, seq_len, vocab_size  = logits.size()
    mask = torch.arange(seq_len).unsqueeze(0).expand(bs, -1).long()
    if length.is_cuda : mask = mask.cuda()
    mask = mask  < length.unsqueeze(1).expand(-1, seq_len)

    log_probs = F.log_softmax(logits, dim=2)
    log_probs = torch.gather(log_probs, 2, target.unsqueeze(2)).squeeze(2)
    loss = - (log_probs * Variable(mask.float())).sum(dim=1)
    return loss.sum() / length.float().sum()


def reinforce_critic_loss(cumulative_rewards, fake_baselines):
    return F.mse_loss(fake_baselines, cumulative_rewards.detach())


def reinforce_gen_loss(cumulative_rewards, fake_logits, fake_sentence, baseline, args):
    # cumulative rewards : bs x seq_len             
    # fake logits        : bs x seq_len x vocab_size  (distribution @ every timestep)
    # fake sentence      : bs x seq_len               (indices for the words)
    # baseline           : bs x seq_len               (baseline coming from critic)
    assert cumulative_rewards.shape == baseline.shape == fake_sentence.shape

    bs, seq_len, vocab_size = fake_logits.shape
    advantages = cumulative_rewards
    
    # use a baseline in regular mode
    if args.use_baseline: 
        advantages -= baseline
    if args.adv_clip > 0: 
        advantages = torch.clamp(advantages, -args.adv_clip, args.adv_clip)
    advantages.detach()

    loss = 0.
    for t in range(seq_len):
        dist = Categorical(logits=fake_logits[:, t])
        log_prob = dist.log_prob(fake_sentence[:, t])
        ment_reg = args.beta * dist.entropy()
        loss += log_prob * advantages[:, t] + ment_reg
    return -loss.sum() / bs # average loss over batches

def cot_gen_loss(gen_logits, med_logits):
    '''
    gen_logits: (bs, seq_len, vocab_size)
    med_logits: (bs, seq_len, vocab_size)
    '''

    assert gen_logits.size() == med_logits.size()
    bs, seq_len, vocab_size = gen_logits.size()

    gen_logits = gen_logits.reshape(bs * seq_len, vocab_size)
    med_logits = med_logits.reshape(bs * seq_len, vocab_size)

    # target = Categorical(logits=med_logits)
    # pred   = Categorical(logits=gen_logits)

    # loss = torch.distributions.kl.kl_divergence(target, pred)
    # loss as done in https://github.com/desire2020/CoT/blob/master/generator.py line 125
    
    loss = -1 * F.softmax(gen_logits, -1) * (F.log_softmax(med_logits, -1) - F.log_softmax(gen_logits, -1))
    return loss.sum() / bs


'''
Metrics and Divergences
'''        
def KLD(p_logits, q_logits):
    if len(p_logits.size()) == 3: 
        kls = []
        for t in range(p_logits.size(1)):
            p = Categorical(logits=p_logits[:, t])
            q = Categorical(logits=q_logits[:, t])
            kl_t = kl_divergence(p, q)
            kls.append(kl_t)
        return torch.stack(kls, dim=1)
    else: 
        p = Categorical(logits=p_logits)
        q = Categorical(logits=q_logits)
        return kl_divergence(p, q)


def NLL(logits, target):
    assert logits.shape[:-1] == target.shape
    log_probs = F.log_softmax(logits, dim=-1)
    nll = torch.gather(log_probs, 2, target.unsqueeze(2))
    return -1 * nll.mean()
