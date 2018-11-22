import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import copy

from utils  import * 
from models import * 

def train_discriminator(gen, disc=None, args=None):

    # fetch args used to train generator if not provided
    args = args or gen.args
    print(args)

    # dataset creation
    dataset_train, word_dict = tokenize(os.path.join(args.data_dir, 'train.txt'), \
            train=True, char_level=args.character_level, dataset=args.dataset) #, skip=True)
    dataset_valid,  word_dict = tokenize(os.path.join(args.data_dir, 'valid.txt'), train=False, \
            word_dict=word_dict, char_level=args.character_level, dataset=args.dataset)

    # add extra args
    args.vocab_size = len(word_dict)
    args.cuda = False if args.no_cuda else True

    # create disc
    disc = disc or Discriminator(args)
    print(disc)
    if args.cuda: disc = disc.cuda()

    # build optimizer
    optimizer_disc = torch.optim.Adam(disc.parameters(), lr=1e-3) #args.disc_lr)
    BCEL = F.binary_cross_entropy_with_logits

    best_val = 1e9
    gen.train() 
    for epoch in range(1000):
        train_loader = minibatch_generator(dataset_train, args, shuffle=True)
        valid_loader = minibatch_generator(dataset_valid, args, shuffle=True)
        losses_real, losses_fake, accs_real, accs_fake, ps_real, ps_fake = [[] for _ in range(6)]
        avg = lambda x : torch.stack(x).mean().item()

        # Training Loop
        disc = disc.train()
        gen.args.alpha_train = 1. # min(1., float(epoch + 2.) / 50.)
        print('alpha :{:.4f}'.format(gen.args.alpha_train))
        for i, minibatch in enumerate(train_loader):
            if i > 250 : break
            input, target, lens = minibatch
            
            # train on real data
            real_out   = disc(target)[0]# [:, -1]
            real_loss  = BCEL(real_out, torch.ones_like(real_out))
            p_real     = F.sigmoid(real_out)
            ps_real   += [p_real.data]
            acc_real   = (p_real > 0.5).type(torch.float).mean().data
            accs_real += [acc_real.data]

            # train on fake data
            _, fake_sentences = gen(input[:, [0]])
            fake_out   = disc(fake_sentences.detach())[0]# [:, -1]
            fake_loss  = BCEL(fake_out, torch.zeros_like(fake_out))
            p_fake     = F.sigmoid(fake_out)
            ps_fake   += [p_fake.data]
            acc_fake   = (p_fake < 0.5).type(torch.float).mean().data
            accs_fake += [acc_fake.data]

            disc_loss = (fake_loss + real_loss) / 2
            losses_real += [real_loss.data]
            losses_fake += [fake_loss.data]

            apply_loss(optimizer_disc, disc_loss, clip_norm=args.grad_clip)

        print('epoch', epoch)
        print('TRAIN real loss : {:.4f}'.format(avg(losses_real)))
        print('TRAIN fake loss : {:.4f}'.format(avg(losses_fake)))
        print('TRAIN real p()  : {:.4f}'.format(avg(ps_real)))
        print('TRAIN fake p()  : {:.4f}'.format(avg(ps_fake)))
        print('TRAIN real acc  : {:.4f}'.format(avg(accs_real)))
        print('TRAIN fake acc  : {:.4f}'.format(avg(accs_fake)))
        losses_real, losses_fake, accs_real, accs_fake, ps_real, ps_fake = [[] for _ in range(6)]

        # Test Loop
        # disc.eval()
        with torch.no_grad():
            for i, minibatch in enumerate(valid_loader):
                input, target, lens = minibatch
                if i > 25 : break

                # test on real data
                # import pdb; pdb.set_trace()
                real_out   = disc(target)[0]# [:, -1]
                real_loss  = BCEL(real_out, torch.ones_like(real_out))
                p_real     = F.sigmoid(real_out)
                ps_real   += [p_real.data]
                acc_real   = (p_real > 0.5).type(torch.float).mean().data
                accs_real += [acc_real.data]

                # test on fake data
                _, fake_sentences = gen(input[:, [0]])
                fake_out   = disc(fake_sentences.detach())[0]# [:, -1]
                fake_loss  = BCEL(fake_out, torch.zeros_like(fake_out))
                p_fake     = F.sigmoid(fake_out)
                ps_fake   += [p_fake.data]
                acc_fake   = (p_fake < 0.5).type(torch.float).mean().data
                accs_fake += [acc_fake.data]

                disc_loss = (fake_loss + real_loss) / 2
                losses_real += [real_loss.data]
                losses_fake += [fake_loss.data]
            
            print('epoch ', epoch)
            print('TEST  real loss : {:.4f}'.format(avg(losses_real)))
            print('TEST  fake loss : {:.4f}'.format(avg(losses_fake)))
            print('TEST  real p()  : {:.4f}'.format(avg(ps_real)))
            print('TEST  fake p()  : {:.4f}'.format(avg(ps_fake)))
            print('TEST  real acc  : {:.4f}'.format(avg(accs_real)))
            print('TEST  fake acc  : {:.4f}'.format(avg(accs_fake)))
           
            valid_loss  = avg(losses_real) + avg(losses_fake)
            if valid_loss < best_val:
                best_val = valid_loss
                # save current model state
                best_disc_yet = copy.deepcopy(disc)
            else: 
                # break
                # continue
                x = 1

            '''
            if (avg(accs_real) + avg(accs_fake)) / 2. > 0.8: 
                alpha_train = (1. + alpha_train) / 2.
                print('alpha train now {:.4f}'.format(alpha_train))
            else:
                print('avg : {:.4f}'.format(((avg(accs_real) + avg(accs_fake)) / 2)))
            '''    

    return best_disc_yet
    

def sample_from_model(model, method, num_samples, *args, **kwargs):
    
    method    = method.lower()

    # always carry at most 1000 samples on GPU to avoid memory issues
    bs        = 1000 // (kwargs['beam_size'] if 'beam' in method else 1)
    sos_token = torch.LongTensor(bs, 1).fill_(2)
    if model.args.cuda: sos_token = sos_token.cuda()

    hidden_state    = None
    input_idx       = sos_token
    words, words_ll = [], []

    if 'disc' in method: 
        disc = train_discriminator(model)

    # loop similar to Generator.forward()
    for t in range(model.args.max_seq_len):
        input = model.embedding(input_idx)

        output, hidden_state = model.step(input, hidden_state, t, var_drop_p=0.)
        logits = model.output_layer(output)

        if 'temp' in method: 
            alpha = kwargs['alpha']
            logits = logits * alpha
            input_idx = Categorical(logits=logits.squeeze(1)).sample().unsqueeze(1)
        
        elif 'topk' in method:
            k = kwargs['k']

            # fetch most likely logits
            top_values, top_indices = torch.topk(logits, k, dim=-1)

            if 'weighted' in method:
                random_draw = Categorical(logits=top_values.squeeze()).sample()
            else:
                random_draw = torch.zeros_like(logits[:, 0, 0]).uniform_(0, k).long()

            # torch.take only works for flat tensors, so we need to manually calc. the offset
            offset = (torch.arange(random_draw.size(0)) * k)
            if model.args.cuda: offset = offset.cuda()
            random_draw_flat = (random_draw + offset).flatten()
            input_idx        = torch.take(top_indices, random_draw_flat).unsqueeze(-1)
        
        # generator (self) likelihood
        elif 'gen' in method:
            # regular sampling
            input_idx = Categorical(logits=logits.squeeze(1)).sample().unsqueeze(1)
            offset = torch.arange(input_idx.size(0)).reshape(*input_idx.size()) * logits.size(-1)
            if model.args.cuda: offset = offset.cuda()

            input_idx_flat = (input_idx + offset)
            log_probs = F.log_softmax(logits, dim=-1)
            ll_t = torch.take(log_probs, input_idx_flat)
            words_ll += [ll_t]
        
        elif 'beam' in method:
            beam_size = kwargs['beam_size']

            if t == 0 : 
                # since all "beams" start for the same dist (p(x1 | sos)), we only calc. once
                # for every beam, we sample `beam_size` words, and pick the highest scoring word.
                sample = Categorical(logits=logits[0]).sample((bs * beam_size,))
                sample = sample.reshape(bs, beam_size)    

                # select log probs of selected words
                log_probs_0 = F.log_softmax(logits[0].squeeze(), dim=-1)
                log_probs_0 = torch.take(log_probs_0, sample)
                buffer      = log_probs_0

                # (bs, beam_size) --> (bs * beam_size, )
                input_idx = sample.reshape(-1, 1)

                # prefix is there just to make sure we are fetching the correct amt of h_s for t=0
                prefix = torch.zeros_like(input_idx).squeeze(-1)
                
            else: 
                sample = Categorical(logits=logits).sample((beam_size, ))

                # (beam_size, bs * beam_size) --> (bs, beam_size, beam_size)
                sample = sample.transpose(1, 0).reshape(bs, beam_size, beam_size)

                # calculate the log likelihood of all "new" sentences
                ll_t   = F.log_softmax(logits, dim=-1).reshape(bs, beam_size, -1)
                offset = torch.arange(bs * beam_size).reshape(bs, beam_size, 1) * logits.size(-1)
                offset = offset.cuda().long() if model.args.cuda else offset.long()
                ll_t   = torch.take(ll_t, sample + offset)
                ll_t   = buffer.unsqueeze(-1).expand_as(ll_t) + ll_t

                # pick sentences with highest likelihood
                top_v, top_i = torch.topk(ll_t.view(ll_t.size(0), -1), beam_size, dim=-1)

                # we need to make sure we know the index of the prefix for v \in top_v
                prefix_i = top_i / beam_size

                # prefix to fetch hidden_states
                # TODO: I have not checked this one yet
                offset = torch.arange(bs).reshape(-1, 1) * beam_size
                offset = offset.cuda().long() if model.args.cuda else offset.long()
                prefix = (prefix_i + offset).flatten()
                
                # words chosen at timestep t that maximize likelihood for a given beam
                offset    = torch.arange(bs).reshape(-1, 1) * beam_size * beam_size
                offset    = offset.cuda().long() if model.args.cuda else offset.long()
                sample_t  = torch.take(sample, top_i + offset) 
                input_idx = sample_t.reshape(-1, 1)
            
                # choose correct words given the new vectors
                words = torch.index_select(words, 0, prefix) 

            # we need to expand hidden state for compatibility
            if isinstance(hidden_state, tuple):
                h_t, c_t = hidden_state
                h_t = torch.index_select(h_t, 1, prefix)
                c_t = torch.index_select(c_t, 1, prefix)
                hidden_state = (h_t, c_t)
            else: 
                hidden_state = torch.index_select(hidden_state, 1, prefix)
        
        words = input_idx if t == 0 else torch.cat([words, input_idx], dim=1)

    if 'll' in method: # gen_ll or dis_ll
        th = kwargs['threshold']
        PAD_TOKEN = 0.

        if 'gen' in method:
            # bs, seq_len
            words_ll = torch.stack(words_ll, dim=1).squeeze(-1)
        
            # we do NOT want to include <PAD> tokens in the calculation. We mask them out
            is_not_pad = (words != PAD_TOKEN).float()

            # get the likelihood of the joint by summing over the joint likelihoods
            # (bs, ) tensor 
            joint_ll = (words_ll * is_not_pad).sum(dim=1) / is_not_pad.sum(dim=1)
        
        elif 'disc' in method:
            # the "likelihood we wish to look at is simply the TODO
            pass

        # TODO: remove this
        th = joint_ll.mean().item()

        accept = (joint_ll > th).long()
        arange = torch.arange(accept.size(0))
        if model.args.cuda: arange = arange.cuda()

        accept_indices = (accept * arange).nonzero().squeeze(-1)
        words = words[accept_indices]

    return words

if __name__ == '__main__':
    lm_path = '../real_data_experiments/trained_models/news/word/best_mle'
    model = load_model_from_file(lm_path, None)[0].cuda()
    model.args.data_dir = '../real_data_experiments/data/news'
    model.args.num_layers_disc = 1
    model.args.hidden_dim_disc = 512
    model.args.var_dropout_p_disc = 0.5

    kwargs = {'k':10, 'beam_size':4, 'alpha':2, 'threshold' : 0.6}
    # with torch.no_grad():
    sentences = sample_from_model(model, 'disc ll', 1000, **kwargs)

    _, word_dict = tokenize('../real_data_experiments/data/news/train.txt', train=True)
    print_and_save_samples(sentences[::10], word_dict, 'test', 0, max_print=100)
