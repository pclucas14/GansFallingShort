import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import copy
import time 

from utils  import * 
from models import * 

def train_discriminator(gen, path, disc=None, args=None):

    # fetch args used to train generator if not provided
    args = args or gen.args
    print(args)

    if args.num_layers_disc==2:
        print('multi layer LSTM bug --> setting to single layer')
        args.num_layers_disc=1

    # dataset creation
    dataset_train, word_dict = tokenize(os.path.join(args.data_dir, 'train.txt'), \
            train=True, char_level=args.character_level, dataset=args.dataset)
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
    optimizer_disc = torch.optim.Adam(disc.parameters(), lr=1e-3) 
    BCEL = F.binary_cross_entropy_with_logits

    best_val = 1e9
    gen.train() 
    for epoch in range(1000):
        train_loader = minibatch_generator(dataset_train, args, shuffle=True)
        valid_loader = minibatch_generator(dataset_valid, args, shuffle=True)
        losses_real, losses_fake, accs_real, accs_fake, ps_real, ps_fake = [[] for _ in range(6)]
        avg  = lambda x : torch.stack(x).mean().item()
        avg_ = lambda x : torch.cat(x, dim=0).mean(dim=0)

        # Training Loop
        disc = disc.train()
        for i, minibatch in enumerate(train_loader):
            # if i > 25 : break
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
        print('TRAIN real p()  : {}'.format(avg_(ps_real)))
        print('TRAIN fake p()  : {}'.format(avg_(ps_fake)))
        print('TRAIN real acc  : {:.4f}'.format(avg(accs_real)))
        print('TRAIN fake acc  : {:.4f}'.format(avg(accs_fake)))
        losses_real, losses_fake, accs_real, accs_fake, ps_real, ps_fake = [[] for _ in range(6)]

        # Test Loop
        disc.eval()
        with torch.no_grad():
            for i, minibatch in enumerate(valid_loader):
                input, target, lens = minibatch
                # if i > 5 : break

                # test on real data
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
            print('TEST  real p()  : {}'.format(avg_(ps_real)))
            print('TEST  fake p()  : {}'.format(avg_(ps_fake)))
            print('TEST  real acc  : {:.4f}'.format(avg(accs_real)))
            print('TEST  fake acc  : {:.4f}'.format(avg(accs_fake)))
           
            valid_loss  = avg(losses_real) + avg(losses_fake)
            if valid_loss < best_val:
                best_val = valid_loss
                # save current model state
                best_disc_yet = copy.deepcopy(disc)
            else: 
                break
    
    # I want the model to also have the arguments of the Discriminator, so I'm copying back the weights
    disc.load_state_dict(best_disc_yet.state_dict())
    
    # save model
    save_models([('disc', disc, optimizer_disc)], path, 0)  
    return disc
    

def sample_from_model(model, method, num_samples, *args, **kwargs):
    """ method to sample according to the following heuristics: 
            'temperature/temp' --> temperature tuned softmax.                 Required Param : 'alpha'
            'topk'             --> random draw from top k                     Required Param : 'k'
            'weighted topk'    --> draw from top k according to o.g. mass     Required Param : 'k'
            'beam/beam search' --> stochastic beam search                     Required Param : 'beam size'
            'gen ll'           --> (self) likelihood on (own) gen. sentences  Required Param : 'threshold'
            'disc ll'          --> discriminator rejection sampling           Required Param : 'threshold'
    """

    start  = time.time()   
    method = method.lower()
    all_words = []

    # always carry at most 1000 samples on GPU to avoid memory issues
    bs        = 1000 // kwargs.get('beam_size', 1)
    sos_token = torch.LongTensor(bs, 1).fill_(2)
    if model.args.cuda: sos_token = sos_token.cuda()

    if 'disc' in method:
        model_path = kwargs['model_path']
        files = os.listdir(os.path.join(model_path,'models'))
        if len([s for s in files  if 'disc' in s]) > 0:
            try:
                disc = load_model_from_file(model_path, model='disc')[0]
            except:
                disc = load_model_from_file(model_path, epoch=0, model='disc')[0]
            print('loading old Discriminator')
            disc.cuda()
        else:
            disc = train_discriminator(model, model_path)

    while sum([x.size(0) for x in all_words]) < num_samples:
        hidden_state    = None
        input_idx       = sos_token
        words, words_ll = [], []

        # loop similar to Generator.forward()
        for t in range(model.args.max_seq_len):
            input = model.embedding(input_idx)

            output, hidden_state = model.step(input, hidden_state, t, var_drop_p=0.)
            logits = model.output_layer(output)
    
            # temperature decoding 
            if 'temp' in method: 
                alpha = kwargs['alpha']
                logits = logits * alpha
                input_idx = Categorical(logits=logits.squeeze(1)).sample().unsqueeze(1)
            
            # topk decoding 
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
            
            # generator self likelihood decoding 
            elif 'gen' in method:
                # regular sampling
                input_idx = Categorical(logits=logits.squeeze(1)).sample().unsqueeze(1)
                offset = torch.arange(input_idx.size(0)).reshape(*input_idx.size()) * logits.size(-1)
                if model.args.cuda: offset = offset.cuda()

                input_idx_flat = (input_idx + offset)
                log_probs = F.log_softmax(logits, dim=-1)
                ll_t = torch.take(log_probs, input_idx_flat)
                words_ll += [ll_t]

            # discriminator rej. sampling decoding 
            elif 'disc' in method:
                input_idx = Categorical(logits=logits.squeeze(1)).sample().unsqueeze(1)
           
            # stochastic beam search decoding  
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

                    # (bs, beam_size, beam_size) --> (bs, beam_size ** 2)
                    ll_t   = ll_t.view(ll_t.size(0), -1)

                    # TODO(lucas): do we want to mask out pad tokens and normalize inside the beam search?

                    if kwargs.get('remove_duplicates', False) and beam_size > 1: 
                        """ edit: let's try and remove duplicates """
                        vals, ids = torch.sort(ll_t, dim=-1, descending=True)
                        delta = vals[:, :-1] - vals[:, 1:]
                        valid_ids = delta != 0
                        
                        # by construction of delta, 0th position is not included
                        valid_ids = torch.cat([torch.ones_like(valid_ids[:, [0]]), 
                                               valid_ids], dim=1)

                        # we mask out valid indices
                        invalid_indices = ids.clone()
                        invalid_indices.masked_fill_(valid_ids, -1)

                        # we add the offset to the mask
                        offset = torch.arange(ll_t.size(0)).long() * beam_size ** 2
                        offset = offset.cuda() if model.args.cuda else offset

                        to_be_masked = (invalid_indices + offset.view(-1, 1))
                        
                        # add 1 and substract 1 to use torch.nonzero() as filtering
                        to_be_masked = (to_be_masked+1) * (invalid_indices != -1).long()
                        to_be_masked = to_be_masked.flatten().squeeze()
                        non_zero_ids = to_be_masked.nonzero()
                        to_be_masked = to_be_masked[non_zero_ids] - 1

                        # mask for non unique values
                        mask = torch.zeros_like(ll_t.flatten())
                        mask[to_be_masked] = 1
                        mask = mask.view(*ll_t.size())
                        
                        # what remains from here is to put very low values for values in `ll_t` 
                        # where mask == 1, i.e. where values are duplicates. This way, duplicates
                        # will only be sampled if all non-duplicate values first all been sampled
                        # ll_t.masked_fill_(mask.byte(), -9999999.)

                        # actually, by adding a big negative penalty, we can keep the original ordering.
                        # this way, if there are too many duplicates, we will still select them based on ll.
                        ll_t_with_dup_penalty = ll_t.clone() + -99999999
                        ll_t_ = ll_t_with_dup_penalty * mask + (1 - mask) * ll_t
                        """ end of edit  """ 
                        
                    # pick sentences with highest likelihood
                    top_v, top_i = torch.topk(ll_t, beam_size, dim=-1)
                    buffer = top_v

                    # we need to make sure we know the index of the prefix for v \in top_v
                    # this is actually the *per beam* index. (what will be used with offset to fetch hs)
                    prefix_i = top_i / beam_size

                    # prefix to fetch hidden_states
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
                new_hidden_state = []
                
                # iterate over layers
                for hs in hidden_state:
                    if isinstance(hs, tuple):
                        h_t, c_t = hs
                        h_t = torch.index_select(h_t, 1, prefix)
                        c_t = torch.index_select(c_t, 1, prefix)
                        hs = (h_t, c_t)
                    else: 
                        hs = torch.index_select(hs, 1, prefix)
                    
                    new_hidden_state += [hs]

                hidden_state = new_hidden_state
            
            else:
                raise ValueError('%s does not match any known method' % method)


            words = input_idx if t == 0 else torch.cat([words, input_idx], dim=1)

        """ generation over, we process gen. samples """ 
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
                accept = (joint_ll > -th).long()
            
            elif 'disc' in method:
                # push the generated sentences through the discriminator to get score
                is_real  = F.sigmoid(disc(words)[0])[:, -1]
                # we actually use the last conditional, and not the joint for this prob.
                joint_ll = is_real

                # TODO(lucas) do we want to do something about PAD tokens ? in this setting I would assume not
                # The only feasible way would be to consider all conditionals (not just the last), mask out the 
                # conditionals over PAD tokens, and average them out.
                accept = (joint_ll > th).long()
            
            arange = torch.arange(accept.size(0))
            if model.args.cuda: arange = arange.cuda()

            accept_indices = (accept * arange).nonzero().squeeze(-1)
            words = words[accept_indices]

        elif 'beam' in method: 
            # all that remains is to pick the most likely sentence *per beam*. 
            # since, for a single beam, all sentences end up being the same due to sampling with replacement, 
            # we can just pick the first sentence of each beam
            words = words[::beam_size]

        all_words += [words]
        
    output =  torch.cat(all_words, dim=0) if len(all_words) > 1 else all_words[0]
    
    # let's track sentence length (to investigate sharp drop at beam size == 2)
    EOS = 1
    is_eos = output == EOS
    sentence_length = is_eos.argmax(dim=1)
    print('average sentence length : {:.4f}'.format(sentence_length.float().mean().item()))
    print('%d / %d sentences are unique' % (np.unique(output, axis=0).shape[0], output.shape[0]))

    print('took {:.6f} seconds to sample {} with method {}'.format(time.time() - start, int(output.size(0)), method))
    return output 


def compute_lm_score(sentences, oracle_lm, verbose=False, word_dict=None, args=None):

    with torch.no_grad():
        hidden_state_oracle = None
        oracle_nlls = []
        
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
       
        if verbose: 
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
    
        return np.mean(avg_oracle_nll)


if __name__ == '__main__':
    lm_path = '../real_data_experiments/trained_models/news/word/best_mle'
    lm_path = '../real_data_experiments/exps/news/2l_fixed'
    model = load_model_from_file(lm_path, None)[0].cuda()
    model.args.data_dir = '../real_data_experiments/data/news'
    model.args.num_layers_disc = 1
    model.args.hidden_dim_disc = 512
    model.args.var_dropout_p_disc = 0.5
    
    _, word_dict = tokenize('../real_data_experiments/data/news/train.txt', train=True)

    for beam_size in [1, 5, 10, 25]: 
        for rm in [True, False]:
            print('\n\n')
            print('beam size : %d\t remove duplicates %d' % (beam_size, rm))
            kwargs = {'k':10, 'beam_size':beam_size, 'alpha':2, 'threshold' : 0.6, 'remove_duplicates' : rm}
            sentences = sample_from_model(model, 'beam', 1000, **kwargs)
            print_and_save_samples(sentences, word_dict, 'test', 0, max_print=20)
        
