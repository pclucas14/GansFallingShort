import argparse
import pdb
import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
import tensorboardX
import __init__

from common.utils  import * 
from common.data   import * 
from common.models import * 
from common.losses import * 
from common.args   import * 


def main(args=None, max_writes=1e5):
    lambda_ = 0.
    temp = 100

    if args is None: args = get_train_args()
    # assert 'synthetic' in args.base_dir, 'make sure you are logging correctly'

    # reproducibility
    torch.manual_seed(2)
    np.random.seed(2)

    # add extra args
    args.vocab_size = 5000
    args.num_oracle_samples = 10000
    args.num_oracle_samples_test = 5000
    args.cuda = False if args.no_cuda else True

    # Logging
    maybe_create_dir(args.base_dir)
    maybe_create_dir(os.path.join(args.base_dir, 'models'))
    print_and_save_args(args, args.base_dir)
    writer = tensorboardX.SummaryWriter(log_dir=os.path.join(args.base_dir, 'TB'))
    writes = 0

    gen  = Generator(args)
    oracle = get_oracle(args)

    if args.cuda: 
        gen  = gen.cuda()
        oracle = oracle.cuda()

    optimizer_gen    = optim.Adam(gen.parameters(),         lr=args.gen_lr)

    # makes logging easier
    MODELS = [('gen', gen, optimizer_gen)]

    # we first create the synthetic dataset
    start_token = Variable(torch.zeros(args.batch_size, 1)).long().cuda() 
    if args.cuda: start_token = start_token.cuda()

    sentences = []
    for _ in range((args.num_oracle_samples + args.num_oracle_samples_test) // args.batch_size + 1):
        _, oracle_data = oracle(start_token)
        sentences += [oracle_data]

    sentences = torch.cat(sentences, dim=0).cpu().data.numpy()
    dataset_train = sentences[:args.num_oracle_samples]
    dataset_test  = sentences[args.num_oracle_samples:args.num_oracle_samples+args.num_oracle_samples_test]
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(dataset_test,   batch_size=1000, shuffle=False)

    # wrapper for loss
    NLL = lambda logits, target: F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.flatten())

    def build_cost_matrix(gen):
        
        def cos_sim(tensor):
            inner_prod = tensor.matmul(tensor.transpose(1,0))
            l2_norm = (tensor ** 2).sum(dim=-1) ** 0.5
            return inner_prod / l2_norm.unsqueeze(0) / l2_norm.unsqueeze(1)

        # we use the learned embeddings from the generator
        cos = cos_sim(gen.embedding.weight)

        # remove negative ones
        cos = (cos >= 0.).float() * cos

        # dist = cos
        dist = 1. - cos
        
        # normalize sum 
        dist /= dist.sum(dim=0)
        
        return dist

    def Wass(logits, target, dist):
        '''
        logits (torch.cuda.FloatTensor) : bs, seq_len, vocab_size
        target (torch.cuda.FloatTensor) : bs, seq_len
        dist   (torch.cuda.FloatTensor) : vocab_size, vocab_size
        '''
        bs = logits.size(0)
        logits = logits.view(-1, logits.size(-1))  # bs x seq_len, vocab_size
        target = target.flatten()                  # bx x seq_len, 
        
        target_dist = torch.index_select(dist, 0, target)
        cost = F.softmax(logits) * target_dist
        return cost.sum() / bs
    
    '''
    MLE pretraining
    '''
    for epoch in range(args.mle_epochs + args.adv_epochs + args.mle_warmup_epochs):
        print('MLE pretraining epoch {}/{}'.format(epoch, args.mle_epochs))
        nll_train, nll_test, wass_train, wass_test, oracle_nlls = [], [], [], [], []
        gen.train()
        
        # how much MLE are we talkin here?
        if epoch < args.mle_epochs:
            lambda_mle = 1.
        elif epoch < args.mle_epochs + args.mle_warmup_epochs: 
            lambda_mle = 1. - float(epoch - args.mle_epochs) / \
                    args.mle_warmup_epochs
        lambda_mle = max(lambda_mle, args.lambda_mle)
        print('lambda mle :', lambda_mle)

        # Training loop
        for i, minibatch in enumerate(train_loader):
            if args.cuda: 
                minibatch = minibatch.cuda()

            start_token = torch.zeros_like(minibatch[:, [0]])
            input  = torch.cat([start_token, minibatch[:, :-1]], dim=1)
            target = minibatch
 
            gen_logits, _ = gen(input)
            nll = NLL(gen_logits, target)
            nll_train += [nll.data]

            dist = build_cost_matrix(gen)
            wass = Wass(gen_logits, target, dist)
            wass_train += [wass.data]

            loss = lambda_mle * nll + (1. - lambda_mle) * wass
            apply_loss(optimizer_gen, loss, clip_norm=args.grad_clip)
        
        print_and_log_scalar(writer, 'train/nll', nll_train, writes)
        print_and_log_scalar(writer, 'train/wass', wass_train, writes, end_token='\n')

        if (epoch + 1) % args.test_every == 0 :
            with torch.no_grad():
                for i, minibatch in enumerate(test_loader):
                    if args.cuda: 
                        minibatch = minibatch.cuda()

                    start_token = torch.zeros_like(minibatch[:, [0]])
                    input  = torch.cat([start_token, minibatch[:, :-1]], dim=1)
                    target = minibatch
             
                    gen_logits, _ = gen(input)
                    nll = NLL(gen_logits, target) 
                    nll_test += [nll.data]
            
                    dist = build_cost_matrix(gen)
                    wass = Wass(gen_logits, target, dist)
                    wass_test += [wass.data]

                start_token = start_token[[0]].expand(1000, -1)
                # generate a sentence, a sentence, and feed to oracle lm
                # provide discriminator for leak signal (if args.leak_info is True)
                gen_logits, gen_sample = gen(start_token)
                oracle_input = torch.cat([start_token, gen_sample], dim=1)
                oracle_logits, _ = oracle(oracle_input.detach())
                        
                nll = NLL(oracle_logits[:, :-1], gen_sample)
                oracle_nlls += [nll.data] 

                print_and_log_scalar(writer, 'test/oracle_nll', oracle_nlls, writes)
                print_and_log_scalar(writer, 'test/nll', nll_test, writes)
                print_and_log_scalar(writer, 'test/wass', wass_test, writes)
                print_and_log_scalar(writer, 'test/final_obj', [x+y for (x,y) in zip(oracle_nlls, nll_test)], writes, end_token='\n')

        writes += 1
        if writes > max_writes: return gen, None


if __name__ == '__main__':
    main()
