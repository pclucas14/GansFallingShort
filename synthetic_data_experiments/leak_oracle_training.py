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
    if args is None: args = get_train_args()
    assert 'synthetic' in args.base_dir, 'make sure you are logging correctly'

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
    disc = Discriminator(args)
    oracle = get_oracle(args)

    if args.cuda: 
        gen  = gen.cuda()
        disc = disc.cuda()
        oracle = oracle.cuda()

    optimizer_gen    = optim.Adam(gen.parameters(),         lr=args.gen_lr)
    optimizer_critic = optim.Adam(disc.critic.parameters(), lr=args.critic_lr)
    optimizer_disc   = optim.Adam([p for (n,p) in disc.named_parameters() if 'critic' not in n], lr=args.disc_lr)

    # makes logging easier
    MODELS = [ ('gen', gen, optimizer_gen), 
               ('disc', disc, optimizer_disc), 
               ('critic', None, optimizer_critic) ]

    # small wrapper to sample from model
    def sample_from(model, sample_size, disc=None):
        with torch.no_grad():
            num_iters = sample_size // 512 + 1
            start_token = torch.zeros(512, 1).long()
            if args.cuda: 
                start_token = start_token.cuda()

            samples = []
            for _ in range(num_iters):
                if disc is not None:
                    samples += [model(start_token, disc=disc)[1]]
                else: 
                    samples += [model(start_token)[1]]
            
            samples = torch.cat(samples, dim=0)
            samples = samples[:sample_size]
            return samples
        
    dataset_train = sample_from(oracle, args.num_oracle_samples)
    dataset_test  = sample_from(oracle, args.num_oracle_samples_test)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(dataset_test,  batch_size=1024, shuffle=False)

    # wrapper for loss
    NLL = lambda logits, target: F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.flatten())


    # Wrappers for running 1 pretraining epoch
    # ------------------------------------------------------------------------------------------------

    def disc_pretrain_epoch(fake_dataset=None):
        # if in Leak(ish) Gan setup, perform disc pretraining prior to MLE
        if fake_dataset is None:
            fake_dataset = sample_from(gen, args.num_oracle_samples, disc=disc)
        
        real_loader = train_loader if disc.training else test_loader
        fake_loader  = torch.utils.data.DataLoader(fake_dataset)
        
        metrics = [[] for _ in range(6)]
        ps_real, real_accs, ps_fake, fake_accs, disc_losses, critic_losses = metrics

        for i, (real, fake) in enumerate(zip(real_loader, fake_loader)):
            # train disc on real data
            real_out, _  = disc(real)
            real_loss = F.binary_cross_entropy_with_logits(real_out, torch.ones_like(real_out))
            p_real = F.sigmoid(real_out)
            real_acc = (p_real[:, -1] > 0.5).type(torch.float).mean().data
            p_real = p_real.mean().data
            ps_real += [p_real]
            real_accs += [real_acc]
                           
            # train disc on fake data
            fake_out, fake_baseline = disc(fake)
            fake_loss = F.binary_cross_entropy_with_logits(fake_out, torch.zeros_like(fake_out))
            p_fake = F.sigmoid(fake_out)
            fake_acc = (p_fake[:, -1] < 0.5).type(torch.float).mean().data
            p_fake = p_fake.mean().data
            ps_fake += [p_fake]
            fake_accs += [fake_acc]
            disc_loss = (fake_loss + real_loss) / 2
            disc_losses += [disc_loss.data]

            if disc.training: 
                apply_loss(optimizer_disc, disc_loss, clip_norm=args.grad_clip)
            
            # train critic
            if args.use_baseline: 
                cumulative_rewards = get_cumulative_rewards(fake_out, args)
                critic_loss = reinforce_critic_loss(cumulative_rewards, fake_baseline)
                critic_losses += [critic_loss.data]            

                if disc.training: 
                    apply_loss(optimizer_critic, critic_loss, clip_norm=args.grad_clip)
            
        return metrics


    def gen_pretrain_epoch():
        losses = []
        loader = train_loader if gen.training else test_loader
        for i, minibatch in enumerate(loader):
            if args.cuda: 
                minibatch = minibatch.cuda()

            start_token = torch.zeros_like(minibatch[:, [0]])
            input  = torch.cat([start_token, minibatch[:, :-1]], dim=1)
            target = minibatch

            # provide discriminator for leak signal (if args.leak_info is True)
            gen_logits, _ = gen(input, disc=disc)
            loss = NLL(gen_logits, target)
            losses += [loss.data]
            
            if gen.training: 
                apply_loss(optimizer_gen, loss, clip_norm=args.grad_clip)

        return losses


    # ------------------------------------------------------------------------------------------------
    # MLE Pretraining Phase
    # ------------------------------------------------------------------------------------------------

    # start with discriminator if its hidden state is leaked to generator
    if args.leak_info:
        for epoch in range(10): #args.disc_pretrain_epochs):
            disc.train()
            print('Disc Pretrain Epoch {}/{}'.format(epoch, args.disc_pretrain_epochs))
            train_metrics = disc_pretrain_epoch()
            disc_metric_names = ['P(real)', 'real accuracy', 'P(fake)', 
                                 'fake accuracy', 'Disc Loss', 'Critic Loss']
            
            for value, name in zip(train_metrics, disc_metric_names):
                print_and_log_scalar(writer, 'train/%s' % name, value, writes)
            
            if (epoch + 1) % args.test_every == 0 :
                with torch.no_grad():
                    disc.eval()
                    test_metrics = disc_pretrain_epoch()

                    for value, name in zip(test_metrics, disc_metric_names):
                        print_and_log_scalar(writer, 'test/%s' % name, value, writes)
                    
            print('')
            writes += 1
            if writes > max_writes: return gen, disc


    # carry on with normal pretraining
    for epoch in range(args.mle_epochs):
        print('MLE pretraining epoch {}/{}'.format(epoch, args.mle_epochs))
        
        gen.train(), disc.train()
        nll_train = gen_pretrain_epoch()
        print_and_log_scalar(writer, 'train/nll', nll_train, writes)
        
        # train disc if needed
        if args.leak_info:
            disc_train_metrics = disc_pretrain_epoch()

            for value, name in zip(disc_train_metrics, disc_metric_names):
                print_and_log_scalar(writer, 'train/%s' % name, value, writes)

        if (epoch + 1) % args.test_every == 0 :
            gen.eval()

            with torch.no_grad():
                nll_test = gen_pretrain_epoch()

                # calculate nll_oracle
                gen_sample = sample_from(gen, 1024, disc=disc)
                oracle_input = torch.cat([torch.zeros_like(gen_sample[:, [0]]), gen_sample], dim=1)
                oracle_logits, _ = oracle(oracle_input.detach())
                nll_oracle = NLL(oracle_logits[:, :-1], gen_sample)
                nll_oracle_plus_test = nll_oracle + torch.stack(nll_test).mean()
                
                print_and_log_scalar(writer, 'test/nll', nll_test, writes)
                print_and_log_scalar(writer, 'test/nll_oracle', nll_oracle, writes)
                print_and_log_scalar(writer, 'test/final_obj', nll_oracle_plus_test, writes)

                if args.leak_info:
                    for i in range(args.disc_train_iterations):
                        disc.eval()
                        disc_test_metrics = disc_pretrain_epoch()

                        for value, name in zip(disc_test_metrics, disc_metric_names):
                            print_and_log_scalar(writer, 'test/%s' % name, value, writes)
            
                        if i < args.disc_train_iterations - 1: writes += 1
                        if writes > max_writes: return gen, disc
        
        print('')
        writes += 1
        if writes > max_writes: return gen, disc

    if args.transfer_weights_after_pretraining and args.mle_epochs > 0:
        transfer_weights(gen, disc)
        print('transfered weights from generator to discriminator')


    # ------------------------------------------------------------------------------------------------
    # Adversarial Training: TODO: refactor the following code
    # ------------------------------------------------------------------------------------------------

    for epoch in range(args.adv_epochs):
        print('ADV training epoch {}'.format(epoch))
        gen_losses, disc_losses, critic_losses, ps_real, ps_fake, real_accs, fake_accs, nlls = \
                [[] for _ in range(8)]
        gen.train(); disc.train()

        # Training loop
        for i, minibatch in enumerate(train_loader):
            if args.cuda: 
                minibatch = minibatch.cuda()

            start_token = torch.zeros_like(minibatch[:, [0]])
            input  = torch.cat([start_token, minibatch[:, :-1]], dim=1)
            target = minibatch
            
            should_train_gen, should_train_disc, should_train_mle = assign_training(i, epoch, args)

            if should_train_disc:
                # train disc on real data
                real_out, _  = disc(target)
                real_loss = F.binary_cross_entropy_with_logits(real_out, torch.ones_like(real_out))
                p_real = F.sigmoid(real_out)
                real_acc = (p_real[:, -1] > 0.5).type(torch.float).mean().data
                p_real = p_real.mean().data
                ps_real += [p_real]
                real_accs += [real_acc]
                               
                # train disc on fake data
                _, fake_sentences = gen(input[:, [0]], disc=disc)
                fake_out, fake_baseline = disc(fake_sentences.detach())
                fake_loss = F.binary_cross_entropy_with_logits(fake_out, torch.zeros_like(fake_out))
                p_fake = F.sigmoid(fake_out)
                fake_acc = (p_fake[:, -1] < 0.5).type(torch.float).mean().data
                p_fake = p_fake.mean().data
                ps_fake += [p_fake]
                fake_accs += [fake_acc]
                disc_loss = (fake_loss + real_loss) / 2
                disc_losses += [disc_loss.data]

                apply_loss(optimizer_disc, disc_loss, clip_norm=args.grad_clip)
                
                # train critic
                if args.use_baseline: 
                    cumulative_rewards = get_cumulative_rewards(fake_out, args)
                    critic_loss = reinforce_critic_loss(cumulative_rewards, fake_baseline)
                    critic_losses += [critic_loss.data]            

                    apply_loss(optimizer_critic, critic_loss, clip_norm=args.grad_clip)
            
            if should_train_gen:
                # train generator
                fake_logits, fake_sentence = gen(input[:, [0]], disc=disc)
                fake_out, fake_baseline = disc(fake_sentence.detach())
                cumulative_rewards = get_cumulative_rewards(fake_out, args)
                gen_loss = reinforce_gen_loss(cumulative_rewards, fake_logits, fake_sentence, 
                                              fake_baseline, args)
                gen_losses += [gen_loss.data]

                apply_loss(optimizer_gen, gen_loss, clip_norm=args.grad_clip)

            if should_train_mle:
                fake_logits, _  = gen(input, disc=disc)
                nll = NLL(fake_logits, target)
                nlls += [nll.data]
                
                apply_loss(optimizer_gen, nll, clip_norm=args.grad_clip)
            

        # logging
        print_and_log_scalar(writer, 'train/P(real)', ps_real, writes)      
        print_and_log_scalar(writer, 'train/real Accuracy', real_accs, writes)
        print_and_log_scalar(writer, 'train/P(fake)', ps_fake, writes)      
        print_and_log_scalar(writer, 'train/fake Accuracy', fake_accs, writes)
        print_and_log_scalar(writer, 'train/nll', nlls, writes)      
        print_and_log_scalar(writer, 'train/Gen Loss', gen_losses, writes)      
        print_and_log_scalar(writer, 'train/Disc Loss', disc_losses, writes)      
        print_and_log_scalar(writer, 'train/Critic Loss', critic_losses, writes, end_token='\n')      


        if (epoch + 1) % args.test_every == 0: 
            with torch.no_grad():
                gen_losses, disc_losses, critic_losses, ps_real, ps_fake, real_accs, \
                    fake_accs, nlls, oracle_nlls, mixed_nlls = [[] for _ in range(10)]
                gen.eval(); disc.eval()

                # Test loop
                for i, minibatch in enumerate(test_loader):
                    if args.cuda: 
                        minibatch = minibatch.cuda()

                    start_token = torch.zeros_like(minibatch[:, [0]])
                    input  = torch.cat([start_token, minibatch[:, :-1]], dim=1)
                    target = minibatch
                    
                    # disc on real data
                    real_out, _  = disc(target)
                    real_loss = F.binary_cross_entropy_with_logits(real_out, torch.ones_like(real_out))
                    p_real = F.sigmoid(real_out)
                    real_acc = (p_real[:, -1] > 0.5).type(torch.float).mean().data
                    p_real = p_real.mean().data
                    ps_real += [p_real]
                    real_accs += [real_acc]
                                   
                    # disc on fake data
                    _, fake_sentences = gen(input[:, [0]], disc=disc)
                    fake_out, fake_baseline = disc(fake_sentences.detach())
                    fake_loss = F.binary_cross_entropy_with_logits(fake_out, torch.zeros_like(fake_out))
                    p_fake = F.sigmoid(fake_out)
                    fake_acc = (p_fake[:, -1] < 0.5).type(torch.float).mean().data
                    p_fake = p_fake.mean().data
                    ps_fake += [p_fake]
                    fake_accs += [fake_acc]
                    disc_loss = (fake_loss + real_loss) / 2
                    disc_losses += [disc_loss.data]
                    
                    # critic
                    if args.use_baseline: 
                        cumulative_rewards = get_cumulative_rewards(fake_out, args)
                        critic_loss = reinforce_critic_loss(cumulative_rewards, fake_baseline)
                        critic_losses += [critic_loss.data]            
                      
                    # generator in free sampling mode
                    fake_logits, fake_sentence = gen(input[:, [0]], disc=disc)
                    fake_out, fake_baseline = disc(fake_sentence.detach())
                    cumulative_rewards = get_cumulative_rewards(fake_out, args)
                    gen_loss = reinforce_gen_loss(cumulative_rewards, fake_logits, fake_sentence, 
                                                  fake_baseline, args)
                    gen_losses += [gen_loss.data]

                    # generator in teacher forcing mode
                    fake_logits, _  = gen(input, disc=disc)
                    nll = NLL(fake_logits, target) 
                    nlls += [nll.data]

                    # oracle nll
                    oracle_input = torch.cat([start_token, fake_sentence], dim=1)
                    oracle_logits, _ = oracle(oracle_input)
                    oracle_nll = NLL(oracle_logits[:, :-1], fake_sentence)
                    oracle_nlls += [oracle_nll.data] 
                    


                # logging
                nll_oracle_plus_test = torch.stack([x + y for (x,y) in zip(oracle_nlls, nlls)]).mean()
                print_and_log_scalar(writer, 'test/final_obj', nll_oracle_plus_test, writes)
                print_and_log_scalar(writer, 'test/nll_oracle', oracle_nlls, writes)
                print_and_log_scalar(writer, 'test/P(real)', ps_real, writes)
                print_and_log_scalar(writer, 'test/real Accuracy', real_accs, writes)
                print_and_log_scalar(writer, 'test/P(fake)', ps_fake, writes)
                print_and_log_scalar(writer, 'test/fake Accuracy', fake_accs, writes)
                print_and_log_scalar(writer, 'test/nll', nlls, writes)
                print_and_log_scalar(writer, 'test/Gen Loss', gen_losses, writes)      
                print_and_log_scalar(writer, 'test/Disc Loss', disc_losses, writes)      
                print_and_log_scalar(writer, 'test/Critic Loss', critic_losses, writes, end_token='\n')      
                
        writes += 1
        if writes > max_writes: return gen, disc

        # save models
        if (epoch + 1) % args.save_every == 0: 
            save_models(MODELS, args.base_dir, writes)

    return gen, disc
    
if __name__ == '__main__':
    main()
