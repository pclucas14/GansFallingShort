import argparse
import pdb
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
import tensorboardX
import __init__

from common.utils  import * 
from common.data   import * 
from common.models import * 
from common.losses import * 
from common.args   import * 


def main(rlm=False, rlm_dir=None):
    
    # Wrappers for running 1 pretraining epoch
    # --------------------------------------------------------------------------------------------

    # small wrapper to sample from model
    def sample_from(model, sample_size, disc=None, size=2048):
        with torch.no_grad():
            num_iters = sample_size // size + 1
            start_token = torch.zeros(size, 1).long() + 2
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

    def disc_pretrain_epoch(fake_dataset=None):
        # if in Leak(ish) Gan setup, perform disc pretraining prior to MLE
        if fake_dataset is None:
            fake_dataset = sample_from(gen, len(dataset_train), disc=disc)
       
        dl = lambda ds, bs, sh: torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=sh)
        
        if gen.training: 
            real_loader = minibatch_generator(dataset_train, args, shuffle=True)
            fake_loader = dl(fake_dataset, args.batch_size, True)
        else: 
            real_loader = minibatch_generator(dataset_test, args, shuffle=False)
            fake_loader = dl(fake_dataset, args.batch_size, False)
        
        metrics = [[] for _ in range(6)]
        ps_real, real_accs, ps_fake, fake_accs, disc_losses, critic_losses = metrics

        for i, (real, fake) in enumerate(zip(real_loader, fake_loader)):
            _, real, _  = real
            if args.cuda: 
                fake = fake.cuda()

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


    def gen_pretrain_epoch(dataset):
        if gen.training: 
            loader = minibatch_generator(dataset, args, shuffle=True)
        else: 
            loader = minibatch_generator(dataset, args, shuffle=False)

        losses = []
        for i, minibatch in enumerate(loader):
            input, target, len = minibatch

            # provide discriminator for leak signal (if args.leak_info is True)
            gen_logits, _ = gen(input, disc=disc)
            loss = masked_cross_entropy(gen_logits, target, len)
            losses += [loss.data]
            
            if gen.training: 
                apply_loss(optimizer_gen, loss, clip_norm=args.grad_clip)

        return losses

    
    # main
    # --------------------------------------------------------------------------------------------

    args = get_train_args()

    # reproducibility
    torch.manual_seed(2)
    np.random.seed(2)

    # dataset creation
    dataset_train, word_dict = tokenize(os.path.join(args.data_dir, 'train.txt'), \
            train=True, char_level=args.character_level, dataset=args.dataset)
    dataset_valid,  word_dict = tokenize(os.path.join(args.data_dir, 'valid.txt'), train=False, \
            word_dict=word_dict, char_level=args.character_level, dataset=args.dataset)
    dataset_test,  word_dict = tokenize(os.path.join(args.data_dir, 'test.txt'), train=False, \
            word_dict=word_dict, char_level=args.character_level, dataset=args.dataset)

    # TODO: remove this
    # dataset_train = dataset_train[:2000]
    # dataset_test  = dataset_test[:500]
    # dataset_valid = dataset_valid[:500]

    if rlm:
        args = get_rlm_args()
        dataset_train,  word_dict = tokenize(os.path.join(rlm_dir, 'train.txt'), \
                train=False, word_dict=word_dict, char_level=args.character_level, skip=True)
        

    # add extra args
    args.vocab_size = len(word_dict)
    args.cuda = False if args.no_cuda else True

    # Logging
    maybe_create_dir(args.base_dir)
    maybe_create_dir(os.path.join(args.base_dir, 'samples'))
    maybe_create_dir(os.path.join(args.base_dir, 'models'))
    print_and_save_args(args, args.base_dir)
    writer = tensorboardX.SummaryWriter(log_dir=os.path.join(args.base_dir, 'TB'))
    writes = 0
    best_valid, best_test = 1e5, 1e5

    gen  = Generator(args)
    disc = Discriminator(args)

    if args.load_gen_path:
        gen  = load_model_from_file(args.load_gen_path)[0]
    if args.load_disc_path:
        disc = load_model_from_file(args.load_disc_path, model='disc')[0]

    # load a pretrained lm as Oracle to evaluate quality of samples from our model
    if args.lm_path: 
        oracle_lm = load_model_from_file(args.lm_path, epoch=args.lm_epoch)[0]

    if args.cuda: 
        gen  = gen.cuda()
        disc = disc.cuda()
        if args.lm_path: oracle_lm = oracle_lm.cuda()

    optimizer_gen    = optim.Adam(gen.parameters(),         lr=args.gen_lr)
    optimizer_critic = optim.Adam(disc.critic.parameters(), lr=args.critic_lr)
    optimizer_disc   = optim.Adam([p for (n,p) in disc.named_parameters() if 'critic' not in n], lr=args.disc_lr)

    # makes logging easier
    MODELS = [ ('gen', gen, optimizer_gen), ('disc', disc, optimizer_disc), ('critic', None, optimizer_critic)]

    
    # ------------------------------------------------------------------------------------------------
    # MLE Pretraining Phase
    # ------------------------------------------------------------------------------------------------
    
    # start with discriminator if its hidden state is leaked to generator
    if args.leak_info:
        for epoch in range(1): #args.disc_pretrain_epochs):
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
    
    
    # carry on with normal pretraining
    for epoch in range(args.mle_epochs):
        print('MLE pretraining epoch {}/{}'.format(epoch, args.mle_epochs))
        best_valid = best_test = 1e5
        
        gen.train(), disc.train()
        nll_train = gen_pretrain_epoch(dataset_train)
        print_and_log_scalar(writer, 'train/nll', nll_train, writes)
        
        # train disc if needed
        if args.leak_info:
            disc_train_metrics = disc_pretrain_epoch()

            for value, name in zip(disc_train_metrics, disc_metric_names):
                print_and_log_scalar(writer, 'train/%s' % name, value, writes)

        if (epoch + 1) % args.test_every == 0 :
            gen.eval()
            
            if args.leak_info:
                disc.eval()
                disc_test_metrics = disc_pretrain_epoch()

                for value, name in zip(disc_test_metrics, disc_metric_names):
                    print_and_log_scalar(writer, 'test/%s' % name, value, writes)

            for split in ['valid','test']:
                dataset = dataset_valid if split=='valid' else dataset_test
                with torch.no_grad():
                    gen.eval()
                    nll_test = gen_pretrain_epoch(dataset)

                    # calculate nll_oracle
                    gen_sample = sample_from(gen, 1000, disc=disc)
                    start_token = torch.zeros_like(gen_sample[:, [0]]) + 2
                    oracle_input = torch.cat([start_token, gen_sample], dim=1)
                    oracle_logits, _ = oracle_lm(oracle_input.detach())
                    nll_oracle = NLL(oracle_logits[:, :-1], gen_sample)
                
                    print_and_log_scalar(writer, '%s/nll' % split, nll_test, writes)
                    print_and_log_scalar(writer, '%s/nll_oracle' % split, nll_oracle, writes)

                    # keep tab of best valid error in order to get legit test error:
                    if split == 'valid':
                        curr_valid_loss = np.mean(nll_test)
                        best_valid = min(best_valid,curr_valid_loss)
                    if split == 'test':
                        best_test = np.mean(nll_test) if best_valid==curr_valid_loss else best_test
        
        print('')
        writes += 1
    
        # save samples
        gen.eval()
        fake_sentences = sample_from(gen, args.batch_size, disc=disc)
        print_and_save_samples(fake_sentences, word_dict, args.base_dir, epoch, char_level=args.character_level)

        if (epoch + 1) % args.save_every == 0: 
            save_models(MODELS[0:1], args.base_dir, writes)

    # if in rlm mode, store the rlm_score
    if rlm:
        return best_test

    if args.transfer_weights_after_pretraining and args.mle_epochs > 0:
        transfer_weights(gen, disc)
        print('transfered weights from generator to discriminator')


    # ------------------------------------------------------------------------------------------------
    # Adversarial training: TODO: refactor the following code
    # ------------------------------------------------------------------------------------------------
    
    for epoch in range(args.adv_epochs):
        print('ADV training epoch {}'.format(epoch))
        train_loader = minibatch_generator(dataset_train, args, shuffle=True)
        gen_losses, disc_losses, critic_losses, ps_real, ps_fake, real_accs, fake_accs, nlls = \
                [[] for _ in range(8)]
        gen.train(); disc.train()

        # Training loop
        for i, minibatch in enumerate(train_loader):
            input, target, lens = minibatch
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
                fake_logits, _  = gen(input)
                nll = masked_cross_entropy(fake_logits, target, lens)
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
            valid_loader  = minibatch_generator(dataset_valid,  args, shuffle=False)
            with torch.no_grad():
                gen_losses, disc_losses, critic_losses, ps_real, ps_fake, real_accs, \
                    fake_accs, nlls, oracle_nlls, mixed_nlls = [[] for _ in range(10)]
                gen.eval(); disc.eval()

                # Test loop
                for i, minibatch in enumerate(valid_loader):
                    input, target, lens = minibatch
                    
                    # disc on real data
                    real_out, _  = disc(target)
                    real_loss = F.binary_cross_entropy_with_logits(real_out, torch.ones_like(real_out))
                    p_real = F.sigmoid(real_out)
                    real_acc = (p_real[:, -1] >0.5).type(torch.float).mean().data
                    p_real = p_real.mean().data
                    ps_real += [p_real]
                    real_accs += [real_acc]
                    
                                   
                    # disc on fake data
                    _, fake_sentences = gen(input[:, [0]], disc=disc)
                    fake_out, fake_baseline = disc(fake_sentences.detach())
                    fake_loss = F.binary_cross_entropy_with_logits(fake_out, torch.zeros_like(fake_out))
                    p_fake = F.sigmoid(fake_out)
                    fake_acc = (p_fake[:, -1] <0.5).type(torch.float).mean().data
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
                    nll = masked_cross_entropy(fake_logits, target, lens)
                    nlls += [nll.data]
                    
                    if args.lm_path: 
                        # generate a sentence, a sentence, and feed to oracle lm
                        oracle_input = torch.cat([input[:, [0]], fake_sentence], dim=1)
                        oracle_logits, _ = oracle_lm(oracle_input.detach())
                    
                        oracle_nll = NLL(oracle_logits[:, :-1], fake_sentence)
                        oracle_nlls += [oracle_nll.data] 
                        
                        mixed_nlls += [(nll.data+oracle_nll.data)/2]
                    
            
                # logging
                print_and_log_scalar(writer, 'valid/oracle_nll', oracle_nlls, writes)
                print_and_log_scalar(writer, 'valid/P(real)', ps_real, writes)
                print_and_log_scalar(writer, 'valid/real Accuracy', real_accs, writes)
                print_and_log_scalar(writer, 'valid/P(fake)', ps_fake, writes)
                print_and_log_scalar(writer, 'valid/fake Accuracy', fake_accs, writes)
                print_and_log_scalar(writer, 'valid/nll', nlls, writes)
                print_and_log_scalar(writer, 'valid/mixed nll', mixed_nlls, writes)
                print_and_log_scalar(writer, 'valid/Gen Loss', gen_losses, writes)      
                print_and_log_scalar(writer, 'valid/Disc Loss', disc_losses, writes)      
                print_and_log_scalar(writer, 'valid/Critic Loss', critic_losses, writes, end_token='\n')      
                
        writes += 1

        # save samples
        gen.eval()
        fake_logits, fake_sentences = gen(input[:, [0]], disc=disc)
        print_and_save_samples(fake_sentences, word_dict, args.base_dir, epoch, char_level=args.character_level)

        # save models
        if (epoch + 1) % args.save_every == 0: 
            save_models(MODELS, args.base_dir, writes)


if __name__ == '__main__':
    main()

