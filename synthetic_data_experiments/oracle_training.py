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
    args.max_seq_len = 20
    args.num_oracle_samples = 10000
    args.num_oracle_samples_test = 5000
    args.cuda = False if args.no_cuda else True

    # Logging
    maybe_create_dir(args.base_dir)
    maybe_create_dir(os.path.join(args.base_dir, 'models'))
    print_and_save_args(args, args.base_dir)
    writer = tensorboardX.SummaryWriter(log_dir=os.path.join(args.base_dir, 'TB'))
    writes = 0

    oracle = get_oracle(args)
    gen  = Generator(args)
    disc = Generator(get_cot_args(args)) if args.cot else Discriminator(args)
    print('generator', gen, '\ndiscriminator', disc)

    if args.cuda: 
        gen  = gen.cuda()
        disc = disc.cuda()
        oracle = oracle.cuda()

    optimizer_gen = optim.Adam(gen.parameters(), lr=args.gen_lr)

    if args.cot:  
        optimizer_disc   = optim.Adam(disc.parameters(), lr=args.disc_lr)
        optimizer_critic = None
    else:
        optimizer_critic = optim.Adam(disc.critic.parameters(), lr=args.critic_lr)
        optimizer_disc   = optim.Adam([p for (n,p) in disc.named_parameters() if 'critic' not in n], lr=args.disc_lr)

    # makes logging easier
    MODELS = [ ('gen', gen, optimizer_gen), ('disc', disc, optimizer_disc), ('critic', None, optimizer_critic)]

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

    '''
    MLE pretraining
    '''
    for epoch in range(args.mle_epochs):
        print('MLE pretraining epoch {}/{}'.format(epoch, args.mle_epochs))
        losses_train, losses_test, oracle_nlls = [], [], []
        gen.train()

        # Training loop
        for i, minibatch in enumerate(train_loader):
            if args.cuda: 
                minibatch = minibatch.cuda()

            start_token = torch.zeros_like(minibatch[:, [0]])
            input  = torch.cat([start_token, minibatch[:, :-1]], dim=1)
            target = minibatch
 
            # provide discriminator for leak signal (if args.leak_info is True)
            gen_logits, _ = gen(input, disc=disc)
            loss = NLL(gen_logits, target)
            losses_train += [loss.data]
            apply_loss(optimizer_gen, loss, clip_norm=args.grad_clip)
        
        print_and_log_scalar(writer, 'train/nll', losses_train, writes, end_token='\n')

        if (epoch + 1) % args.test_every == 0 :
            with torch.no_grad():
                for i, minibatch in enumerate(test_loader):
                    if args.cuda: 
                        minibatch = minibatch.cuda()

                    start_token = torch.zeros_like(minibatch[:, [0]])
                    input  = torch.cat([start_token, minibatch[:, :-1]], dim=1)
                    target = minibatch
             
                    gen_logits, _ = gen(input, disc=disc)
                    loss = NLL(gen_logits, target)
                    losses_test += [loss.data]

                start_token = start_token[[0]].expand(1000, -1)
                # generate a sentence, a sentence, and feed to oracle lm
                # provide discriminator for leak signal (if args.leak_info is True)
                gen_logits, gen_sample = gen(start_token, disc=disc)
                oracle_input = torch.cat([start_token, gen_sample], dim=1)
                oracle_logits, _ = oracle(oracle_input.detach())
                        
                nll = NLL(oracle_logits[:, :-1], gen_sample)
                oracle_nlls += [nll.data] 
                
                final_obj = oracle_nlls[0].mean() + torch.stack(losses_test).mean()

                print_and_log_scalar(writer, 'test/oracle_nll', oracle_nlls, writes)
                print_and_log_scalar(writer, 'test/nll', losses_test, writes)
                print_and_log_scalar(writer, 'test/final_obj', final_obj, writes, end_token='\n')

        writes += 1
        if writes > max_writes: return gen, disc

    if args.transfer_weights_after_pretraining and args.mle_epochs > 0:
        transfer_weights(gen, disc)
        print('transfered weights from generator to discriminator')


    '''
    Adversarial training
    '''
    for epoch in range(args.adv_epochs):
        print('ADV training epoch {}'.format(epoch))
        gen_losses, disc_losses, critic_losses, ps_real, ps_fake, real_accs, fake_accs, nlls, \
                cot_real_loss, cot_fake_loss = [[] for _ in range(10)]
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
                if args.cot:
                    real_logits, _ = disc(input)
                    real_loss = NLL(real_logits, target)
                    cot_real_loss += [real_loss.data]
                else:
                    real_out, _  = disc(target)
                    real_loss = F.binary_cross_entropy_with_logits(real_out, torch.ones_like(real_out))
                    p_real = F.sigmoid(real_out)
                    real_acc = (p_real[:, -1] > 0.5).type(torch.float).mean().data
                    p_real = p_real.mean().data
                    ps_real += [p_real]
                    real_accs += [real_acc]
                               
                # train disc on fake data
                _, fake_sentences = gen(input[:, [0]], disc=disc)
                if args.cot:
                    # prepend sos_token to generated sentence
                    fake_logits, _ = disc(torch.cat([input[:, [0]], fake_sentences[:, :-1]], dim=1))
                    fake_loss = NLL(fake_logits, fake_sentences)
                    cot_fake_loss += [fake_loss.data]
                else:
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
                if args.use_baseline and not args.cot: 
                    cumulative_rewards = get_cumulative_rewards(fake_out, args)
                    critic_loss = reinforce_critic_loss(cumulative_rewards, fake_baseline)
                    critic_losses += [critic_loss.data]            

                    apply_loss(optimizer_critic, critic_loss, clip_norm=args.grad_clip)
            
            if should_train_gen:
                # train generator
                fake_logits, fake_sentence = gen(input[:, [0]], disc=disc)

                if args.cot: 
                    disc_logits, _ = disc(torch.cat([input[:, [0]], fake_sentence[:, :-1]], dim=1))
                    gen_loss = cot_gen_loss(fake_logits, disc_logits)
                else:
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
        print_and_log_scalar(writer, 'train/Critic Loss', critic_losses, writes)
        print_and_log_scalar(writer, 'train/CoT Real Loss', cot_real_loss, writes)
        print_and_log_scalar(writer, 'train/CoT Fake Loss', cot_fake_loss, writes, end_token='\n')      


        if (epoch + 1) % args.test_every == 0: 
            with torch.no_grad():
                gen_losses, disc_losses, critic_losses, ps_real, ps_fake, real_accs, fake_accs, nlls, \
                        oracle_nlls, cot_real_loss, cot_fake_loss = [[] for _ in range(11)]
                gen.eval(); disc.eval()

                # Test loop
                for i, minibatch in enumerate(test_loader):
                    if args.cuda: 
                        minibatch = minibatch.cuda()

                    start_token = torch.zeros_like(minibatch[:, [0]])
                    input  = torch.cat([start_token, minibatch[:, :-1]], dim=1)
                    target = minibatch
                    
                    # disc on real data
                    if args.cot:
                        real_logits, _ = disc(input)
                        real_loss = NLL(real_logits, target)
                        cot_real_loss += [real_loss.data]
                    else:
                        real_out, _  = disc(target)
                        real_loss = F.binary_cross_entropy_with_logits(real_out, torch.ones_like(real_out))
                        p_real = F.sigmoid(real_out)
                        real_acc = (p_real[:, -1] > 0.5).type(torch.float).mean().data
                        p_real = p_real.mean().data
                        ps_real += [p_real]
                        real_accs += [real_acc]
                        
                                   
                    # disc on fake data
                    _, fake_sentences = gen(input[:, [0]], disc=disc)
                    if args.cot:
                        # prepend sos_token to generated sentence
                        fake_logits, _ = disc(torch.cat([input[:, [0]], fake_sentences[:, :-1]], dim=1))
                        fake_loss = NLL(fake_logits, fake_sentences)
                        cot_fake_loss += [fake_loss.data]
                    else:
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
                    if args.use_baseline and not args.cot: 
                        cumulative_rewards = get_cumulative_rewards(fake_out, args)
                        critic_loss = reinforce_critic_loss(cumulative_rewards, fake_baseline)
                        critic_losses += [critic_loss.data]            
                      
                    # generator in free sampling mode
                    fake_logits, fake_sentence = gen(input[:, [0]], disc=disc)
                    if args.cot: 
                        disc_logits, _ = disc(torch.cat([input[:, [0]], fake_sentence[:, :-1]], dim=1))
                        gen_loss = cot_gen_loss(fake_logits, disc_logits)
                    else:
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

                final_obj = sum([x + y for (x,y) in zip(oracle_nlls, nlls)]) / len(nlls)
                if args.cot: 
                    final_obj_cot = sum([x + y for (x,y) in zip(disc_losses, nlls)]) / len(nlls)
                    print_and_log_scalar(writer, 'test/final_obj_cot', final_obj_cot, writes)

                # logging
                print_and_log_scalar(writer, 'test/oracle_nll', oracle_nlls, writes)
                print_and_log_scalar(writer, 'test/P(real)', ps_real, writes)
                print_and_log_scalar(writer, 'test/real Accuracy', real_accs, writes)
                print_and_log_scalar(writer, 'test/P(fake)', ps_fake, writes)
                print_and_log_scalar(writer, 'test/fake Accuracy', fake_accs, writes)
                print_and_log_scalar(writer, 'test/nll', nlls, writes)
                print_and_log_scalar(writer, 'test/Gen Loss', gen_losses, writes)      
                print_and_log_scalar(writer, 'test/Disc Loss', disc_losses, writes)      
                print_and_log_scalar(writer, 'test/Critic Loss', critic_losses, writes) 
                print_and_log_scalar(writer, 'test/CoT Real Loss', cot_real_loss, writes)
                print_and_log_scalar(writer, 'test/CoT Fake Loss', cot_fake_loss, writes)      
                print_and_log_scalar(writer, 'test/final_obj', final_obj, writes, end_token='\n')               
 
        writes += 1
        if writes > max_writes: return gen, disc

        # save models
        if (epoch + 1) % args.save_every == 0: 
            save_models(MODELS, args.base_dir, writes)

    return gen, disc

if __name__ == '__main__':
    main()

