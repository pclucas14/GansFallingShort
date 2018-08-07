import argparse
import pdb
import numpy as np
import torch
import torch.optim as optim
import tensorboardX

from utils  import * 
from data   import * 
from models import * 
from losses import * 
from args   import * 

args = get_train_args()
args.cuda = False if args.no_cuda else True

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

gen  = Generator(args)
disc = Discriminator(args)

if args.cuda: 
    gen  = gen.cuda()
    disc = disc.cuda()

optimizer_gen    = optim.Adam(gen.parameters(),         lr=args.gen_lr)
optimizer_critic = optim.Adam(disc.critic.parameters(), lr=args.critic_lr)
optimizer_disc   = optim.Adam([p for (n,p) in disc.named_parameters() if 'critic' not in n], lr=args.disc_lr)

# makes logging easier
MODELS = [ ('gen', gen, optimizer_gen), ('disc', disc, optimizer_disc), ('critic', None, optimizer_critic)]

'''
MLE pretraining
'''
for epoch in range(args.mle_epochs):
    print('MLE pretraining epoch {}/{}'.format(epoch, args.mle_epochs))
    train_loader = minibatch_generator(dataset_train, args, shuffle=True)
    losses_train, losses_test = [], []
    gen.train()

    # Training loop
    for i, minibatch in enumerate(train_loader):
        input, target, lens = minibatch
        
        gen_logits, _ = gen(input)
        loss = masked_cross_entropy(gen_logits, target, lens)
        losses_train += [loss.data]
        apply_loss(optimizer_gen, loss, clip_norm=args.grad_clip)
    
    print_and_log_scalar(writer, 'train/nll', losses_train, writes, end_token='\n')

    if (epoch + 1) % args.test_every == 0: 
        test_loader  = minibatch_generator(dataset_test,  args, shuffle=False)
        with torch.no_grad():
            gen.eval()

            # Test loop
            for i, minibatch in enumerate(test_loader):
                input, target, lens = minibatch

                gen_logits, _ = gen(input)
                loss = masked_cross_entropy(gen_logits, target, lens)
                losses_test += [loss.data]

            print_and_log_scalar(writer, 'test/nll', losses_test, writes, end_token='\n')

    writes += 1
       
    # save samples
    gen.eval()
    fake_logits, fake_sentences = gen(input[:, [0]])
    print_and_save_samples(fake_sentences, word_dict, args.base_dir, epoch)

    if (epoch + 1) % args.save_every == 0: 
        save_models(MODELS[0:1], args.base_dir, writes)

if args.transfer_weights_after_pretraining and args.mle_epochs > 0:
    transfer_weights(gen, disc)
    print('transfered weights from generator to discriminator')


'''
Adversarial training
'''
for epoch in range(args.adv_epochs):
    print('ADV training epoch {}'.format(epoch))
    train_loader = minibatch_generator(dataset_train, args, shuffle=True)
    gen_losses, disc_losses, critic_losses, ps_real, ps_fake, nlls = [], [], [], [], [], []
    gen.train(); disc.train()

    # Training loop
    for i, minibatch in enumerate(train_loader):
        input, target, lens = minibatch
        should_train_gen, should_train_disc, should_train_mle = assign_training(i, epoch, args)

        if should_train_disc:
            # train disc on real data
            real_out, _  = disc(target)
            real_loss = F.binary_cross_entropy_with_logits(real_out, torch.ones_like(real_out))
            p_real = F.sigmoid(real_out).mean().data
            ps_real += [p_real]
                           
            # train disc on fake data
            _, fake_sentences = gen(input[:, [0]])
            fake_out, fake_baseline = disc(fake_sentences.detach())
            fake_loss = F.binary_cross_entropy_with_logits(fake_out, torch.zeros_like(fake_out))
            p_fake = F.sigmoid(fake_out).mean().data
            ps_fake += [p_fake]
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
            fake_logits, fake_sentence = gen(input[:, [0]])
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
    print_and_log_scalar(writer, 'train/P(fake)', ps_fake, writes)      
    print_and_log_scalar(writer, 'train/nll', nlls, writes)      
    print_and_log_scalar(writer, 'train/Gen Loss', gen_losses, writes)      
    print_and_log_scalar(writer, 'train/Disc Loss', disc_losses, writes)      
    print_and_log_scalar(writer, 'train/Critic Loss', critic_losses, writes, end_token='\n')      


    if (epoch + 1) % args.test_every == 0: 
        test_loader  = minibatch_generator(dataset_test,  args, shuffle=False)
        with torch.no_grad():
            gen_losses, disc_losses, critic_losses, ps_real, ps_fake, nlls = [], [], [], [], [], []
            gen.eval(); disc.eval()

            # Test loop
            for i, minibatch in enumerate(test_loader):
                input, target, lens = minibatch
                
                # disc on real data
                real_out, _  = disc(target)
                real_loss = F.binary_cross_entropy_with_logits(real_out, torch.ones_like(real_out))
                p_real = F.sigmoid(real_out).mean().data
                ps_real += [p_real]
                               
                # disc on fake data
                _, fake_sentences = gen(input[:, [0]])
                fake_out, fake_baseline = disc(fake_sentences.detach())
                fake_loss = F.binary_cross_entropy_with_logits(fake_out, torch.zeros_like(fake_out))
                p_fake = F.sigmoid(fake_out).mean().data
                ps_fake += [p_fake]
                disc_loss = (fake_loss + real_loss) / 2
                disc_losses += [disc_loss.data]
                
                # critic
                if args.use_baseline: 
                    cumulative_rewards = get_cumulative_rewards(fake_out, args)
                    critic_loss = reinforce_critic_loss(cumulative_rewards, fake_baseline)
                    critic_losses += [critic_loss.data]            
                  
                # generator in free sampling mode
                fake_logits, fake_sentence = gen(input[:, [0]])
                fake_out, fake_baseline = disc(fake_sentence.detach())
                cumulative_rewards = get_cumulative_rewards(fake_out, args)
                gen_loss = reinforce_gen_loss(cumulative_rewards, fake_logits, fake_sentence, 
                                              fake_baseline, args)
                gen_losses += [gen_loss.data]
                
                # generator in teacher forcing mode
                fake_logits, _  = gen(input)
                nll = masked_cross_entropy(fake_logits, target, lens)
                nlls += [nll.data]
        
            # logging
            print_and_log_scalar(writer, 'test/P(real)', ps_real, writes)      
            print_and_log_scalar(writer, 'test/P(fake)', ps_fake, writes)      
            print_and_log_scalar(writer, 'test/nll', nlls, writes)      
            print_and_log_scalar(writer, 'test/Gen Loss', gen_losses, writes)      
            print_and_log_scalar(writer, 'test/Disc Loss', disc_losses, writes)      
            print_and_log_scalar(writer, 'test/Critic Loss', critic_losses, writes, end_token='\n')      
            
    writes += 1

    # save samples
    gen.eval()
    fake_logits, fake_sentences = gen(input[:, [0]])
    print_and_save_samples(fake_sentences, word_dict, args.base_dir, epoch)

    # save models
    if (epoch + 1) % args.save_every == 0: 
        save_models(MODELS, args.base_dir, writes)
