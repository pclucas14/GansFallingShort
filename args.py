import argparse

def get_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setup', type=str, default='real', choices=['real', 'oracle'])

    # LOGGING args
    parser.add_argument('--base_dir', type=str, default='runs/test')
    parser.add_argument('--bleu_every', type=int, default=15)
    parser.add_argument('--save_every', type=int, default=30)
    parser.add_argument('--test_every', type=int, default=5)

    # MODEL args
    parser.add_argument('--rnn', type=str, default='LSTM', choices=['LSTM', 'GRU'])
    parser.add_argument('--hidden_dim_disc', type=int, default=256)
    parser.add_argument('--hidden_dim_gen', type=int, default=256)
    parser.add_argument('--num_layers_disc', type=int, default=1)
    parser.add_argument('--num_layers_gen', type=int, default=1)
    parser.add_argument('--var_dropout_p_gen', type=float, default=0.5)
    parser.add_argument('--var_dropout_p_disc', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.95)

    # TRAINING args
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mle_epochs', type=int, default=1)
    parser.add_argument('--adv_epochs', type=int, default=2)
    parser.add_argument('--alpha_train', type=float, default=1.)
    parser.add_argument('--alpha_test', type=float, default=1.)
    parser.add_argument('--beta', type=float, default=0.)
    parser.add_argument('--grad_clip', type=float, default=10.)
    parser.add_argument('--adv_clip',  type=float, default=5.)
    parser.add_argument('--seqgan_reward', type=int, default=0, help='reward is only at the final timestep')
    parser.add_argument('--use_baseline', type=int, default=1)
    parser.add_argument('--disc_train_iterations', '-dti', type=int, default=5) 
    parser.add_argument('--gen_train_iterations',  '-gti', type=int, default=1) 
    parser.add_argument('--mle_train_iterations',  '-mti', type=int, default=0) 
    parser.add_argument('--disc_pretrain_epochs', type=int, default=0)
    parser.add_argument('--gen_lr', type=float, default=1e-3)
    parser.add_argument('--disc_lr', type=float, default=1e-3)
    parser.add_argument('--critic_lr', type=float, default=1e-3)

    # DATA args
    parser.add_argument('--data_dir', type=str, default='data/news')
    parser.add_argument('--stream_data', action='store_true')
    parser.add_argument('--max_seq_len', type=int, default=20)

    # OTHER args
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--transfer_weights_after_pretraining', type=int, default=1)
    parser.add_argument('--sample_size_fast', type=int, default=500)
    parser.add_argument('--LM_path', type=str)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    # validate a few things
    if args.transfer_weights_after_pretraining:
        assert args.hidden_dim_gen == args.hidden_dim_disc and \
            args.num_layers_gen == args.num_layers_disc, \
                'GEN and DISC architectures must be identical to enable weight sharing'

    return args 
