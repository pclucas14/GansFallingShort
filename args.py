import argparse

def get_train_args(allow_unmatched_args=False):
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
    parser.add_argument('--mle_epochs', type=int, default=100)
    parser.add_argument('--adv_epochs', type=int, default=100)
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
    parser.add_argument('--lm_path', type=str, default=None)
    parser.add_argument('--lm_epoch', type=int, default=None)
    parser.add_argument('--debug', action='store_true')

    if allow_unmatched_args: 
        args, unmatched = parser.parse_known_args()
    else: 
        args = parser.parse_args()
    
    args.cuda = False if args.no_cuda else True
    # validate a few things
    if args.transfer_weights_after_pretraining:
        assert args.hidden_dim_gen == args.hidden_dim_disc and \
            args.num_layers_gen == args.num_layers_disc, \
                'GEN and DISC architectures must be identical to enable weight sharing'

    return (args, unmatched) if allow_unmatched_args else args


# right now train and test args are kept separate. It could make sense later on to merge them
# current code is such that merging train / test args won't break anything. Let's try and keep
# it that way
def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='path to model')
    parser.add_argument('--model_epoch', type=int, default=None, help='epoch of saved model')
    parser.add_argument('--tsne_log_every', type=int, default=50, help='... every _ timestep')
    parser.add_argument('--tsne_max_t', type=int, default=400, help='run tsne exp for _ steps')
    parser.add_argument('--tsne_batch_size', type=int, default=1000)
    parser.add_argument('--draw_ellipse', action='store_true', default=False)
    args, unmatched = parser.parse_known_args()

    args.batch_size = args.tsne_batch_size

    # TODO: Check with Will & Mass what kind of behavior we want.
    args.stream_data = True
    args.max_seq_len = args.tsne_max_t

    train_args, train_unmatched = get_train_args(allow_unmatched_args=True)
    args.data_dir = train_args.data_dir
    args.debug = train_args.debug
    args.cuda = train_args.cuda

    # make sure we did not parse any invalid args
    unmatched = [x for x in unmatched if '--' in x]
    for arg_ in unmatched: 
        if arg_ in train_unmatched: 
            raise ValueError('%s is not a valid argument' % arg_)

    return args
