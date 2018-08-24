import argparse

def get_train_args(allow_unmatched_args=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--setup', type=str, default='real', choices=['real', 'oracle','rlm'])

    # LOGGING args
    parser.add_argument('--base_dir', type=str, default='runs/test')
    parser.add_argument('--bleu_every', type=int, default=0)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--test_every', type=int, default=2)

    # MODEL args
    parser.add_argument('--rnn', type=str, default='LSTM', choices=['LSTM', 'GRU'])
    parser.add_argument('--hidden_dim_disc', type=int, default=512)
    parser.add_argument('--hidden_dim_gen', type=int, default=512)
    parser.add_argument('--num_layers_disc', type=int, default=2)
    parser.add_argument('--num_layers_gen', type=int, default=2)
    parser.add_argument('--var_dropout_p_gen', type=float, default=0.5)
    parser.add_argument('--var_dropout_p_disc', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--load_gen_path', type=str, default=None)
    parser.add_argument('--load_disc_path', type=str, default=None)

    # TRAINING args
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--mle_epochs', type=int, default=80)
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
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--stream_data', action='store_true', default=False)
    parser.add_argument('--max_seq_len', type=int, default=51)
    parser.add_argument('--mask_padding', action='store_true', default=False)
    parser.add_argument('--character_level', action='store_true', default=False)

    # OTHER args
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--transfer_weights_after_pretraining', type=int, default=1)
    parser.add_argument('--sample_size_fast', type=int, default=500)
    parser.add_argument('--lm_path', type=str, default='trained_models/news/word/best_mle')
    parser.add_argument('--lm_epoch', type=int, default=None)

    # for RLM:
    parser.add_argument('--rlm_log_dir', type=str, default="")
    parser.add_argument('--rlm_tb', type=str, default="")

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
    parser.add_argument('--model_path', type=str, default="trained_models/news/word/best_mle", help='path to model')
    parser.add_argument('--model_epoch', type=int, default=None, help='epoch of saved model')
    parser.add_argument('--tsne_log_every', type=int, default=1, help='... every _ timestep')
    parser.add_argument('--tsne_max_t', type=int, default=55, help='run tsne exp for _ steps')
    parser.add_argument('--tsne_batch_size', type=int, default=10000)
    parser.add_argument('--draw_ellipse', action='store_true', default=False)
    parser.add_argument('--n_topics', type=int, default=2, help="topics in VTSNE")
    parser.add_argument('--n_iter', type=int, default=10, help="number of tsne iterations")
    parser.add_argument('--tsne_perp', type=int, default=30, help="perplexity in TSNE")
    parser.add_argument('--oracle_nll_log_every', type=int, default=2)
    parser.add_argument('--alpha_test', type=float, default=1.0)
    parser.add_argument('--breakpoint', type=int, default=8, help="sentence completion breakpoint")
    parser.add_argument('--n_grams', nargs="+", type=int)
    parser.add_argument('--use_conv_net', action='store_true')
    parser.add_argument('--classify_embeddings', action='store_true')
    
    # classifer exps
    parser.add_argument('--run_svm',  action='store_true', default=False)
    parser.add_argument('--run_nn' ,  action='store_true', default=False)
    parser.add_argument('--run_rnn',  action='store_true', default=False)
    parser.add_argument('--run_tsne', action='store_true', default=False)
    parser.add_argument('--run_rlm',  action='store_true', default=False)
    parser.add_argument('--run_sc',  action='store_true', default=False)


    args, unmatched = parser.parse_known_args()

    args.batch_size = args.tsne_batch_size

    # TODO: Check with Will & Mass what kind of behavior we want.
    args.stream_data = False
    args.max_seq_len = args.tsne_max_t
    args.mask_padding = False

    train_args, train_unmatched = get_train_args(allow_unmatched_args=True)
    args.data_dir = train_args.data_dir
    args.cuda = train_args.cuda
    args.lm_path = train_args.lm_path
    args.lm_epoch = train_args.lm_epoch
    args.character_level = train_args.character_level

    # make sure we did not parse any invalid args
    unmatched = [x for x in unmatched if '--' in x]
    for arg_ in unmatched: 
        if arg_ in train_unmatched: 
            raise ValueError('%s is not a valid argument' % arg_)

    return args



# args for Reverse Language Modeling:
def get_rlm_args():

    args, _ = get_train_args(allow_unmatched_args=True)
    
    args.setup = 'rlm'

    # LOGGING args
    args.bleu_every=0
    args.save_every=1e5
    args.test_very=2

    # MODEL args
    args.rnn = 'LSTM'
    args.hidden_dim_disc = args.hidden_dim_gen = 512
    args.num_layers_disc = 2
    args.num_layers_gen = 2
    args.var_dropout_p_gen = 0.5


    # TRAINING args
    args.batch_size=128
    args.mle_epochs=100
    args.adv_epochs=0
    args.alpha_train=1.
    args.alpha_test=1.
    args.beta=0.
    args.grad_clip=10.
    args.gen_lr = 1e-3

    # DATA args
    args.stream_data = False
    args.max_seq_len = 60
    args.mask_padding = True

    # OTHER args
    args.lm_path = None

    return args

