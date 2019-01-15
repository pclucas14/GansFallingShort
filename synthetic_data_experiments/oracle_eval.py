import argparse
import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
import tensorboardX
from oracle_training import main as train
import __init__
import sys

from common.utils  import * 
from common.data   import * 
from common.models import * 
from common.losses import * 
from common.args   import * 

TEMPERATURES = np.arange(0.1, 2.5, 0.03)

# wrapper for loss
NLL = lambda logits, target: F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.flatten())

# reproducibility
torch.manual_seed(1994)
np.random.seed(1994)    

# small wrapper to sample from model
def sample_from(model, sample_size, disc=None, cuda=True):
    with torch.no_grad():
        num_iters = sample_size // 2000 + 1
        start_token = torch.zeros(2000, 1).long()
        if cuda: 
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


# build a new test set from oracle
oracle = get_oracle().cuda()
dataset_test  = sample_from(oracle, 10000) #20000)[10000:]
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1000)

# Wrappers for Models to be evaluated
class Model_eval:
    def __init__(self, name, params_to_override, stop_write):
        self.name = name
        args = vars(get_train_args())
        for key, value in params_to_override.items():
            if key not in args.keys():
                raise ValueError('%s is an invalid argument' % key)

            args[key] = value
        
        args = to_attr(args)
        base_dir = 'synthetic_eval'
        args.base_dir = os.path.join(base_dir, name)
        args.max_seq_len = 20

        self.args = args
        self.stop_write = stop_write
    
    def get_trained_models(self):
        try:
            gen = load_model_from_file(self.args.base_dir, model='gen')[0]
            # disc = load_model_from_file(self.args.base_dir, model='disc')[0]
            if self.args.cuda: 
                gen.cuda(), # disc.cuda()

            self.gen, self.disc = gen, None #disc
            print('loaded pretrained model')
        except:
            print('training models')
            gen, disc = train(args=self.args, max_writes=self.stop_write)
            self.gen, self.disc = gen, disc
            
            # save models
            save_models([('gen', gen, None), ('disc', disc, None)], self.args.base_dir, 0)

    def eval_gen(self):
        with torch.no_grad():
            start_token = torch.cuda.LongTensor(1000, 1).fill_(0) # SOS Token
            oracle_temp_nlls = {}      
            test_temp_nlls   = {}    

            for alpha in TEMPERATURES:
                oracle_nlls, test_nlls = [], []
                gen, disc = self.gen, self.disc
                gen.eval(); # disc.eval()
                assert not gen.training

                gen.args.alpha_test = alpha; 
                assert gen.args.alpha_test == alpha
            
                for i in range(10): # 10k 
                    gen_sample = gen(start_token, disc=disc)[1]
                    oracle_input = torch.cat([start_token, gen_sample], dim=1)
                    oracle_logits = oracle(oracle_input)[0]
                    oracle_nll = NLL(oracle_logits[:, :-1], gen_sample)
                    oracle_nlls += [oracle_nll.item()]

                oracle_temp_nlls[alpha] = np.mean(oracle_nlls)

                for minibatch in test_loader: 
                    input = torch.cat([start_token, minibatch[:, :-1]], dim=1)
                    target = minibatch

                    gen_logits = gen(input, disc=disc)[0]
                    test_nll = NLL(gen_logits, target)
                    test_nlls += [test_nll.item()]

                test_temp_nlls[alpha] = np.mean(test_nlls)

        self.nll_test = test_temp_nlls
        self.nll_oracle = oracle_temp_nlls  
        print('oracle nll')
        print(self.nll_oracle)


    def log(self):
        writer = tensorboardX.SummaryWriter(self.args.base_dir)
        for alpha in TEMPERATURES:
            nll_test = self.nll_test[alpha]
            nll_oracle = self.nll_oracle[alpha]
            final_obj = nll_oracle + nll_test
            alpha = int(alpha * 100)
            print_and_log_scalar(writer, 'eval_more_t/nll_oracle', nll_oracle, alpha)
            print_and_log_scalar(writer, 'eval_more_t/nll_test',   nll_test,  alpha)
            print_and_log_scalar(writer, 'eval_more_t/final_obj',  final_obj, alpha)
            print('')
            
    def __call__(self):
        print('processing model %s' % self.name)    
        self.get_trained_models()
        self.eval_gen()
        self.log()


if __name__ == '__main__':

    """ Cross-Validating on NLL_{oracle + test} """
    best_mle = Model_eval('best_mle', {'num_layers_gen' : 1, 
                                       'var_dropout_p_gen' : 0.6, 
                                       'batch_size' : 256, 
                                       'gen_lr' : 0.0005, 
                                       'hidden_dim_gen' : 512, 
                                       'mle_epochs' : 300}, 165)
    
    best_gan = Model_eval('best_gan', {'var_dropout_p_gen' : 0.5, 
                                       'var_dropout_p_disc' : 0.2, 
                                       'batch_size' : 64, 
                                       'gen_lr' : 0.0005, 
                                       'disc_lr' : 5e-5, 
                                       'mle_epochs' : 40, 
                                       'disc_pretrain_epochs' : 1, 
                                       'disc_train_iterations' : 20, 
                                       'gen_train_iterations' : 1, 
                                       'mle_train_iterations' : 0, 
                                       'hidden_dim_gen' : 256, 
                                       'hidden_dim_disc' : 256, 
                                       'seqgan_reward' : 1, 
                                       'num_layers_gen' : 1, 
                                       'num_layers_disc' : 1}, 299) 

    best_gan_mle = Model_eval('best_gan_mle', 
                                      {'var_dropout_p_gen' : 0.6, 
                                       'var_dropout_p_disc' : 0.3, 
                                       'batch_size' : 64, 
                                       'gen_lr' : 0.0005, 
                                       'disc_lr' : 5e-5, 
                                       'mle_epochs' : 80, 
                                       'disc_pretrain_epochs' : 16, 
                                       'disc_train_iterations' : 10, 
                                       'gen_train_iterations' : 1, 
                                       'mle_train_iterations' : 1, 
                                       'hidden_dim_disc' : 512, 
                                       'seqgan_reward' : 0, 
                                       'num_layers_gen' : 1}, 299) 
    
    # gan_VDGEN0.6_VDDISC0.2_BS512_GLR0.0005_DLR0.0005_MLE80_DE40_DTI5_GTI1_MTI0_HD256_SQ20_ats1_beta2.0_SQR0_LEAK1 
    best_gan_beta = Model_eval('best_gan_beta',
                                      {'var_dropout_p_gen' : 0.6, 
                                       'var_dropout_p_disc' : 0.2, 
                                       'batch_size' : 512, 
                                       'gen_lr' : 0.0005, 
                                       'disc_lr' : 0.0005, 
                                       'mle_epochs' : 80, 
                                       'disc_pretrain_epochs' : 40, 
                                       'disc_train_iterations' : 5, 
                                       'gen_train_iterations' : 1, 
                                       'mle_train_iterations' : 0, 
                                       'hidden_dim_disc' : 256, 
                                       'hidden_dim_gen' : 256, 
                                       'seqgan_reward' : 0, 
                                       'leak_info' : 1, 
                                       'transfer_weights_after_pretraining' : 0,
                                       'beta' : 2,
                                       'num_layers_gen' : 1}, 280)

                
    """ Cross-Validating on NLL_{oracle} only """
    
    # gan_VDGEN0.4_VDDISC0.5_BS512_GLR0.001_DLR0.0005_MLE10_DE1_DTI1_GTI1_MTI0_HD512_SQ20_ats1_beta0_SQR0_LEAK0 
    best_gan_cvo = Model_eval('best_gan_cvo', 
                                      {'var_dropout_p_gen' : 0.4, 
                                       'var_dropout_p_disc' : 0.5, 
                                       'batch_size' : 512, 
                                       'gen_lr' : 0.001, 
                                       'disc_lr' : 0.0005, 
                                       'mle_epochs' : 10, 
                                       'disc_pretrain_epochs' : 1, 
                                       'disc_train_iterations' : 1, 
                                       'gen_train_iterations' : 1, 
                                       'mle_train_iterations' : 0, 
                                       'hidden_dim_disc' : 512,
                                       # default gen_size is 512, so no need to rerun 
                                       'seqgan_reward' : 0, 
                                       'num_layers_gen' : 1, 
                                       'num_layers_disc' : 1}, 100) 

    # gan_VDGEN0.6_VDDISC0.3_BS64_GLR0.0005_DLR5e-05_MLE80_DE16_DTI10_GTI1_MTI1_HD512_SQ20_ats1_beta0_SQR0_LEAK0
    # actually the same as the other gan_mle, so we won't rerun it.
    #best_gan_mle_cvo = Model_eval('best_gan_mle_cvo', 
    #                                  {'var_dropout_p_gen' : 0.6, 
    #                                   'var_dropout_p_disc' : 0.3, 
    #                                   'batch_size' : 64, 
    #                                   'gen_lr' : 0.0005, 
    #                                   'disc_lr' : 5e-5, 
    #                                   'mle_epochs' : 80, 
    #                                   'disc_pretrain_epochs' : 16, 
    #                                   'disc_train_iterations' : 10, 
    #                                   'gen_train_iterations' : 1, 
    #                                   'mle_train_iterations' : 1, 
    #                                   'hidden_dim_disc' : 512, 
    #                                  # default gen_size is 512, so no need to rerun
    #                                   'seqgan_reward' : 0, 
    #                                   'num_layers_gen' : 1}, 299) 

    # mle_LY1_VDGEN0.4_BS128_GLR0.0005_HD512_SQ20
    best_mle_cvo = Model_eval('best_mle_cvo',
                                      {'num_layers_gen' : 1, 
                                       'var_dropout_p_gen' : 0.4, 
                                       'batch_size' : 128, 
                                       'gen_lr' : 0.0005, 
                                       'hidden_dim_gen' : 512, 
                                       'mle_epochs' : 300}, 300)

    # gan_VDGEN0.5_VDDISC0.2_BS256_GLR0.001_DLR0.0005_MLE40_DE20_DTI1_GTI1_MTI0_HD512_SQ20_ats1_beta1.0_SQR0_LEAK1 
    best_gan_beta_cvo = Model_eval('best_gan_beta_cvo2',  
                                      {'var_dropout_p_gen' : 0.5, 
                                       'var_dropout_p_disc' : 0.2, 
                                       'batch_size' : 256, 
                                       'gen_lr' : 0.001, 
                                       'disc_lr' : 0.0005, 
                                       'mle_epochs' : 40, 
                                       'disc_pretrain_epochs' : 20, 
                                       'disc_train_iterations' : 1, 
                                       'gen_train_iterations' : 1, 
                                       'mle_train_iterations' : 0, 
                                       'hidden_dim_disc' : 512,
                                       'hidden_dim_gen' : 512, 
                                       'seqgan_reward' : 0, 
                                       'leak_info' : 1, 
                                       'transfer_weights_after_pretraining' : 0,
                                       'beta' : 1.0, 
                                       'num_layers_gen' : 1, 
                                       'num_layers_disc' : 1}, 150)

    """ MLE CROSS VALIDATED ON NLL_TEST AS SHOULD ALWAYS BE DONE """
    # mle_LY2_VDGEN0.6_BS256_GLR0.001_HD128_SQ20
    best_mle_cvt = Model_eval('best_mle_cvt', 
                                      {'num_layers_gen' : 2, 
                                       'var_dropout_p_gen' : 0.6, 
                                       'batch_size' : 256, 
                                       'gen_lr' : 0.001, 
                                       'hidden_dim_gen' : 128, 
                                       'mle_epochs' : 300}, 139)

    #COT_VDGEN0.3_VDDISC0.2_BS128_GLR0.0005_DLR0.0005_MLE0_DE0_DTI1_GTI1_MTI0_HDG512_HDD1024
    best_cot_cvt = Model_eval('best_cot_cvt_fixed', 
                                        {'cot'                : 1,
                                         'var_dropout_p_gen'  : 0.3, 
                                         'var_dropout_p_disc' : 0.2, 
                                         'batch_size'         : 128, 
                                         'gen_lr'             : 0.0005, 
                                         'disc_lr'            : 0.0005, 
                                         'mle_epochs'         : 0,
                                         'adv_epochs'         : 300,
                                         'disc_train_iterations' : 1, 
                                         'hidden_dim_gen'     : 512, 
                                         'hidden_dim_disc'    : 1024, 
                                         'num_layers_gen'     : 1, 
                                         'num_layers_disc'    : 1,
                                         'transfer_weights_after_pretraining' : 0}, 33) 
                                        
    args = get_train_args()
    # models = [best_gan_but_volatile, best_mle, best_gan, best_gan_mle]
    # models = [best_gan_beta, best_gan_beta_cvo]
    # models = [best_mle_cvt, best_gan, best_gan_cvo, best_gan_mle, best_gan_beta, best_gan_beta_cvo]
    models = [best_cot_cvt] 
    for model in models:
        model()

