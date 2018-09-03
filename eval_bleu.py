import argparse
import pdb
import numpy as np
import torch
import torch.utils.data
import tensorboardX
from collections import OrderedDict as OD
from PIL import Image
import matplotlib; matplotlib.use('Agg')

from utils  import * 
from data   import * 
from models import * 
from args   import * 

args  = get_test_args()

# reproducibility
torch.manual_seed(2)
np.random.seed(2)

# dataset creation
dataset_train, word_dict = tokenize(os.path.join(args.data_dir, 'train.txt'), \
        train=True, char_level=args.character_level)
dataset_test,  word_dict = tokenize(os.path.join(args.data_dir, 'valid.txt'), \
        train=False, word_dict=word_dict, char_level=args.character_level)

# fetch one minibatch of data
train_batch = next(minibatch_generator(dataset_train, args, shuffle=False))
test_batch  = next(minibatch_generator(dataset_test,  args, shuffle=False))

# load model that will be evaluated
gen, loaded_epoch = load_model_from_file(args.model_path, epoch=args.model_epoch)
gen.args.alpha_test = args.alpha_test
gen.eval()
print('switching the temperature to {}'.format(gen.args.alpha_test))

# Logging
writer = tensorboardX.SummaryWriter(log_dir=os.path.join(args.model_path, \
        'TB_alpha{}'.format(gen.args.alpha_test)))
writes = 0

if args.cuda: 
    gen  = gen.cuda()

def save_samples_for_bleu(gen, input, word_dict, epoch, sample_size=10000):
    file_name = os.path.join(args.model_path, 'samples/gen_for_bleu_{}.txt'.format(epoch))
    with torch.no_grad():
        with open(file_name, 'w') as f:
            tot_sent=0
            while tot_sent < sample_size:
                _, fake_sentences = gen(input[:, [0]])
                sentences = id_to_words(fake_sentences.cpu().data.numpy(), word_dict)
                for sentence in sentences: 
                    xx = str(sentence) #[1:-1]
                    xx = xx.replace('\n', '')
                    f.write(xx + '\n')
                    tot_sent +=1
                    if tot_sent >= sample_size: return
        

from metrics import Bleu, SelfBleu
input = train_batch[0][:128]
save_samples_for_bleu(gen, input, word_dict, 'final')
bleu5 = Bleu(test_text=os.path.join(*[args.model_path,'samples','gen_for_bleu_final.txt']),
             real_text=os.path.join(args.data_dir,'valid.txt'),
             num_real_sentences=10000,
             num_fake_sentences=10000,
             gram=5).get_score()
print_and_log_scalar(writer, 'test/bleu5', bleu5, writes)      
sbleu5 = SelfBleu(test_text=os.path.join(*[args.model_path,'samples','gen_for_bleu_final.txt']),
                  num_sentences=10000,
                  gram=5).get_score()
print_and_log_scalar(writer, 'test/sbleu5', sbleu5, writes, end_token='\n')      


