import argparse
import pdb
import numpy as np
import torch
import torch.utils.data
import tensorboardX
from collections import OrderedDict as OD
from PIL import Image
import matplotlib; matplotlib.use('Agg')

from tsne import compute_tsne
from tsne_utils import create_matrix_for_tsne

from utils  import * 
from data   import * 
from models import * 
from losses import * 
from args   import * 

args  = get_test_args()

# reproducibility
torch.manual_seed(2)
np.random.seed(2)

# dataset creation
if args.debug: # --> allows for faster iteration when coding 
    dataset_train, word_dict = tokenize(os.path.join(args.data_dir, 'valid.txt'), \
        train=True, char_level=args.character_level)
    dataset_test = dataset_train
else: 
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
# maybe_create_dir(os.path.join(args.model_path, 'eval/%s_epoch' % loaded_epoch)) # TODO: maybe put in TB directly ?
writer = tensorboardX.SummaryWriter(log_dir=os.path.join(args.model_path, 'TB_alpha{}'.format(gen.args.alpha_test)))
writes = 0

if args.lm_path: 
    oracle_lm = load_model_from_file(args.lm_path, epoch=args.lm_epoch)[0]
    oracle_lm.eval()

if args.cuda: 
    gen  = gen.cuda()
    if args.lm_path: oracle_lm = oracle_lm.cuda()

# First experiment : log hidden states for T-SNE plots
MODE = [('train', train_batch, OD(), []), ('test', test_batch, OD(), []), ('free_running', test_batch, OD(), [])]

with torch.no_grad():
    for mode, data, hs_dict, oracle_nlls in MODE: 
        input, _, _ = data

        # here we basically expose the model's forward pass to fetch the hidden states efficiently
        teacher_force = mode != 'free_running'
        print('teacher forcing : {}'.format(teacher_force))
        hidden_state, hidden_state_oracle = None, None

        for t in range(args.tsne_max_t):
            if teacher_force or t == 0: 
                input_idx = input[:, [t]]

            input_t = gen.embedding(input_idx)
            output, hidden_state = gen.step(input_t, hidden_state, t)
            
            if args.lm_path: 
                if t > 0: 
                    # query the oracle for NLL of the next word (i.e. use x_t to index p(x_t | x_{i<t})
                    # oracle_nlls += [-1. * oracle_dist.log_prob(input_idx.squeeze()).mean(dim=0).item()]
                    oracle_nll_t = -1. * oracle_dist.log_prob(input_idx.squeeze())
                    oracle_nlls += [remove_pad_tokens(oracle_nll_t, input_idx.squeeze()).item()]
                    full_oracle_nll = oracle_nll_t.view(-1,1) if t==1 else torch.cat((full_oracle_nll,oracle_nll_t.view(-1,1)),1)

                # feed the current word to the model. 
                input_oracle = oracle_lm.embedding(input_idx)
                output_oracle, hidden_state_oracle = oracle_lm.step(input_oracle, \
                        hidden_state_oracle, t)
                oracle_dist = oracle_lm.output_layer(output_oracle)
                oracle_dist = Categorical(logits=oracle_dist.squeeze(1))
           
            # compute entropy (! does not take car of <pad>)
            dist = gen.output_layer(output)
            entropy = Categorical(logits=dist.squeeze(1)).entropy().cpu().numpy().mean()
            print_and_log_scalar(writer, 'eval/%s_entropy' % mode, entropy, t) 

            if not teacher_force: 
                dist = gen.output_layer(output)
                dist *= gen.args.alpha_test
                input_idx = Categorical(logits=dist.squeeze(1)).sample().unsqueeze(1)
                seq = input_idx if t==0 else torch.cat((seq,input_idx), 1)


            if (t+1) % args.tsne_log_every == 0: 
                # for lstm we take the hidden state (i.e. h_t of (h_t, c_t))
                hs = hidden_state[0] if isinstance(hidden_state, tuple) else hidden_state
                hs_dict[t] = hs.cpu().data.numpy()

            if (t+1) % args.oracle_nll_log_every == 0 and args.lm_path and t > 0: 
                p_x_1t = sum(oracle_nlls)
                p_x_t = oracle_nlls[-1]
                #writer.add_scalar('eval/%s_oracle_nll' % mode , p_x_t, t)
                print_and_log_scalar(writer, 'eval/%s_oracle_nll' % mode, p_x_t, t) 

        # print most/less likely sequences
        if teacher_force:
            seq = input[:,1:]
        
        seq_len = (seq != 0).sum(1)
        tot_oracle_nll = full_oracle_nll.sum(1)
        avg_oracle_nll = tot_oracle_nll.cpu().numpy() / seq_len.cpu().numpy()

        sentences = id_to_words(seq.cpu().data.numpy(), word_dict)
        sorted_idx = np.argsort(avg_oracle_nll)
        
        print("most likely sentences under oracle:")
        for i in range(3):
            print(sentences[sorted_idx[i]])
            print("nll oracle: {}".format(avg_oracle_nll[sorted_idx[i]]))
        
        print("least likely sentences under oracle:")
        for i in range(1,4):
            print(sentences[sorted_idx[-i]])
            print("nll oracle: {}".format(avg_oracle_nll[sorted_idx[-i]]))


# -------------------------------------------------------------------------------------
# Evaluating the similarity of hidden states
# -------------------------------------------------------------------------------------

""" processing the data """
timesteps = list(MODE[0][2].keys())

split = int(args.tsne_batch_size * 0.8)
# let's do a train-test split and see if we can train a simple SVM on it
#tf_states    = [MODE[0][2][t] for t in timesteps]
tf_states    = [MODE[1][2][t] for t in timesteps]
fr_states    = [MODE[2][2][t] for t in timesteps]

train_tf_states = [x[:, :split].squeeze() for x in tf_states]
train_fr_states = [x[:, :split].squeeze() for x in fr_states]
test_tf_states  = [x[:, split:].squeeze() for x in tf_states]
test_fr_states  = [x[:, split:].squeeze() for x in fr_states]

labels_train = np.concatenate([np.ones_like(train_tf_states[0][:, 0]),\
                         np.zeros_like(train_fr_states[0][:, 0])])

labels_test  = np.concatenate([np.ones_like(test_tf_states[0][:, 0]),\
                         np.zeros_like(test_fr_states[0][:, 0])])

train_Xs = [np.concatenate([x,y]) for (x,y) in zip(train_tf_states, train_fr_states)]
test_Xs  = [np.concatenate([x,y]) for (x,y) in zip(test_tf_states,  test_fr_states)]

""" 1st model : Linear SVM """
if args.run_svm:
    from sklearn.svm import SVC
    clfs = [SVC() for _ in train_Xs]
    _    = [clf.fit(x,labels_train) for (clf, x) in zip(clfs, train_Xs)]
    accs = [clf.score(x, labels_test) for (clf, x) in zip(clfs, test_Xs)]
    for t, acc in zip(timesteps, accs):
        print_and_log_scalar(writer, 'eval/SVM_test_acc', acc, t)

""" 2nd model : simple NN """
if args.run_nn or args.run_rnn:
    hidden_state_size = train_Xs[0].shape[1]

    def run_epoch(model, X, Y, opt=None):
        train_model = opt is not None
        model.train() if train_model else model.eval()
        accs, losses = [], []
        data = [d for d in zip(X,Y)]
        loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=64)
        
        for (x,y) in loader: 
            x, y = x.cuda(), y.long().cuda()
            pred = model(x)
            loss = F.cross_entropy(pred, y)
            loss = loss.sum(dim=0) / pred.shape[0]

            if train_model: apply_loss(opt, loss)
            
            # calculate acc
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(y.data).cpu().sum()    
            acc = float(correct.item()) / int(x.shape[0])   
        
            losses += [loss.item()]
            accs   += [acc]

        return np.mean(losses), np.mean(accs)
    
    if args.run_nn:
        for i in range(len(train_Xs)): 
            model = nn.Sequential(
                nn.Linear(hidden_state_size, hidden_state_size // 2),
                nn.ReLU(True),
                nn.Linear(hidden_state_size // 2, hidden_state_size // 4),
                nn.ReLU(True), 
                nn.Linear(hidden_state_size // 4, 2)).cuda()
            opt = torch.optim.Adam(model.parameters())

            for ep in range(100):
                train_loss, train_acc = run_epoch(model, train_Xs[0], labels_train, opt=opt)
                test_loss,  test_acc  = run_epoch(model, test_Xs[0], labels_test)

                if ep % 5 == 0 : 
                    print_and_log_scalar(writer, 'eval/NN_test_acc_t=%d'   \
                        % timesteps[i], test_acc, ep)
                    print_and_log_scalar(writer, 'eval/NN_train_acc_t=%d'  \
                        % timesteps[i], train_acc, ep)
                    print_and_log_scalar(writer, 'eval/NN_test_loss_t=%d'  \
                        % timesteps[i], test_loss, ep)
                    print_and_log_scalar(writer, 'eval/NN_train_loss_t=%d' \
                        % timesteps[i], train_loss, ep)
        

""" 3rd model : RNN on the hidden state sequences """
if args.run_rnn:
     assert args.tsne_log_every == 1, 'states are not from a continuous sequence!'

     class RNN(nn.Module):
         def __init__(self):
             super(RNN, self).__init__()
             self.lstm = nn.LSTM(hidden_state_size, hidden_state_size, num_layers=2, batch_first=True)
             self.out = nn.Linear(hidden_state_size, 2)

         def forward(self, x):
             last_hs = self.lstm(x)[0][:, -1]
             return self.out(last_hs)

     train_X, test_X = [np.stack(x, axis=1) for x in [train_Xs, test_Xs]]
     model = RNN().cuda()
     opt = torch.optim.Adam(model.parameters(), lr=1e-4)

     for ep in range(250):
         train_loss, train_acc = run_epoch(model, train_X, labels_train, opt=opt)
         test_loss,  test_acc  = run_epoch(model, test_X, labels_test)

         if ep % 10 == 0 : 
             print_and_log_scalar(writer, 'eval/RNN_test_acc'  , test_acc, ep)
             print_and_log_scalar(writer, 'eval/RNN_train_acc' , train_acc, ep)
             print_and_log_scalar(writer, 'eval/RNN_test_loss' , test_loss, ep)
             print_and_log_scalar(writer, 'eval/RNN_train_loss', train_loss, ep)
     

""" finally, create T-SNE plots of hidden states """
if args.run_tsne: 
    for t in timesteps:
        X, y = create_matrix_for_tsne(MODE,t)
        distances, image = compute_tsne(X, y, t, args)
        writer.add_image('eval/tsne-plot', image, t)
    
        # also backup as a separate image
        img_path = os.path.join(os.path.join(args.model_path, \
            'TB_tnse{}'.format(args.n_iter)), 'tsne-plot_%d.png' % t)
        Image.fromarray(image).save(img_path)

        for i in range(distances.shape[0]):
            for j in range(i + 1, distances.shape[1]):
                writer.add_scalar('eval/distance_centroids%d-%d' % (i, j), distances[i,j], t)


