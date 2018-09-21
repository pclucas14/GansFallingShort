import os
import pdb
import numpy as np
import re
import torch
from torch.autograd import Variable
import _pickle as pickle

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.vocab_set = set()

        # add <unk> <sos> and <eos> tokens
        # really important not to change (hardcoded in minibatch_generator)
        self.add_word(u'<pad>')  # ID 0
        self.add_word(u'<eos>')  # ID 1
        self.add_word(u'<sos>')  # ID 2
        self.add_word(u'<unk>')  # ID 3

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

def tokenize(path, train=False, word_dict=None, char_level=False, dataset=None, skip=False):
    # tokenizing process is somewhat lenghty. Let's try to avoid 
    # it when possible
    if not skip:
        try:
            path_word_dict = path + '_word_dict.pickle'
            path_ids = path + '_ids.pickle'
            with open(path_ids, 'rb') as f: 
                ids = pickle.load(f)
            if train: 
                with open(path_word_dict, 'rb') as f: 
                    word_dict = pickle.load(f)
            
            print('loaded preprocessed data from %s' % path)
            return ids, word_dict
        except: 
            pass

    """Tokenizes a text file."""
    if word_dict is None : 
        print('creating new word dictionary')
        word_dict = Dictionary() 
    assert os.path.exists(path), '{} does not exist'.format(path)
    # Add words to the dictionary
    with open(path, 'r') as f:
        sentences = 0
        max_tokens = 0
        for line in f:
            # line = line.decode('utf-8', 'strict')
            words = re.findall(r"[\w']+|[.,!?;]", line,
                    flags=re.UNICODE) 
            
            if char_level: 
                chars = []
                for word in words: 
                    for cc in word: 
                        chars += [cc]
                    chars += [' ']
            
                # remove last space
                chars = chars[:-1]
                words = chars
            else: 
                if words[-1] == '.':
                    words[-1] = '<eos>'
                elif words[-1] == '?':
                    words[-1] =  '<qm>'
                elif words[-1] == '!':
                    words[-1]  ='<em>'
            
            if dataset=='ptb':
                words += ['<eos>']

                
            # only add words if in training set
            if train:
                for word in words:
                    word_dict.add_word(word)
                word_dict.vocab_set = \
                    set(word_dict.idx2word)

            # track stats for building tokenized version
            tokens = len(words)
            sentences += 1
            if tokens > max_tokens:
                max_tokens = tokens

    # Tokenize file content
    with open(path, 'r') as f:
        ids = []
        for i, line in enumerate(f):
            # line = line.decode('utf-8', 'strict')
            words = re.findall(r"[\w']+|[.,!?;]", line, 
                    flags=re.UNICODE)
            
            if char_level: 
                chars = []
                for word in words: 
                    for cc in word: 
                        chars += [cc]
                    chars += [' ']
            
                # remove last space
                chars = chars[:-1]
                words = chars
            else: 
                if words[-1] == '.':
                    words[-1] = '<eos>'
                elif words[-1] == '?':
                    words[-1] =  '<qm>'
                elif words[-1] == '!':
                    words[-1]  ='<em>'

            if dataset=='ptb':
                words += ['<eos>']

            token = 0
            idx = list(range(len(words)))
            for word in words:
                if word not in word_dict.vocab_set:
                    word = u'<unk>'
                idx[token] = word_dict.word2idx[word]
                token += 1

            # create list of lists for easier process later on
            ids.append(idx)

    # save to file 
    path_word_dict = path + '_word_dict.pickle'
    path_ids = path + '_ids.pickle'
    with open(path_ids, 'wb') as f: 
        pickle.dump(ids, f)
    if train: 
        with open(path_word_dict, 'wb') as f: 
            pickle.dump(word_dict, f)
    
    return ids, word_dict
    

if __name__ == '__main__':
    path = 'data/news/train.txt'
    ids, word_dict = tokenize(path, train=True)

