import numpy as np
import nltk
from collections import Counter
import argparse
import pandas as pd
import pickle
import os
from os import path
from config import Config

class Vocab(object):
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.id = 0
        
    def add_word(self, word):
        if not word in self.word2id:
            self.word2id[word] = self.id
            self.id2word[self.id] = word
            self.id += 1
            
    def __call__(self, word):
        if not word in self.word2id:
            return self.word2id['<unk>']
        return self.word2id[word]
    
    def __len__(self):
        return len(self.word2id)
    
def build_vocab(config, threshold):   
       
    
    data_path = path.join(args.data_frame_path, "train{}data.csv".format(config.lang))
     
    annotations = pd.read_csv(data_path, encoding="utf8")
    
    captions = annotations['caption'].values
    image_ids = annotations['image_id'].values
    image_files = annotations['image_file'].values
    
    counter = Counter()

    for i, caption in enumerate(captions):
        
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(captions)))    
    
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocab()

    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for i, word in enumerate(words):     
        vocab.add_word(word)
    return vocab

def main(args):
    config = Config()    
    config.lang = args.lang
    vocab_path = path.join(args.data_frame_path, "clean_vocab_{}.pkl".format(config.lang))   
    vocab = build_vocab(config, args.threshold)
    
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)   
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()   
    parser.add_argument('--threshold', type=int, default=1, 
                        help='minimum word count threshold')
    parser.add_argument('--lang', type=str, default='en', 
                        help='minimum word count threshold')
    parser.add_argument('--data_frame_path', type=str, default='./data/', 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
