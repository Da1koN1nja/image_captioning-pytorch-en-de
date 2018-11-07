import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
import pandas as pd
from PIL import Image
from vocabulary import Vocab
from pycocotools.coco import COCO
from config import Config

class Flickr30kValDataset(data.Dataset):
 
    def __init__(self, root, csv, vocab, lang, transform=None):
       
        self.root_fl = root
        
        if lang == "de":
            
            self.anno_fl = pd.read_csv(csv, encoding="utf8")
        else:
            self.anno_fl = pd.read_csv(csv, encoding="utf8")
        self.ids_fl = self.anno_fl['image_id'].values
        self.vocab_fl = vocab
        self.transform = transform

    def __getitem__(self, index): 
        vocab = self.vocab_fl
        ann_id = self.ids_fl[index]
        caption = self.anno_fl['caption'][index]
        img_id = self.anno_fl['image_id'][index]
        path = self.anno_fl['image_file'][index]
        
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)        
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())       
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)      
        return img_id, image, target

    def __len__(self):
        return len(self.ids_fl)

class Flickr30kDataset(data.Dataset):
    
    def __init__(self, root, csv, vocab, lang, transform=None):
        
        self.lang = lang
        self.root_fl = root      
        if lang == "de":            
            self.anno_fl = pd.read_csv(csv, encoding="utf8")
        else:
            self.anno_fl = pd.read_csv(csv, encoding="utf8")
        self.ids_fl = self.anno_fl['image_id'].values
        self.vocab_fl = vocab
        self.transform = transform

    def __getitem__(self, index):       
        vocab = self.vocab_fl
        ann_id = self.ids_fl[index]
        caption = self.anno_fl['caption'][index]
        img_id = self.anno_fl['image_id'][index]
        path = self.anno_fl['image_file'][index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)      
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
   
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        
        return image, target

    def __len__(self):
        return len(self.ids_fl)
    
def collate_fn(data):       
    data.sort(key=lambda x: len(x[1]), reverse=True)    
    images, captions = zip(*data)       
    images = torch.stack(images, 0) 
    lengths = [len(cap) for cap in captions]
        
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]    
    return images, targets, lengths

def collate_fn_no(data):        
    img_id, images, captions = zip(*data) 
    images = torch.stack(images, 0)    
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return img_id, images, targets, lengths



def get_loader_flickr(root, csv, vocab, lang, transform, batch_size, shuffle, num_workers):    
    flickr = Flickr30kDataset(root=root,
                       csv=csv,
                       vocab=vocab, lang=lang,
                       transform=transform)   
    data_loader = torch.utils.data.DataLoader(dataset=flickr, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)   
    return data_loader

def get_loader_flickr_val(root, csv, vocab, lang, transform, batch_size, shuffle, num_workers):
    
   
    flickrval = Flickr30kValDataset(root=root,
                       csv=csv,
                       vocab=vocab, lang=lang,
                       transform=transform)    
    data_loader = torch.utils.data.DataLoader(dataset=flickrval, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn_no)  
    return data_loader
