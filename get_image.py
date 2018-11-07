import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
import pandas as pd
import nltk
import shutil
from torchvision import transforms 
from datetime import datetime

from config import Config
#from resize import 
from vocabulary import Vocab
from attentionmodel import CNNEncoder, AttentionRNNDecoder

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from dp_align import system_wer_score
from os import path
from data_loader import get_loader_flickr, get_loader_flickr_val
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(config, image_path, transform=None):
    image = Image.open(image_path)      
    image = image.resize([config.crop_size, config.crop_size], Image.LANCZOS)
    if transform is not None:        
        image = transform(image).unsqueeze(0)
       
    return image

    
    
def main(args):
    
    config = Config()
    

    transform_2 = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                            (0.229, 0.224, 0.225))])
    
        
    lang = args.lang    
    
    encoder_path = args.encoder_path
    decoder_path = args.decoder_path
     
   
    if lang == 'de':  
        vocab_path = 'data/clean_vocab_de.pkl'        
    else:
        vocab_path = 'data/clean_vocab_en.pkl'        
  
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    encoder = CNNEncoder(config).eval() 
    decoder = AttentionRNNDecoder(config, len(vocab)).eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    loaded_encoder = torch.load(encoder_path)
    loaded_decoder = torch.load(decoder_path)
    
    encoder.load_state_dict(loaded_encoder)
    decoder.load_state_dict(loaded_decoder)
    
   
    start = vocab.word2id['<start>']
    end = vocab.word2id['<end>']
     
    
                  
    image_path = args.image
    image = load_image(config, os.path.join(image_path), transform_2)            
    image_tensor = image.to(device)            
     
    feature = encoder(image_tensor)

    sampled_ids = decoder.sample(feature, config.top_k, start, end)
    sampled_caption = []          
    print(sampled_ids)
    sampled_ids = sampled_ids[0].cpu().numpy()         
    for word_id in sampled_ids:
        word = vocab.id2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)           
    sentence_temp = nltk.tokenize.word_tokenize(sentence)
    sentence_temp = sentence_temp[3:-3]
    sentence_plt = ' '.join(sentence_temp)
    
                 
    user = path.splitext(image_path.strip())[0] 
    
    img = plt.imread(os.path.join(image_path))
    plt.imshow(img)
    plt.axis('off')
    plt.title(sentence_plt)                
    directory = os.path.dirname('./user_images/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, 'user_result.jpg'))              
        
    print(sentence_plt)
    

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')    
    parser.add_argument('--lang', type=str, default='en', help='what language the user wants the image to be generated')
    parser.add_argument('--encoder_path', type=str, default='models/best_attention_encoder--resnet50-lang--en.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/best_attention_decoder--lstm-lang--en.ckpt', help='path for trained decoder')
    args = parser.parse_args()
    main(args)
