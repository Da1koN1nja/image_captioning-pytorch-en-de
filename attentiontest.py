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

def good_counter(bleu, wer, value_bleu, value_wer): 
    if 0 <= bleu and bleu < 0.1:
        value_bleu[0] += 1
    elif bleu < 0.2:
        value_bleu[1] += 1
    elif bleu < 0.3:
        value_bleu[2] += 1
    elif bleu < 0.4:
        value_bleu[3] += 1
    elif bleu < 0.5:
        value_bleu[4] += 1
    elif bleu < 0.6:
        value_bleu[5] += 1
    elif bleu < 0.7:
        value_bleu[6] += 1
    elif bleu < 0.8:
        value_bleu[7] += 1
    elif bleu < 0.9:
        value_bleu[8] += 1
    elif bleu <= 1:
        value_bleu[9] += 1
    else:
        value_bleu[10] += 1
    
    if 0 <= wer and wer < 0.1:
        value_wer[0] += 1
    elif wer < 0.2:
        value_wer[1] += 1
    elif wer < 0.3:
        value_wer[2] += 1
    elif wer < 0.4:
        value_wer[3] += 1
    elif wer < 0.5:
        value_wer[4] += 1
    elif wer < 0.6:
        value_wer[5] += 1
    elif wer < 0.7:
        value_wer[6] += 1
    elif wer < 0.8:
        value_wer[7] += 1
    elif wer < 0.9:
        value_wer[8] += 1
    elif wer <= 1:
        value_wer[9] += 1
    else:
        value_wer[10] += 1
    
    
    
def main():
   
    config = Config()
    config.mode = 'val'
    config.loader = False    
    transform_2 = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                            (0.229, 0.224, 0.225))])
    
    
    transform = transforms.Compose([ 
        transforms.RandomCrop(config.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    
    lang = config.lang
    high_images = []
    low_images = []
    value_bleu = np.zeros(11)
    value_wer = np.zeros(11)
    
    encoder_path = 'models/best_attention_encoder--{}-lang--{}.ckpt'.format(config.cnn_architecture, lang)
    decoder_path = 'models/best_attention_decoder--{}-lang--{}.ckpt'.format(config.rnn_architecture, lang)
   
    image_dir = config.test_set   
    
    caption_file = "AttentionTestSentences--{}--{}-{}.txt".format(lang, config.decoder_type, config.top_k)
    
    open(caption_file, "w+").close()
    
    test_table = pd.read_csv("./data/test{}data.csv".format(lang), encoding="utf8")
    smooth = SmoothingFunction()    
    
    bleu_max = 0
    if lang == 'de':  
        vocab_path = 'data/clean_vocab_de.pkl'
        caption_dir = config.de_test_cap   
        print("Starting Validation of German Captions")
    else:
        vocab_path = 'data/clean_vocab_en.pkl'
        caption_dir = config.en_test_cap           
        print("Starting Testing of English Captions")
    
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    test_list = []
    test_split = "./data/splits/test_images.txt"
    fn = test_split
    with open(fn) as f:
        for line in f:
            test_list.append(os.path.splitext(line.strip())[0])
                
    
    show_val = config.show_val
    
    encoder = CNNEncoder(config).eval()
    decoder = AttentionRNNDecoder(config, len(vocab)).eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    loaded_encoder = torch.load(encoder_path)
    loaded_decoder = torch.load(decoder_path)
    
    encoder.load_state_dict(loaded_encoder)
    decoder.load_state_dict(loaded_decoder)
    

    ids = test_table['image_id'].values
    captions = test_table['caption'].values
   
    lowish = 0
    ref = {}
    ref1 = {}
    refs = []*len(test_list)
    test = []*len(test_list)
    test_ids = []*len(test_list)
    test_lengths = []*len(test_list)    
    j = 0
    for i, idss in enumerate(ids):
        caption = nltk.tokenize.word_tokenize(captions[i].lower())        
       
        if idss in ref:
            ref[idss].append(caption)
        else:
            ref[idss] = [caption, ]   
        
            
    for i, idss in enumerate(test_list):
        refs.append(ref[int(idss)])    
    start = vocab.word2id['<start>']
    end = vocab.word2id['<end>']
  
    
    
    time_bef = datetime.now()
    
    if config.loader is False:
        for i, testss in enumerate(tqdm(test_list)):            
            image_path = image_dir + "/" + testss + ".jpg"
            image = load_image(config, os.path.join(image_path), transform_2)
            
            image_tensor = image.to(device)   
            feature = encoder(image_tensor)

            sampled_ids = decoder.sample(feature, config.top_k, start, end)
            sampled_caption = []            
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
            test.append(sentence_temp)
            bleu_tot = sentence_bleu(refs[i], sentence_temp, smoothing_function=smooth.method4)
            wer_tot = system_wer_score([refs[i]], [sentence_temp])
            good_counter(bleu_tot, wer_tot, value_bleu, value_wer)                
            with open(caption_file, "a") as myfile:
               
                myfile.write(testss + ": " + sentence + "\n")
           
            if bleu_tot > bleu_max:
                bleu_max = bleu_tot
            if bleu_tot > 0.7:
                high_images.append(testss)            
            
            if bleu_tot < 0.3:                             
                lowish += 1
                if lowish < 50:
                    img = plt.imread(os.path.join(image_path))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(sentence_plt)
                    directory = os.path.dirname('./data/test_results/attention/{}/low_bleu/{}/{}/{}/'.format(config.decoder_type, config.lang, config.cnn_architecture, config.rnn_architecture))
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    plt.savefig(os.path.join(directory,
                                             testss+ '_result.jpg'))                               
                               
            if bleu_tot > 0.7:
                img = plt.imread(os.path.join(image_path))
                plt.imshow(img)
                plt.axis('off')
                plt.title(sentence_plt)                
                directory = os.path.dirname('./data/attention/{}/high_bleu/{}/{}/{}/'.format(config.decoder_type, config.lang, config.cnn_architecture, config.rnn_architecture))
                if not os.path.exists(directory):
                    os.makedirs(directory)
                plt.savefig(os.path.join(directory, testss +'_result.jpg'))
    else:    
        for i, (img_ids, images, captions, lengths) in enumerate(tqdm(data_loader)):

            images = images.to(device)
           
            features = encoder(images)
            
            sampled_ids = decoder.sample(features, config.top_k, start, end)

            
            sampled_caption = []
                
            for s in range(len(sampled_ids)):
                
                sampled_caption = []
                s_ids = sampled_ids[s].cpu().numpy()
                for word_id in s_ids:

                    word = vocab.id2word[word_id]
                    sampled_caption.append(word)
                    if word == '<end>':
                        break


                sentence = ' '.join(sampled_caption)
             
                sentence_temp = nltk.tokenize.word_tokenize(sentence)
                sentence_temp = sentence_temp[3:-3]
                sentence_plt = ' '.join(sentence_temp)
                test.append(sentence_temp)
                
                if show_val is True:
                
                    img = plt.imread(os.path.join(config.val_image_dir + "/" + str(img_ids[s]) + ".jpg"))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(sentence_plt)      
                    plt.savefig(os.path.join("./data/test_results/",
                                                     str(img_ids[s]) +'_attention_result.jpg'),)

                    if i == config.show_size:
                        show_val = False
                   
      
    
    bleu1 = corpus_bleu(refs, test, (1, 0, 0, 0), smoothing_function=smooth.method4)
    bleu2 = corpus_bleu(refs, test, (0.5, 0.5, 0, 0), smoothing_function=smooth.method4)
    bleu3 = corpus_bleu(refs, test, (0.33, 0.33, 0.33, 0), smoothing_function=smooth.method4)
    bleu4 = corpus_bleu(refs, test, smoothing_function=smooth.method4)        
    
    wer = system_wer_score(refs, test) 
   
    
    test_result_file = "./data/test_results/test_Language_{}--CNN_{}--RNN_{}--{}-{}.txt".format(lang, config.cnn_architecture, config.rnn_architecture, config.decoder_type, config.top_k)
    
    open(test_result_file, "w+").close() 
    
    with open(test_result_file, "a") as testfile:
                     
            testfile.write("BLEU-1 Score of the System: " + str(bleu1) + "\n")
            testfile.write("BLEU-2 Score of the System: " + str(bleu2) + "\n")
            testfile.write("BLEU-3 Score of the System: " + str(bleu3) + "\n")
            testfile.write("BLEU-4 Score of the System: " + str(bleu4) + "\n")
            testfile.write("WER of the system: " + str(wer) + "\n")
            testfile.write("Word accuracy of the system: " + str(1-wer) + "\n")    
   
    print("BLEU-1 Score of the System: " + str(bleu1))
    print("BLEU-2 Score of the System: " + str(bleu2))
    print("BLEU-3 Score of the System: " + str(bleu3))
    print("BLEU-4 Score of the System: " + str(bleu4))   
    
    print("WER of the system: " + str(wer))
    print("Word accuracy of the system: " + str(1-wer))
    print("High Captioning Quality")
    print(len(high_images))
    print("Low Captioning Quality")    
    print(lowish)
    print("Range of BLEU scores for images")
    print(value_bleu)
    print("Range of WER scores for images")
    print(value_wer)
   
    
    time_after = datetime.now()      
    time_taken = (time_after - time_bef)
    print("Time to test model")
    print(time_taken)      
if __name__ == '__main__':    
    main()
