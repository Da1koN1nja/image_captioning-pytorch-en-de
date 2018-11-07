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
from tqdm import tqdm
from googletrans import Translator
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from dp_align import system_wer_score
from os import path
#from bleu import get_bleu

from data_loader import get_loader_flickr, get_loader_flickr_val



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_fn = path.join(os.path.dirname(__file__), "data", "vgg16_weights.npz")
def load_image(config, image_path, transform=None):
    image = Image.open(image_path)   
    
    img = image.resize([config.image_size, config.image_size], Image.ANTIALIAS)
    #image = image.resize([config.crop_size, config.crop_size], Image.LANCZOS)   
   
    if transform is not None:
        image = transform(img).unsqueeze(0)
        #image = transform(image).unsqueeze(0)
       
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
    
    
    
def val_train(config, encoder_path, decoder_path):
    config.mode = 'val'
    smooth = SmoothingFunction()
    lang = config.lang
    image_dir = config.res_val
    config.decoder_type = 'greedy'
    transform = transforms.Compose([ 
        transforms.RandomCrop(config.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    
    
    if lang == 'de':  
        vocab_path = 'data/clean_vocab_de.pkl'
        model_path = 'models/'        
        val_table = pd.read_csv("./data/valdedata.csv", encoding="utf8")
        caption_file = "AttentionValidationSentencesDe.txt--{}".format(config.decoder_type)       
        
        caption_dir = config.de_val_caption_dir          
    else:
        vocab_path = 'data/clean_vocab_en.pkl'
        model_path = 'models/'
        
        val_table = pd.read_csv("./data/valendata.csv", encoding="utf8")
        caption_file = "AttentionValidationEngSentences--{}.txt".format(config.decoder_type)
        
        caption_dir = config.en_val_caption_dir        
    open(caption_file, "w").close()
    
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    val_list = []
    start = vocab.word2id['<start>']
    end = vocab.word2id['<end>']
    encoder = CNNEncoder(config).eval() 
    decoder = AttentionRNNDecoder(config, len(vocab)).eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    loaded_encoder = torch.load(os.path.join(
                    model_path,encoder_path))
    loaded_decoder = torch.load(os.path.join(
                    model_path,decoder_path))

    encoder.load_state_dict(loaded_encoder)
    decoder.load_state_dict(loaded_decoder)
    
    data_loader = get_loader_flickr_val(image_dir, caption_dir, vocab, lang,transform, config.sample_batch_size, shuffle=False, num_workers=config.num_workers) 
    
    ids = val_table['image_id'].values
    captions = val_table['caption'].values
    
    yyy = 0
    ref = {}
    ref1 = {}
    refs = []*len(val_list)
    test = []*len(val_list)
    j = 0
    for i, idss in enumerate(ids):
        caption = nltk.tokenize.word_tokenize(captions[i].lower())        
        if idss in ref:
            ref[idss].append(caption)
        else:
            ref[idss] = [caption, ]       
    val_split = "./data/splits/val_images.txt"
    fn = val_split
    with open(fn) as f:
        for line in f:
            val_list.append(os.path.splitext(line.strip())[0])    
    for i, idss in enumerate(val_list):
        refs.append(ref[int(idss)])
    
    time_bef = datetime.now()
   
    if config.loader is True:
        for i, (img_id, images, captions, lengths) in enumerate(data_loader):
            
            images = images.to(device)
            captions = captions.to(device)
            lengths = lengths   
    
            feature = encoder(images)

            sampled_ids = decoder.sample(feature, config.top_k, start, end)

            sampled_ids = sampled_ids.cpu().numpy()
           
            for i in range(len(sampled_ids)):
                sampled_caption = []
                current_sample = sampled_ids[i]            
                for word_id in current_sample:
                    word = vocab.id2word[word_id]
                    sampled_caption.append(word)
                    if word == '<end>':
                        break
                sentence = ' '.join(sampled_caption)
                
                sentence_temp = nltk.tokenize.word_tokenize(sentence)
                sentence_temp = sentence_temp[3:-3]
                sentence_plt = ' '.join(sentence_temp)
                test.append(sentence_temp)       
    else:
        for i, val in enumerate(tqdm(val_list)):

            image = load_image(config, os.path.join(config.val_image_dir + "/" + val + ".jpg"), transform)            
            image_tensor = image.to(device)            
            bleu_s = 0
            score_s = 0
            max_score = -100000000
            max_length = 0
            min_length = 100
            max_score_id = 0   
            feature = encoder(image_tensor)            
            sampled_ids = decoder.sample(feature, config.top_k, start, end)

           
            sampled_caption = []
            sampled_ids = sampled_ids.cpu().numpy()
      
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
    bleu = corpus_bleu(refs, test, smoothing_function=smooth.method4)    
   
    time_aft = datetime.now()  
    
    
    print("Val Time Taken: {}".format(time_aft-time_bef))
    return bleu

def main():
    
    config = Config()
    config.mode = 'val'    
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

    image_dir = config.res_val 
    val_table = pd.read_csv("./data/val{}data.csv".format(lang), encoding="utf8")
    
    caption_file = "AttentionValidationSentences--{}--{}-{}.txt".format(lang, config.decoder_type, config.top_k)
            
    open(caption_file, "w+").close()
    

        
    smooth = SmoothingFunction()  
    bleu_average = 0
    bleu_max = 0
    if lang == 'de':  
        vocab_path = 'data/clean_vocab_de.pkl'
        caption_dir = config.de_val_caption_dir        
     
        print("Starting Validation of German Captions")
    else:
        vocab_path = 'data/clean_vocab_en.pkl'
        caption_dir = config.en_val_caption_dir          
        
        print("Starting Validation of English Captions")
   
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    val_list = []
    val_split = "./data/splits/val_images.txt"
    fn = val_split
    with open(fn) as f:
        for line in f:
            val_list.append(os.path.splitext(line.strip())[0])
                
    
    show_val = config.show_val

    encoder = CNNEncoder(config).eval() 
    decoder = AttentionRNNDecoder(config, len(vocab)).eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    loaded_encoder = torch.load(encoder_path)
    loaded_decoder = torch.load(decoder_path)
   
    data_loader = get_loader_flickr_val(image_dir, caption_dir, vocab, lang,transform, config.sample_batch_size, shuffle=False, num_workers=config.num_workers)    
    encoder.load_state_dict(loaded_encoder)
    decoder.load_state_dict(loaded_decoder)
    
    ids = val_table['image_id'].values
    captions = val_table['caption'].values    
    
    ref = {}
    ref1 = {}
    refs = []*len(val_list)
    test = []*len(val_list)
    test_ids = []*len(val_list)
    test_lengths = []*len(val_list)
    max_test_length = []*len(val_list)
    min_test_length = []*len(val_list)
    
    best_test_score = []*len(val_list)                                                              
    max_test_score = []*len(val_list)
    j = 0
    for i, idss in enumerate(ids):
        caption = nltk.tokenize.word_tokenize(captions[i].lower())        
       
        if idss in ref:
            ref[idss].append(caption)
        else:
            ref[idss] = [caption, ]          
       
    for i, idss in enumerate(val_list):
        refs.append(ref[int(idss)])
       
    
   
    start = vocab.word2id['<start>']
    end = vocab.word2id['<end>']
  
    
    
    time_bef = datetime.now()
    
    if config.loader is False:
        for i, val in enumerate(tqdm(val_list)):
            
            image = load_image(config, os.path.join(config.val_image_dir + "/" + val + ".jpg"), transform)            
            image_tensor = image.to(device)            
                   
            feature = encoder(image_tensor)

            sampled_ids = decoder.sample(feature, config.top_k, start, end)            

            if config.hacked_val is True and config.decoder_type == 'beam':
                for j in range(len(sampled_ids)):
                    s_ids = []
                    sd = sampled_ids[j][0].cpu().numpy()
                   
                    for word_id in sd:            
                        word = vocab.id2word[word_id]
                        s_ids.append(word)
                        if word == '<end>':
                            break
                    sentence = ' '.join(s_ids)
                    
                    sentence_temp = nltk.tokenize.word_tokenize(sentence)
                    sentence_temp = sentence_temp[3:-3]
                    sentence_plt = ' '.join(sentence_temp)
                    bleu_new = sentence_bleu(refs[i], sentence_temp, smoothing_function=smooth.method4)
                    
                    if bleu_new > bleu_s:
                        
                        best_sentence = sentence_temp
                      
                        best_sentence_id = j
                        best_sentence_length = len(best_sentence)
                        bleu_s = bleu_new
                        score_s = sampled_ids[j][1].cpu().detach().numpy()                                                


                    if len(sentence_temp) > max_length:
                        max_length = len(sentence_temp)

                    if len(sentence_temp) < min_length:
                        min_length = len(sentence_temp)
                       
                    if score_s > max_score:                        
                        max_score = score_s
                        max_score_id = j

                test.append(best_sentence)
                

                with open(id_file, "a") as myfilei:
              
                    myfilei.write(val + ": Best ID: " + str(best_sentence_id) + "/" + str(len(sampled_ids)) +"\n")


                with open(length_file, "a") as myfilel:
                   
                    myfilel.write(val + ": Best Length: " + str(best_sentence_length) + ". Max Length: " + str(max_length) + ". Min Length: " + str(min_length) + "\n")

                with open(score_file, "a") as myfiles:
                   
                    myfiles.write(val + ": Best Score: " + str(score_s) + ". Max Score: " + str(max_score) + ". Max Score ID: " + str(max_score_id) + "\n")
                sentence = ' '.join(best_sentence)

                bestish_sentence = nltk.tokenize.word_tokenize(sentence)       
                bleu_tot = sentence_bleu(refs[i], bestish_sentence, smoothing_function=smooth.method4)
            else:
        
                sampled_caption = []
                sampled_ids = sampled_ids.cpu().numpy()
                
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
                myfile.write(val + ": " + sentence + "\n")
           
            if bleu_tot > bleu_max:
                bleu_max = bleu_tot
            if bleu_tot > 0.7:
                high_images.append(val)
            


            if config.translate is True:  
                
                translate_sentence = translator.translate(sentence_plt, src = 'de').text 
                
                with open(caption_trans_file, "a") as myfile1:
                    myfile1.write(val + ": " + translate_sentence + "\n")
                
            if show_val is True:
              
                img = plt.imread(os.path.join(config.val_image_dir + "/" + val + ".jpg"))
                plt.imshow(img)
                plt.axis('off')
                

                if lang == 'de':  
                    if config.translate is True:
                        plt.title(sentence+'\n'+translate_sentence, fontsize=8)
                    else:
                        if config.hacked_val:
                            plt.title(sentence)
                        else:
                            plt.title(sentence_plt)
                else:
                    if config.hacked_val:
                        plt.title(sentence)
                    else:
                        plt.title(sentence_plt)

                plt.savefig(os.path.join("./data/val_results/",
                                             val+'_attention_result.jpg'),)

                if i == config.show_size:
                    show_val = False

            if bleu_tot > 0.7:
                img = plt.imread(os.path.join(config.val_image_dir + "/" + val + ".jpg"))
                plt.imshow(img)
                plt.axis('off')                
                plt.title(sentence_plt)
                plt.savefig(os.path.join("./data/val_results/topk/",
                                             val+'_{}_attention_result_top_{}_{}_{}.jpg'.format(lang, config.top_k, config.cnn_architecture, config.rnn_architecture)),)
    else:    
        for i, (img_ids, images, captions, lengths) in enumerate(tqdm(data_loader)):

            images = images.to(device)
            
            bleu_s = 0
            score_s = 0
            max_score = -100000000
            max_length = 0
            min_length = 100
            max_score_id = 0
  
            features = encoder(images)
            
            sampled_ids = decoder.sample(features, config.top_k, start, end)

            if config.hacked_val is True and config.decoder_type == 'beam':
                for j in range(len(sampled_ids)):
                    s_ids = []
                    sd = sampled_ids[j][0].cpu().numpy()
                    #print(sd)
                    for word_id in sd:            
                        word = vocab.id2word[word_id]
                        s_ids.append(word)
                        if word == '<end>':
                            break
                    sentence = ' '.join(s_ids)
                 
                    sentence_temp = nltk.tokenize.word_tokenize(sentence)
                    sentence_temp = sentence_temp[3:-3]
                    sentence_plt = ' '.join(sentence_temp)
                    bleu_new = sentence_bleu(refs[i], sentence_temp, smoothing_function=smooth.method4)
                   
                    if bleu_new > bleu_s:                        
                        best_sentence = sentence_temp                       
                        best_sentence_id = j
                        best_sentence_length = len(best_sentence)
                        bleu_s = bleu_new
                        score_s = sampled_ids[j][1].cpu().detach().numpy()                                                


                    if len(sentence_temp) > max_length:
                        max_length = len(sentence_temp)

                    if len(sentence_temp) < min_length:
                        min_length = len(sentence_temp)                       
                    if score_s > max_score:                        
                        max_score = score_s
                        max_score_id = j

                test.append(best_sentence)
                test_ids.append(best_sentence_id)
                best_test_score.append(score_s)
                max_test_score.append(max_score)
                test_lengths.append(best_sentence_length)
                max_test_length.append(max_length)
                min_test_length.append(min_length)


            else:     
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
                        plt.savefig(os.path.join("./data/val_results/",
                                                     str(img_ids[s]) +'_attention_result.jpg'),)

                        if i == config.show_size:
                            show_val = False

                    with open(caption_file, "a") as myfile:    
                        #print(s)
                        #print(ids[s])
                        myfile.write(str(img_ids[s]) + ": " + sentence + "\n")
                        
                        
    bleu1 = corpus_bleu(refs, test, (1, 0, 0, 0), smoothing_function=smooth.method4)
    bleu2 = corpus_bleu(refs, test, (0.5, 0.5, 0, 0), smoothing_function=smooth.method4)
    bleu3 = corpus_bleu(refs, test, (0.33, 0.33, 0.33, 0), smoothing_function=smooth.method4)
    bleu4 = corpus_bleu(refs, test, smoothing_function=smooth.method4)        
 
    wer = system_wer_score(refs, test)     
    val_result_file = "Language_{}--CNN_{}--RNN_{}--{}-{}.txt".format(lang, config.cnn_architecture, config.rnn_architecture, config.decoder_type, config.top_k)
        
    open(val_result_file, "w").close() 
    
    with open(val_result_file, "a") as valfile:                    
            valfile.write("BLEU-1 Score of the System: " + str(bleu1) + "\n")
            valfile.write("BLEU-2 Score of the System: " + str(bleu2) + "\n")
            valfile.write("BLEU-3 Score of the System: " + str(bleu3) + "\n")
            valfile.write("BLEU-4 Score of the System: " + str(bleu4) + "\n")
            valfile.write("WER of the system: " + str(wer) + "\n")
            valfile.write("Word accuracy of the system: " + str(1-wer) + "\n")
      
    
    print("BLEU-1 Score of the System: " + str(bleu1))
    print("BLEU-2 Score of the System: " + str(bleu2))
    print("BLEU-3 Score of the System: " + str(bleu3))
    print("BLEU-4 Score of the System: " + str(bleu4))        
    print("WER of the system: " + str(wer))
    print("Word accuracy of the system: " + str(1-wer))
    print("High Quality Comparisons")
    print(len(high_images))
    print(high_images)
    print("Range of BLEU scores for images")
    print(value_bleu)
    print("Range of WER scores for images")
    print(value_wer)   
    
    time_after = datetime.now()      
    time_taken = (time_after - time_bef)
    print("Time to validate model")
    print(time_taken)  

    
    
if __name__ == '__main__':    
    main()
