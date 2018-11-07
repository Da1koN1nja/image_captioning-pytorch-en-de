import numpy as np

from scipy.misc import imread, imresize
from collections import Counter
import sys
import argparse
import pandas as pd
import pickle
import os
import glob
import torch
import torch.nn as nn

from datetime import datetime
from data_loader import get_loader_flickr 
from vocabulary import Vocab
from attentionmodel import CNNEncoder, AttentionRNNDecoder
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

from attentionval import val_train
from tqdm import tqdm

from config import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    
    config = Config()
    
    lang = config.lang
    model_path =config.model_dir
    if lang == 'de':
        vocab_path = 'data/clean_vocab_de.pkl'
        caption_dir = config.de_caption_dir        
        
    else:    
        vocab_path = 'data/clean_vocab_en.pkl'
        caption_dir = config.en_caption_dir   
        
    best_file = "AttentionTrainingBest--{}.txt".format(lang)        
    bleu_file = "AttentionTrainingBleu--{}.txt".format(lang)
    
    open(best_file, "w").close()
    open(bleu_file, "w").close()
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
       
           
    image_dir = config.res_train    
    
    transform = transforms.Compose([ 
        transforms.RandomCrop(config.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
   
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    
        
    data_loader = get_loader_flickr(image_dir, caption_dir, vocab, lang,
                            transform, config.batch_size,
                           shuffle=True, num_workers=config.num_workers) 
   

    
    encoder = CNNEncoder(config).to(device)

    encoder.retrain(False)
    decoder = AttentionRNNDecoder(config, len(vocab)).to(device)
    bleu_val = 0
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()#.to(device)
    
    
    
    #print(list(encoder.parameters()))
    
    print(config.save_step)
    #print(config.)
    
    decoder_params = list(decoder.parameters())
    encoder_params = list(filter(lambda p: p.requires_grad, encoder.parameters())) 
    params = encoder_params + decoder_params

   
    
    #encoder_optimizer = torch.optim.Adam(encoder_params, lr=1e-4, eps = 1e-6)
    #decoder_optimizer = torch.optim.Adam(decoder_params)  
    #decoder_optimizer = torch.optim.Adam(decoder_params, lr=1e-4, eps = 1e-6) 
    #print(decoder_params)
    
    #optimizer = torch.optim.Adam(params, lr=1e-3, eps = 1e-6)
#     optimizer = torch.optim.Adam(params, lr=2e-3, eps = 1e-6)
    optimizer = torch.optim.Adam(params)    
    
    log_prob = nn.LogSoftmax(dim=1)
    total_step = len(data_loader)
    time_bef = datetime.now()
    print(time_bef)
    print(len(vocab))
    iteration = 0
    e_iteration = 0
    epoch_bleu = 0    
    #delve = 0
    prob = nn.Softmax(dim = 1)
    for epoch in range(config.num_epochs):
        if e_iteration == 8:
            print("Early Stopping")
            e_iteration = 0
            break   
        decoder.train()
        encoder.train()
        
        
        for i, (images, captions, lengths) in enumerate(data_loader):
            
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            lengths = lengths    
            features = encoder(images)                    
            
            outputs, decoder_lengths, alphas = decoder(features, captions, lengths)
            
            outputs = pack_padded_sequence(outputs, decoder_lengths, batch_first=True)[0]
            targets = pack_padded_sequence(captions[:, 1:], decoder_lengths, batch_first=True)[0]
            
            loss = criterion(outputs, targets)
            nextish = alphas.sum(dim=1)
           
            ones1 = torch.ones(nextish.size()).to(device)
           
            attention_loss = config.alpha_c * ((ones1 - alphas.sum(dim=1)) ** 2).mean()
            
            #print(attention_loss)
            loss += attention_loss            
            
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            
            for group in optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.data.clamp_(-config.clip_val, config.clip_val)
            
            optimizer.step()
            #decoder_optimizer.step()
            #encoder_optimizer.step()

            # Print log info
            if i % config.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, config.num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                
            # Save the model checkpoints
            if (i+1) % config.save_step == 0:
                #iteration +=1
                
                encoder_path = 'attention_encoder-{}--{}-{}--{}-{}--backpropall-{}.ckpt'.format(config.cnn_architecture, config.rnn_architecture, lang, epoch+1, i+1, config.train_cnn)
                
                decoder_path = 'attention_decoder-{}--{}-{}--{}-{}--backpropall-{}.ckpt'.format(config.cnn_architecture, config.rnn_architecture, lang, epoch+1, i+1, config.train_cnn)
                
                torch.save(decoder.state_dict(), os.path.join(
                    model_path, decoder_path))
                torch.save(encoder.state_dict(), os.path.join(
                    model_path, encoder_path))
                bleu = val_train(config, encoder_path, decoder_path)
                
                if bleu > bleu_val:
                    #iteration = 0
                    
                    bleu_val = bleu
                    best_encoder_path = 'best_attention_encoder--{}-lang--{}.ckpt'.format(config.cnn_architecture, lang)
                
                    best_decoder_path = 'best_attention_decoder--{}-lang--{}.ckpt'.format(config.rnn_architecture, lang)
                
                    torch.save(decoder.state_dict(), os.path.join(
                        model_path, best_decoder_path))
                    torch.save(encoder.state_dict(), os.path.join(
                        model_path, best_encoder_path))
                   
                    with open(best_file, "a") as myfile:
                        myfile.write("Epoch: {}, Batch: {}, Bleu4: {}\n".format(epoch+1,i+1, bleu_val))
                
                if iteration == 4: 
                    for en_filename in glob.glob("models/attention_encoder-{}--{}-{}*".format(config.cnn_architecture, config.rnn_architecture, lang)):
                        os.remove(en_filename)
                    for de_filename in glob.glob("models/attention_decoder-{}--{}-{}*".format(config.cnn_architecture, config.rnn_architecture, lang)):
                        os.remove(de_filename)
                    iteration = 0
                                                     
                                                     
                                                     
                with open(bleu_file, "a") as myfile:
                    myfile.write("Epoch: {}, Batch: {}, Bleu4: {}\n".format(epoch+1,i+1, bleu))
                    
                                   
                    
            if epoch == config.num_epochs-1 and i == total_step-1:
                encoder_path = 'attention_encoder-{}--{}-{}--{}-{}--backpropall-{}.ckpt'.format(config.cnn_architecture, config.rnn_architecture, lang, epoch+1, i+1, config.train_cnn)
                
                decoder_path = 'attention_decoder-{}--{}-{}--{}-{}--backpropall-{}.ckpt'.format(config.cnn_architecture, config.rnn_architecture, lang, epoch+1, i+1, config.train_cnn)
                
                
                torch.save(decoder.state_dict(), os.path.join(
                    model_path, decoder_path))
                torch.save(encoder.state_dict(), os.path.join(
                    model_path, encoder_path))
            
        
        if bleu_val > epoch_bleu:
            epoch_bleu = bleu_val
            e_iteration = 0
        else:
            e_iteration += 1
            #print(e_iteration)
        iteration += 1
        
        if e_iteration % 2 == 0:
            #encoder_scheduler.step() 
            #decoder_scheduler.step() 
            print("Epoch Iteration")
            print(e_iteration)
            if e_iteration != 0:
                print("Occur")
                adjust_learning_rate(optimizer, config.mod_lr)
            #scheduler.step() 
    time_after = datetime.now()      
    time_taken = (time_after - time_bef)
    print("Time to train model")
    print(time_taken)           

            #print(captions[0])
            #print(lengths.max)
            #print(lengths)

def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))



if __name__ == '__main__':        
    main()
    
