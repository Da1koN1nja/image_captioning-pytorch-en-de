import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import copy
import os
from os import path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class CNNEncoder(nn.Module):
    def __init__(self, config):   
        super(CNNEncoder, self).__init__()
        self.config = config
        layers = []
        
        if self.config.cnn_architecture == 'vgg16':
             
                vgg16 = models.vgg16(pretrained=config.pretrain)                   
                layers = list(vgg16.children())[:-1]            
                layers = layers[0][:-1]                          
        elif self.config.cnn_architecture == 'resnet50':
                resnet50 = models.resnet50(pretrained=config.pretrain)                
                layers = list(resnet50.children())[:-2]
                         
        self.cnn = nn.Sequential(*layers)
     
        if self.config.cnn_architecture == 'vgg16':           
            self.bn = nn.BatchNorm2d(512)
        elif self.config.cnn_architecture == 'resnet50':             
            self.bn = nn.BatchNorm2d(2048)      
        self.retrain()
        
    def forward(self, images):          
        features = self.cnn(images)        
        features = self.bn(features)
        features = features.permute(0, 2, 3, 1)
        return features

    def retrain(self, re = True):  
        if self.config.cnn_architecture == 'resnet50':
            for p in self.cnn.parameters():              
                p.requires_grad = False
            
            for n in list(self.cnn.children())[6:]:                   
                for p in n.parameters():
                    p.requires_grad = re
        else:
            for p in self.cnn.parameters():               
                p.requires_grad = False
            
            for n in list(self.cnn.modules())[11:-1]:                   
                i+=1                
                for p in n.parameters():
                    p.requires_grad = re

class AttentionModel(nn.Module):
    def __init__(self, config, encoder_dim):        
        super(AttentionModel, self).__init__()
        self.config = config        
        self.encoder_dim = encoder_dim
        self.decoder_dim = config.hidden_size
        self.linear_enc = nn.Linear(self.encoder_dim, config.attention_dim, bias = False)
        self.linear_dec = nn.Linear(self.decoder_dim, config.attention_dim) 
        self.attention_layer = nn.Linear(config.attention_dim, 1, bias = False)        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=config.embed_drop_rate)        
        self.init_weights()
        
        
    def init_weights(self):       
        
        torch.nn.init.xavier_uniform_(self.attention_layer.weight)
        torch.nn.init.xavier_uniform_(self.linear_enc.weight)

        torch.nn.init.xavier_uniform_(self.linear_dec.weight)
        self.linear_dec.bias.data.fill_(0) 
    
    def forward(self, encoded_feats, decoder_hidden):
        
        encoded_feats = self.dropout(encoded_feats)
        att1 = self.linear_enc(encoded_feats)
        att2 = self.linear_dec(decoder_hidden)        
        rs = self.relu(att1 + att2.unsqueeze(1))
        att = self.attention_layer(rs).squeeze(2)        
        alpha = self.softmax(att)
        context = (encoded_feats * alpha.unsqueeze(2)).sum(dim=1)       
        return context, alpha
    

class AttentionRNNDecoder(nn.Module):
    def __init__(self, config, vocab_size):
        
        super(AttentionRNNDecoder, self).__init__()
        self.config = config
        if config.cnn_architecture == 'resnet50':            
            self.encoder_dim = 2048
        else:
            self.encoder_dim = 512
        self.embed = nn.Embedding(vocab_size, config.embed_size)
        
        self.dropout = nn.Dropout(p=config.embed_drop_rate)
        self.rnndrop = nn.Dropout(p=config.rnn_drop_rate)        
        self.vocab_size = vocab_size
        #self.beam = self.Beam(start, end, k)
        #print(self.encoder_dim)
        self.init_h = nn.Linear(self.encoder_dim, config.hidden_size)
        
        if config.rnn_architecture == 'lstm':            
            self.rnncell = nn.LSTMCell(config.embed_size+self.encoder_dim, config.hidden_size)
            self.init_c = nn.Linear(self.encoder_dim, config.hidden_size)
        else:            
            self.rnncell = nn.GRUCell(config.embed_size+self.encoder_dim, config.hidden_size)
        
        self.attention = AttentionModel(config, self.encoder_dim)
        
        self.f_gate = nn.Linear(config.hidden_size, self.encoder_dim)
        
        self.decode = nn.Linear(config.hidden_size+self.encoder_dim+config.hidden_size, config.decode_layer)
     
        self.ctx = nn.Linear(self.encoder_dim, config.hidden_size, bias = False)      
        self.linear = nn.Linear(config.hidden_size, vocab_size)       
        
        self.max_seg_length = config.max_caption_length       
               
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        
        self.init_weights()
        
        
    def init_weights(self):
        
        torch.nn.init.xavier_uniform_(self.embed.weight)
        torch.nn.init.xavier_uniform_(self.init_h.weight)
        
        torch.nn.init.xavier_uniform_(self.f_gate.weight)
        torch.nn.init.xavier_uniform_(self.decode.weight)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.xavier_uniform_(self.ctx.weight)
        
        self.init_h.bias.data.fill_(0)
        
        if self.config.rnn_architecture == 'lstm':            
            torch.nn.init.xavier_uniform_(self.init_c.weight)
            self.init_c.bias.data.fill_(0)
                
        self.f_gate.bias.data.fill_(0)
        
        self.decode.bias.data.fill_(0)
        
        self.linear.bias.data.fill_(0)    
    
    def init_states(self, encoded_feats):
        
        encoder_mean = encoded_feats.mean(1)
        
        init_h = self.tanh(self.init_h(encoder_mean))
        if self.config.rnn_architecture == 'lstm':
            init_c = self.tanh(self.init_c(encoder_mean))
            return init_h, init_c
        
        return init_h
    
    def forward(self, features, captions, lengths):      
       
        bat = features.size(0)
        encoded_feats = features.view(bat, -1, self.encoder_dim)       
                
        num_pixels = encoded_feats.size(1)
        decode_lengths = [leng-1 for leng in lengths]
        maximum = max(decode_lengths)
                
        embeddings = self.embed(captions)        
        
        if self.config.rnn_architecture == 'lstm':
            h, c = self.init_states(encoded_feats)
        else:
            h = self.init_states(encoded_feats)

        
        predictions = torch.zeros(bat, maximum, self.vocab_size).to(device)
        alphas = torch.zeros(bat, maximum, num_pixels).to(device)
        log_prob = nn.LogSoftmax(dim=1)
        
        
        for t in range(maximum):
            batch_t = sum([l > t for l in decode_lengths])            
            contexts, alpha = self.attention(encoded_feats[:batch_t], h[:batch_t])
            
            gate = self.sigmoid(self.f_gate(h[:batch_t]))
            attention_vector = gate*contexts
           
            inputss = torch.cat([embeddings[:batch_t, t, :], attention_vector], dim=1)           
            if self.config.rnn_architecture == 'lstm':                        
                h, c = self.rnncell(torch.cat([embeddings[:batch_t, t, :], attention_vector], dim = 1), (h[:batch_t], c[:batch_t]))               
            else:                
                h = self.rnncell(torch.cat([embeddings[:batch_t, t, :], attention_vector], dim = 1), h[:batch_t])
                
            
            attention_vector = self.ctx(attention_vector)
            h = self.dropout(h)
            
                      
            exout = h + attention_vector + embeddings[:batch_t, t,:]
            
            exout = self.tanh(exout)
            
            
            preds = self.linear(self.dropout(exout))          
            predictions[:batch_t, t, :] = preds            
            alphas[:batch_t, t, :] = alpha
        
               
        return predictions, decode_lengths, alphas
    
    def sample(self, features, k, start, end, states=None):        
        sampled_ids = []        
        log_prob = nn.LogSoftmax(dim=1)        
        bsize = features.size(0)
        edim = features.size(-1)        
        encoded_feats = features.view(bsize, -1, edim)  
        predictions = torch.zeros(bsize, self.max_seg_length).to(device)
        
        start = torch.tensor([start]).to(device)
       
        inputs = torch.zeros([bsize, 1, self.config.embed_size]).to(device)
       
        sampled_ids.append(start)
       
        start_emb = self.embed(start).unsqueeze(1)
        for i in range(bsize):
            predictions[i] = start
            inputs[i] = start_emb        
        if self.config.rnn_architecture == 'lstm':
            h, c = self.init_states(encoded_feats)            
        else:
            h = self.init_states(encoded_feats)
       
        if self.config.decoder_type == 'beam':
            sampled_ids = [[] for i in range(bsize)]
            for t in range(bsize):                
                if self.config.rnn_architecture == 'lstm':
                    beam = Beam(self.config, start, end, k, encoded_feats[t], self.rnncell, self.ctx, self.tanh, self.linear, self.embed, self.attention, self.f_gate, self.sigmoid, [h[t], c[t]])
                else:
                    beam = Beam(self.config, start, end, k, encoded_feats[t], self.rnncell, self.ctx, self.tanh, self.linear, self.embed, self.attention, self.f_gate, self.sigmoid, [h[t]])
                a, b = beam.init_beam()
                m = beam.incept()
                if self.config.hacked_val is True:
                    sampled_ids = m
                else:
                    m.sort(key=lambda x:-x[1])                    
                    return_val = m[0]                    
                    sampled_ids[t] = return_val[0]
                    
            
        if self.config.decoder_type == 'greedy':             
            for i in range(1, self.max_seg_length):                
                contexts, alpha = self.attention(encoded_feats, h)            
                gate = self.sigmoid(self.f_gate(h))
                attention_vector = gate*contexts                
                inputs = inputs.squeeze(1)
                if self.config.rnn_architecture == 'lstm':            
                    h, c = self.rnncell(torch.cat([inputs, attention_vector], dim=1), (h, c))
                else:
                    h = self.rnncell(torch.cat([inputs, attention_vector], dim=1), h)                
                               
                attention_vector = self.ctx(attention_vector)                
                exout = h + attention_vector + inputs                
                        
                exout = self.tanh(exout)
            
                preds = self.linear(exout)                
               
                softout = log_prob(preds)
                
                _, predicted = softout.max(1)                                  
                predictions[:, i] = predicted
                pre = self.embed(predicted).unsqueeze(1)                
                inputs = pre                                                
            if self.config.loader == True:
                sampled_ids = predictions                
            else:               
                sampled_ids = predictions[0]                          
        
                
        return sampled_ids
        
    
    
class Beam(object):
    def __init__(self, config, start, end, k, features, rnncell, ctx, tanh, linear, embed, attention, f_gate, sigmoid, states):
        super(Beam, self).__init__()        
        self.bos = start
        self.eos = end
        self.inputs = 0
        self.features = features
        self.k = k        
        self.log_prob = nn.LogSoftmax(dim=1)
        self.log_prob2 = nn.LogSoftmax(dim=0)
        self.prob = nn.Softmax(dim=1)
        self.rnncell = rnncell
        self.linear = linear
        self.embed = embed   
        self.attention = attention
        self.ctx = ctx
        self.tanh = tanh
        self.f_gate = f_gate
        self.sigmoid = sigmoid  
        self.config = config        
        self.nextstate = [None, None, None]        
        self.states = states        
        self.current_score = torch.zeros([k], dtype=torch.float32).to(torch.device("cuda"))
        self.prev_score = torch.zeros([k], dtype=torch.float32).to(torch.device("cuda"))
        self.next_scores = torch.zeros([k, k], dtype=torch.float32).to(torch.device("cuda")) 
        self.current_seq = torch.zeros([k, k], dtype=torch.float32).to(torch.device("cuda"))
        
        self.current_seq = torch.zeros([config.sample_batch_size, k, k], dtype=torch.float32).to(torch.device("cuda"))
        
        self.prev_seq = [[] for i in range(self.k)]      
       
        self.complete = []
        self.partial = []
        #sequence

    def add_sentence(self, sentence, score):
        
        definite = score
     
        arr = np.asarray(sentence)
     
        sentence = torch.tensor(sentence).to(torch.int64)
        
        leng_norm = (5+len(sentence))/(5+1)
       
        leng_norm = leng_norm**self.config.alpha
        
        definite =  definite/leng_norm
             
        element = [sentence, definite]
            
        
            
        
            
        return element
        
    def init_beam(self):
        sampled_ids = []
        
        
        for j in range(self.k):
            self.prev_score[j] = 0           
            self.prev_seq[j].append(self.bos)           
        
        a = self.prev_score
       
        b = self.prev_seq

        return a, b 

    def incept(self):
        index = 1          
        recur = self.advance_short(index, self.inputs, self.prev_seq, self.prev_score, self.states) 
        if len(self.complete) > 0:
            vall = self.complete
        else:
            vall = self.partial
                
        return vall
    
    
    
    

    def advance_short(self, index, inputs, prev_seq, prev_score, prev_state):
        inputss = torch.zeros([self.k])
        revolver = []       
        len_prev = self.k
        previous_sequence = prev_seq[:]
        previous_score = prev_score[:]
        counter = 0        
        previous_states = [None]*self.k
        for i in range(index, self.config.max_caption_length):             
            if i == 1:
                previous_sequence = [previous_sequence[0]]
          
            if i != 1:
                previous_scores = next_scores[:]
                previous_states = next_states[:]
                if self.config.rnn_architecture == 'lstm':    
                    previous_memory = next_memory
                   
            statesss = []
            next_states = [None]*len(previous_sequence)
            if self.config.rnn_architecture == 'lstm':
                next_memory = [None]*len(previous_sequence)
                
            next_scores = torch.zeros([len(previous_sequence), self.k])
        
            predictions = torch.zeros([len(previous_sequence), self.k])
            for j in range(len(previous_sequence)):                                
                if i == 1:
                    
                    prev_pred = previous_sequence
                    prev_pred = torch.tensor(previous_sequence)#[j][i-1]   
                    prev_pred = prev_pred.to(torch.device("cuda"))
                    prev_pred = torch.tensor(prev_pred)  
                    
                    inputs = self.embed(prev_pred[0])                       
                   
                    contexts, alpha = self.attention(self.features, self.states[0].unsqueeze(0))
            
                    gate = self.sigmoid(self.f_gate(self.states[0]))
                    attention_vector = gate*contexts
                   
                    inputs = inputs.squeeze(1)
                    testing = torch.cat([inputs, attention_vector], dim=1)
                   
                    if self.config.rnn_architecture == 'lstm':            
                        h, c = self.rnncell(torch.cat([inputs, attention_vector], dim=1), (self.states[0].unsqueeze(0), self.states[1].unsqueeze(0)))
                    else:
                        h = self.rnncell(torch.cat([inputs, attention_vector], dim=1), self.states[0].unsqueeze(0))              
                                        
                else:
                    
                    
                    prev_pred = previous_sequence[j][i-1]               
                    
                    prev_pred = prev_pred.to(torch.device("cuda"))
                    prev_pred = prev_pred.to(torch.int64)
                    
                    inputs = self.embed(prev_pred)
                    
                    inputs = inputs.unsqueeze(1) 
                    contexts, alpha = self.attention(self.features, previous_states[j])
            
                    gate = self.sigmoid(self.f_gate(previous_states[j]))
                    attention_vector = gate*contexts
                    inputs = inputs.squeeze(1)
                    if self.config.rnn_architecture == 'lstm':            
                        h, c = self.rnncell(torch.cat([inputs, attention_vector], dim=1), (previous_states[j], previous_memory[j]))
                    else:
                        h = self.rnncell(torch.cat([inputs, attention_vector], dim=1), previous_states[j])  
                        
                  
                attention_vector = self.ctx(attention_vector)
                exout = h + attention_vector + inputs

                        
                exout = self.tanh(exout)

                outputs = self.linear(exout) 
                if self.config.prob:
                    softout = self.prob(outputs)     
                else:
                    softout = self.log_prob(outputs)     
                          
                a, b = softout.topk(self.k)
                                
                next_scores[j], predictions[j] = a, b
                
                next_states[j] = h
                if self.config.rnn_architecture == 'lstm':
                    next_memory[j] = c
                for l in range(self.k):
                    statesss.append(j)                
                

            if len(previous_sequence) == 0:
                break           
            
            res_scoress = torch.reshape(next_scores, (-1,))
            statessss = torch.Tensor(statesss)
            res_preds = torch.reshape(predictions, (-1,))
            
            if i == 1:
                summed_scores = torch.zeros([self.k])
                indv_scores = torch.zeros([self.k])
            else:
                summed_scores = torch.zeros([len(previous_sequence)*self.k])
                indv_scores = torch.zeros([len(previous_sequence)*self.k])
            if i == 1:
                for y in range(len(summed_scores)):
                    if self.config.prob:
                        indv_scores[y] = previous_score[0].cpu()
                        tmp_score = res_scoress[y]*indv_scores[y].cpu()
                    else:                        
                        indv_scores[y] = previous_score[0].cpu()
                        tmp_score = res_scoress[y] + indv_scores[y].cpu()
                    summed_scores[y] = tmp_score   
            else: 
                for y in range(len(summed_scores)):            
                    if self.config.prob:
                        indv_scores[y] = previous_score[statesss[y]].cpu()
                        tmp_score = res_scoress[y]*indv_scores[y]
                    else:
                        indv_scores[y] = previous_score[statesss[y]].cpu()
                        tmp_score = res_scoress[y] + indv_scores[y]
                    summed_scores[y] = tmp_score 
            
         
            sort_score, indices = torch.sort(summed_scores, 0, True)
            
            resss = res_preds[indices]
            res_indv = indv_scores[indices]
            res_indv = res_indv[:self.k]
            
            sort_staka = torch.stack([sort_score, resss])
          
            statesssss =statessss[indices].long()            
            next_ish = sort_staka[:, :self.k]
            state_ish = statesssss[:self.k]
          
            prev_sequencer = previous_sequence[:]     
            temp_states = []
            if self.config.rnn_architecture == 'lstm':    
                temp_mem = []
            for v in range(len(state_ish)):
                temp_states.append(next_states[state_ish[v]])
                if self.config.rnn_architecture == 'lstm':    
                    temp_mem.append(next_memory[state_ish[v]])            
            next_states = temp_states[:]
            if self.config.rnn_architecture == 'lstm':    
                next_memory = temp_mem[:]
                next_phase = [next_states, next_memory]
            else:
                next_phase = [next_states]
            
            temp_score = torch.zeros([self.k])
            
            workable_sent = previous_sequence[:]
            case = self.k
            n = 0
            temp_holder = []
            temp_s = []
            
            
            for m in range(case): 
                p = int(statesssss[m].numpy())
               
                tmp_score = sort_staka[0,m]
               
                temp = sort_staka[1,m]
                tmp = torch.Tensor([temp]).to(torch.device("cuda"))
                
                temp_sentence = list(workable_sent[p])
               

                temp_sentence.append(tmp)
                
                
                if tmp == self.eos or i+1 == self.config.max_caption_length: 
                   
                    if tmp == self.eos:                        
                        tmp_sc = copy.copy(tmp_score)
                    else:
                        tmp_sc = copy.copy(tmp_score)
                    
                    element = self.add_sentence(temp_sentence, tmp_sc)           
                    if tmp == self.eos:
                        self.complete.append(element)
                    else:
                        self.partial.append(element)
                   
                else:
                    
                    temp_holder.append(temp_sentence)

                    temp_s.append(tmp_score)
                    
            previous_score = temp_s          
            previous_sequence = temp_holder
               
        return 0
                 
                


    
