from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import string

from collections import Counter
from os import path
from shutil import copyfile
import os
import re
import sys
import nltk
#nltk.download()
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


training_image_dir = "/home/da1kon1nja/Documents/Skripsie Datasets/train_set"
val_image_dir = "/home/da1kon1nja/Documents/Skripsie Datasets/val_set"

data_set_dir = "./data/"

splits_dir = "/home/da1kon1nja/Documents/Skripsie Datasets/splits"

eng_train_caption_dir = "/home/da1kon1nja/Documents/Skripsie Datasets/mmt_task2/en/train"
de_train_caption_dir = "/home/da1kon1nja/Documents/Skripsie Datasets/mmt_task2/de/train"

eng_val_caption_dir = "/home/da1kon1nja/Documents/Skripsie Datasets/mmt_task2/en/val"
de_val_caption_dir = "/home/da1kon1nja/Documents/Skripsie Datasets/mmt_task2/de/val"

caption_dir = "./data/captions/"

en_test_caption_dir = "/home/da1kon1nja/Documents/Skripsie Datasets/mmt16_task2_test/"
de_test_caption_dir = "/home/da1kon1nja/Documents/Skripsie Datasets/mmt16_task2_test/"


def main():
    eng_caption = []
    de_caption = []

    en_train = []
    en_val = []
    en_test = []
    de_train = []
    de_val = []
    de_test = []
    
    image_file = []
    image_id = []

    train_image_file = []
    train_image_id = []
    
    val_image_file = []
    val_image_id = []

    test_image_file = []
    test_image_id = []  
   

    for l in ["en", "de"]:
        for s in ["train", "val", "test"]:
            eng_caption = []
            de_caption = []
            for i in range(5):
                fn = path.join(caption_dir, s + "/" + l + "/" + l + "_"+ s + "." + str(i+1))                
                with open(fn) as f:
                    for line in f:
                        if l == "en":
                            tokens = word_tokenize(line)
                            tokens = [w.lower() for w in tokens]
                            table = str.maketrans('', '', string.punctuation)
                            stripped = [w.translate(table) for w in tokens]
                            
                            words  = [word for word in stripped if word.isalpha()] # str.maketrans('', '', string.punctuation)
                            
                                
                            words = [w for w in words if w != 's']                             
                            sentence = ' '.join(words)
                            eng_caption.append(sentence)
                        else:
                            
                            tokens = word_tokenize(line)
                            tokens = [w.lower() for w in tokens]
                            table = str.maketrans('', '', string.punctuation)
                            stripped = [w.translate(table) for w in tokens]
                            
                            words  = [word for word in stripped if word.isalpha()] # str.maketrans('', '', string.punctuation)
                            
                            
                            words = [w for w in words if w != 's']  
                            sentence = ' '.join(words)
                            de_caption.append(sentence)                   
            if s == "train":
                if l == "en":
                    en_train = eng_caption
                else:
                    de_train = de_caption
            if s == "val":
                if l == "en":
                    en_val = eng_caption
                else:
                    de_val = de_caption

            if s == "test":
                if l == "en":
                    en_test = eng_caption
                else:
                    de_test = de_caption
    
    
    
    for s in ["train", "val", "test"]:
        fn = path.join(splits_dir, s + "_images.txt")
        image_file = []
        image_id = []
        with open(fn) as f:
            for line in f:
                image = path.splitext(line.strip())[0]
                if s != "test":
                    src_fn = str(path.join(data_set_dir, "resized_" + s + "_set" +"/"  + str(image) + ".jpg"))
                else:
                    src_fn = str(path.join(data_set_dir, s + "_set" + "/" + str(image) + ".jpg"))
                image_file.append(src_fn)
                                   
                image_id.append(image)
        if s == "train":
            train_image_file = image_file*5
            train_image_id = image_id*5
        if s == "val":
            val_image_file = image_file*5
            val_image_id = image_id*5
        if s == "test":
            test_image_file = image_file*5
            test_image_id = image_id*5
            
    final_file = image_file*5
    final_id = image_id*5       
    
    print(len(train_image_file))    
    print(len(train_image_id))  
    print(len(val_image_file))    
    print(len(val_image_id)) 
    print(len(test_image_file))    
    print(len(test_image_id)) 
    raw_en_train_data = {'caption': en_train, 'image_file':train_image_file, 'image_id':train_image_id}
    raw_de_train_data = {'caption': de_train, 'image_file':train_image_file, 'image_id':train_image_id}

    endf = pd.DataFrame(raw_en_train_data, columns = ['caption', 'image_file', 'image_id'])
    dedf = pd.DataFrame(raw_de_train_data, columns = ['caption', 'image_file', 'image_id'])
    
    endf.to_csv('./data/trainendata.csv')
    dedf.to_csv('./data/traindedata.csv')


    raw_en_val_data = {'caption': en_val, 'image_file':val_image_file, 'image_id':val_image_id}
    raw_de_val_data = {'caption': de_val, 'image_file':val_image_file, 'image_id':val_image_id}

    endf = pd.DataFrame(raw_en_val_data, columns = ['caption', 'image_file', 'image_id'])
    dedf = pd.DataFrame(raw_de_val_data, columns = ['caption', 'image_file', 'image_id'])
    
    endf.to_csv('./data/valendata.csv')
    dedf.to_csv('./data/valdedata.csv')


    raw_en_test_data = {'caption': en_test, 'image_file':test_image_file, 'image_id':test_image_id}
    raw_de_test_data = {'caption': de_test, 'image_file':test_image_file, 'image_id':test_image_id}

    endf = pd.DataFrame(raw_en_test_data, columns = ['caption', 'image_file', 'image_id'])
    dedf = pd.DataFrame(raw_de_test_data, columns = ['caption', 'image_file', 'image_id'])
    
    endf.to_csv('./data/testendata.csv')
    dedf.to_csv('./data/testdedata.csv')

     
    #dedflim.to_csv('/home/da1kon1nja/Documents/Skripsie Datasets/cleandedatalimit40.csv')  
if __name__ == "__main__":
    main()
