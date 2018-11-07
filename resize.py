import numpy as np
#import nltk
from scipy.misc import imread, imresize
from collections import Counter
import sys
import argparse
import pandas as pd
import pickle
import os

from PIL import Image
from config import Config

def resize_image(image, size): 
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)
        if (i+1) % 100 == 0:
            print ("[{}/{}] Resized the images and saved into '{}'."
                   .format(i+1, num_images, output_dir))

def main(args):
    config = Config()    
    print("Start")
    config.mode = args.mode
    
    if config.mode == 'train':
        
        image_dir = config.train_image_dir
        output_dir = config.res_train
        image_size = [config.image_size, config.image_size]
    
    elif config.mode == 'val':
        image_dir = config.val_image_dir
        output_dir = config.res_val
        image_size = [config.image_size, config.image_size]    
    
    resize_images(image_dir, output_dir, image_size)
    
    
    print("End of Resizing")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,  default='train',                       
                        help='which part of dataset that needs to be resized')    
    args = parser.parse_args()
    main(args)
    
