from __future__ import division
from __future__ import print_function
from collections import Counter
from os import path
from shutil import copyfile
import os
import re
import sys

splits_dir = "./data/splits/"
image_dir = "./data/flickr30k"
data_set_dir = "./data/"

training_image_dir = "./data/train_set/"
val_image_dir = "./data/val_set/"
test_image_dir = "./data/test_set/"


def main():    
    for s in ["train", "val", "test"]:
        
        fn = path.join(splits_dir, s + "_images.txt")
        with open(fn) as f:
            for line in f:               
                src_fn = path.join(image_dir, path.splitext(line.strip())[0] + ".jpg")
                dest_fn_1 = path.join(data_set_dir, s + "_set")
                if not os.path.exists(dest_fn_1):
                    os.makedirs(dest_fn_1)
                dest_fn = path.join(dest_fn_1, path.splitext(line.strip())[0] + ".jpg")
                
                os.rename(src_fn, dest_fn)
            # if i == 25:
             #       break;
        
        
if __name__ == "__main__":
    main()
