import numpy as np 
import pandas as pd
import os 
import nltk
import itertools
import torch
from torch.utils.data import Dataset
from nltk import word_tokenize
# nltk.download('punkt')


class TwitterDataset(Dataset):
    def __init__(self,x,y):
        pass

    def __getitem__(self,idx):
        pass

    def __len__(self):
        pass




def create_vocab(path,vocab_len = 5000):
    f = open(path, "r")
    wordfreq = {}
    tokens = word_tokenize(f.read().lower())
    for token in tokens:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1
    
    #sort the dict based on the occurences and convert to list 
    sorted_list = sorted(wordfreq, key=wordfreq.__getitem__, reverse=True)
    
    #store the vocab as a txt
    with open("vocabulary.txt", 'w') as f:
        for word in sorted_list[:vocab_len]:
            f.write(word + "\n")

    print ("vocab saved to vocabulary.txt")
    


if __name__ == '__main__':
    create_vocab((os.path.join("dataset","training_nolabel.txt")))