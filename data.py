import numpy as np 
import pandas as pd
import os 
import nltk
import itertools
import torch
from torch.utils.data import Dataset
from nltk import word_tokenize
from vocabulary import Vocabulary
# nltk.download('punkt')


class TwitterDataset(Dataset):
    def __init__(self,x,y):
        self.data = x
        self.label = y

    def __getitem__(self,idx):
        return self.data[idx],self.label[idx]

    def __len__(self):
        return len(self.data)
        
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


def load_traindata(data_path="dataset/training_label.txt"):
    with open(data_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip('\n').split(' ') for line in lines]
    x = [line[2:] for line in lines]
    x = preprocess(x)
    y = [line[0] for line in lines]
    return x, y


def load_traindata_nolabel(data_path="dataset/training_nolabel.txt"):
    with open(data_path, 'r') as f:
        lines = f.readlines()
    x = [line.strip('\n').split(' ') for line in lines]
    #TODO encode based on vocab
    return x


def load_testingdata(data_path="dataset/testing_data.txt"):
    with open(data_path,'r') as f:
        lines = f.readlines()
        lines = [line.strip('\n').split(',') for line in lines]
    x = [line[1:] for line in lines]
    y = [line[0] for line in lines]
    #TODO encode based on vocab
    return x, y
    
def preprocess(x):
    vocab = Vocabulary()
    max_len = max([len(word) for word in x]) 
    encoded_x = torch.zeros((len(x),max_len,vocab.size))
    for row,sentence in enumerate(x):
        for idx,word in enumerate(sentence):
            encoded_x[row,] = vocab.word2idx(word)
        encoded_x.append(encoded_word)

if __name__ == '__main__':
    # create_vocab((os.path.join("dataset","training_nolabel.txt")))
    load_traindata()