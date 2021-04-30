import numpy
import torch
import pytorch_lightning as pl 
import torch.nn as nn
from vocabulary import Vocabulary
from torch.utils.data import DataLoader
from data import TwitterDataset,load_traindata,load_traindata_nolabel

class RNN(pl.LightningModule):
    
    def __init__(self,vocab_len=5000,embedding_size=100,hidden_size=32,batch_size=32):
       
        #constants
        self.vocab_len = vocab_len
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        # model architecture
        self.embedding = nn.Embedding(self.vocab_len,self.embedding_size)
        self.LSTM = nn.LSTM(self.embedding_size,self.hidden_size,num_layers=3)
        self.dense1 = nn.Linear(self.hidden_size,16)
        self.dense2 = nn.Linear(16,2)
       
        #activation       
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        
        #criterion
        self.criterion = nn.BCELoss
    
    def forward(self,x):
        
        #get word embeddings
        word_embedding = self.embedding(x)

        #reshape before feeding into lstm layers
        lstm_input = word_embedding[None,:,:]
    
        #lstm layers
        output = self.LSTM(lstm_input)

        #dense layers
        output = self.dropout(self.leaky_relu(self.dense1(output)))
        output = self.dropout(self.leaky_relu(self.dense2(output)))

        return self.sigmoid(output)

    
    def train_dataloader(self):
        x,y = load_traindata()
        train_dataset = TwitterDataset(x[20001:],y[20001:])
        train_dataloader = DataLoader(train_dataset,batch_size=self.batch_size,shuffle=True,num_workers=1)
        return train_dataloader
    
    def val_dataloader(self):
        x,y = load_traindata()
        val_dataset = TwitterDataset(x[0:20000],y[0:20000])
        val_dataloader = DataLoader(val_dataset,batch_size=self.batch_size,shuffle=True,num_workers=1)
        return val_dataloader


    def training_step(self,batch,batch_idx):
        x,y = batch 
        preds = self(x)
        loss = self.criterion(preds,y)
        return loss


    def validation_step(self,batch,batch_idx):
        x,y = batch 
        preds = self(x)
        loss = self.criterion(preds,y)
        return loss


if __name__ == '__main__':
    model = RNN()
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model)
