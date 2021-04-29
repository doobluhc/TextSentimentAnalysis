import numpy
import torch
import pytorch_lightning as pl 
import torch.nn as nn

class RNN(pl.LightningModule):
    
    def __init__(self,vocab_len,embedding_size=100,hidden_size=32):
       
        # model architecture
        self.embedding = nn.Embedding(vocab_len,embedding_size)
        self.LSTM = nn.LSTM(embedding_size,hidden_size,num_layers=3)
        self.dense1 = nn.Linear(hidden_size,16)
        self.dense2 = nn.Linear(16,2)
       
        #activation       
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        
        #criterion
        self.criterion = nn.CrossEntropyLoss
    
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

        return output

    
    def train_dataloader(self):
         pass

    
    def val_dataloader(self):
        pass


    def training_step(self,batch,batch_idx):
        
        x,y = batch
        
        #getting predictions
        preds = self(x)


    def validation_step(self,batch,batch_idx):
        pass
