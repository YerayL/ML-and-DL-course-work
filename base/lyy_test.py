#!/usr/bin/env python3
# coding: utf-8

import sys
import json

import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from cnn import TextCNNEncoder
from rnn import RNNEncoder
from bert import BertEncoder, BertClassification
from transformers import BertForSequenceClassification


from dataset import *
from train import *

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# early stop
PATIENCE = None

torch.cuda.set_device(3)

class Classifier(nn.Module):
    def __init__(self, encoder, out_class, hidden_size=256, dropout_p=0.1):
        super(Classifier,self).__init__()

        out_dim = encoder.out_dim
        self.encoder = encoder
        self.out = nn.Sequential(
                        nn.Linear(out_dim, hidden_size),
                        nn.Dropout(p=dropout_p),
                        nn.ReLU(),
                        nn.Linear(hidden_size, out_class))
    
    def forward(self, x, x_length=[]):
        x = self.encoder(x, x_length)
        # import pdb; pdb.set_trace()
        x = self.out(x)
        return x



def build_encoder(emb, encoder_type, out_dim, dropout_p, **op):
    if encoder_type == "cnn":
        return TextCNNEncoder(emb=emb, out_dim=out_dim, dropout_p=dropout_p, **op)
    elif encoder_type == "rnn":
        return RNNEncoder(emb=emb, out_dim=out_dim, dropout_p=dropout_p, **op)
    elif encoder_type == "bert":
        return BertEncoder(out_dim=out_dim, dropout_p=dropout_p, **op)


def build_model(emb, encoder, out_class, dropout_p):
    encoder = build_encoder(emb=emb, **encoder)
    return Classifier(encoder=encoder, out_class=out_class, dropout_p=dropout_p)


def load_dataset(name, **op):
    return sst_dataset(root=name, device="cuda", **op)



if __name__ == "__main__":
    config_file = './opt.movie.bert.json'

    # config = json.load(open(config_file))
    config = json.load(open('./opt.movie.bert.json'))
    
    train_iter, dev_iter, test_iter, TEXT, LABEL = load_dataset(**config["dataset"])

    if not config["model"]["encoder"]["encoder_type"] == "bert":
        emb = nn.Embedding(num_embeddings=len(TEXT.vocab), embedding_dim=300,
                           padding_idx=TEXT.vocab.stoi["<pad>"])
    else:
        emb = None
        # model = BertClassification(config["model"]["out_dim"])
    model = build_model(emb=emb, **config["model"])
    # BertForSequenceClassification
    model = model.to("cuda")
    criterion = nn.CrossEntropyLoss().to("cuda")
    # encoder = list(map(id, model.encoder.parameters()))
    # out = list(map(id, model.out.parameters()))
    # params = [
    #     {"params": model.encoder.parameters()},
    #     {"params": model.out.parameters(), "lr": 0.0001},
    # ]
    # optimizer = optim.Adam(params, lr=0.00001)
    optimizer = optim.Adam(model.parameters())
    
    def model_train(e):
        e.model.zero_grad()
        (x, x_length), label = e.batch.text, e.batch.label
        # import pdb; pdb.set_trace()
        y_predict = e.model(x, x_length)
        loss = e.criterion(y_predict, label)
        loss.backward()
        e.optimizer.step()
        e.progress.set_postfix(loss=loss.item())


    def model_eval(e):
        (x, x_length), label = e.batch.text, e.batch.label
        # import pdb; pdb.set_trace()
        y_predict = e.model(x, x_length)
        return y_predict.argmax().view(1, 1), label.view(1, 1)


    train(train_iter, dev_iter, model, criterion, optimizer, max_iters=1000, save_every=1000, device="cuda",
          handler=model_train, patience=PATIENCE)



    y, g = evaluate(test_iter, model, model_location="checkpoint/Classifier.500.pt",
                    criterion=criterion, device="cuda", handler=model_eval)



    # Config                                                  : Iters
    # Acc: opt.sst-2.rnn.json tensor(0.7569, device='cuda:0') : 8000
    # Acc: opt.sst-2.cnn.json tensor(0.7706, device='cuda:0') : 8000
    # Acc: opt.sst-2.bert.json tensor(0.8544, device='cuda:0'): 4000
    # Acc: opt.sst-5.rnn.json tensor(0.2958, device='cuda:0') : 4000
    # Acc: opt.sst-5.cnn.json tensor(0.3081, device='cuda:0') : 4000
    # Acc: opt.sst-5.bert.json tensor(0.4484, device='cuda:0'): 4000
    # import pdb; pdb.set_trace()
    print(f"Acc: {config_file}", (y-g).eq(0).sum().true_divide(y.size(0)))