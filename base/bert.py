import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F

from transformers import BertForSequenceClassification, BertModel


class BertClassification(nn.Module):
    def __init__(self, model="bert-base-uncased"):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model)

    def forward(self, x, x_length=[]):
        output = self.bert(x)
        return output[0]


class BertEncoder(nn.Module):
    def __init__(self, out_dim, pretrained_model="bert-base-uncased", dropout_p=0.1):
        super().__init__()

        self.out_dim = out_dim
        self.bert = BertModel.from_pretrained(pretrained_model)

        # for name, param in self.bert.named_parameters():
        #     if param.requires_grad:
        #         print(name)

        # for p in self.bert.parameters():
        #    p.requires_grad = False

        self.out = nn.Sequential(
                        nn.Dropout(p=dropout_p),
                        nn.Linear(self.bert.config.hidden_size, out_dim))
    
    def forward(self, x, x_length=[]):
        x_input = x[:,0,:]
        x_attn = x[:,1,:]
        x, _ = self.bert(x_input, attention_mask=x_attn)
        # out_pooled = outputs[0][:,0]
        x = x[:, 0]
        x = self.out(x)
        return x
        


