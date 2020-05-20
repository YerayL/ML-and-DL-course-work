import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class RNNEncoder(nn.Module):
    def __init__(self, emb, out_dim, rnn_type="lstm", num_layers=2, hidden_size=256, bidirectional=True, dropout_p=0.1):
        super().__init__()

        num_embeddings = emb.num_embeddings
        embedding_dim = emb.embedding_dim

        self.emb = emb
        self.out_dim = out_dim
        
        assert rnn_type in ("lstm", "gru")
        self.rnn_type = rnn_type
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(embedding_dim, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout_p)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(embedding_dim, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout_p)
        
        self.out = nn.Linear((2 if bidirectional else 1) * num_layers * hidden_size, out_dim)
    

    def forward(self, x, x_length=[]):
        x = self.emb(x)

        x_length = x_length.view(-1).tolist()
        packed_x = pack(x, x_length)
        packed_oe, (h, c) = self.rnn(packed_x)
        oe, x_length = unpack(packed_oe)
        # oe = oe.transpose(0,1)
        # import pdb; pdb.set_trace()
        h = h.transpose(0,1).contiguous().view(h.size(1), -1)
        # import pdb; pdb.set_trace()
        y_ = self.out(h)
        return y_