import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F



class TextCNNEncoder(nn.Module):
    def __init__(self, emb, out_dim, cnn_filter_num=64, dropout_p=0.1):
        super().__init__()
        # padding_idx: If given, pads the output with the embedding vector at
        #              padding_idx (initialized to zeros) whenever it encounters the index.
        # self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        num_embeddings = emb.num_embeddings
        embedding_dim = emb.embedding_dim
        
        self.emb = emb
        self.out_dim = out_dim
        self.conv_w2 = nn.Sequential(
                            nn.Conv1d(embedding_dim, cnn_filter_num, 2),
                            # nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.AdaptiveMaxPool1d(1))
        self.conv_w3 = nn.Sequential(
                            nn.Conv1d(embedding_dim, cnn_filter_num, 3),
                            # nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.AdaptiveMaxPool1d(1))
        self.conv_w4 = nn.Sequential(
                            nn.Conv1d(embedding_dim, cnn_filter_num, 4),
                            # nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.AdaptiveMaxPool1d(1))
        self.conv_w5 = nn.Sequential(
                            nn.Conv1d(embedding_dim, cnn_filter_num, 5),
                            # nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.AdaptiveMaxPool1d(1))
        self.conv_w6 = nn.Sequential(
                            nn.Conv1d(embedding_dim, cnn_filter_num, 6),
                            # nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.AdaptiveMaxPool1d(1))
        self.out = nn.Linear(cnn_filter_num * 5, out_dim)

    def forward(self, x, x_length=[]):
        # import pdb; pdb.set_trace()
        batch_size = x.size(0)
        x = self.emb(x)
        x = x.permute(0, 2, 1)
        # x = x.view(batch_size, 1, *x.shape[1:])
        x_w2 = self.conv_w2(x)
        x_w3 = self.conv_w3(x)
        x_w4 = self.conv_w4(x)
        x_w5 = self.conv_w5(x)
        x_w6 = self.conv_w6(x)
        x = torch.cat([x_w2, x_w3, x_w4, x_w5, x_w6], dim=1)
        x = x.view(batch_size, -1)
        y_ = self.out(x)
        return y_