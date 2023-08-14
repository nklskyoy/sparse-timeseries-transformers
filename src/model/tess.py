from torch import nn
from src.model.ts_encoder import MLPTSencoder, TimeEmbedding
import torch
from src.model.model_util import make_dense




class PredictHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PredictHead, self).__init__()
        self.dense_obs = make_dense(in_channels, hidden_size= 128, output_size=out_channels,num_layers=1)
        self.dense_mask = make_dense(in_channels, hidden_size= 128, output_size=out_channels,num_layers=1)


    def forward(self, z):
        y = self.dense_obs(z)
        m = self.dense_mask(z)
        return y, m




class Tess(nn.Module):
    def __init__(self, 
                 ts_dim, time_embedding_dim,
                 ts_encoder_hidden_size=128, ts_encoder_num_layers=2, ts_encoder_dropout=0.1,
                 n_heads=8):
        super(Tess, self).__init__()
        self.ts_dim = ts_dim

        self.ts_encoder = MLPTSencoder(
            input_size=ts_dim, 
            hidden_size=ts_encoder_hidden_size, 
            output_size=time_embedding_dim,
            num_layers=ts_encoder_num_layers, 
            dropout=0.1
        )

        self.time_embedding = TimeEmbedding(dim=ts_encoder_hidden_size)

        self.mha = nn.MultiheadAttention(
            embed_dim=ts_encoder_hidden_size, num_heads=n_heads, dropout=0.1
        )


    def forward(self, x, t):
        # x : B x T x 2 x D
        # t : B x T x 1

        # encode time series
        x = self.ts_encoder(x)
        t = self.time_embedding(t)

        x = x + t

        # apply multihead attention
        x = x.permute(1, 0, 2)
        x, _ = self.mha(x, x, x)

        return x
    

