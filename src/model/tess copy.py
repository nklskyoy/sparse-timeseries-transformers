from torch import nn
from src.model.ts_encoder import MLPTSencoder, TimeEmbedding
import torch
from src.model.model_util import make_dense, make_mha
from torch.nn.functional import sigmoid,tanh



class PredictHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PredictHead, self).__init__()
        self.dense_obs = make_dense(in_channels, hidden_size= 128, output_size=out_channels,num_layers=3)
        self.dense_mask = make_dense(in_channels, hidden_size= 128, output_size=out_channels,num_layers=3, last_layer_activation=nn.Identity)


    def forward(self, z):
        y = self.dense_obs(z)
        y = tanh(y)

        m = self.dense_mask(z)
        return y, m




class Tess(nn.Module):
    def __init__(self, dataset,
                 ts_dim, time_embedding_dim,
                 ts_encoder_hidden_size=128, ts_encoder_num_layers=2, ts_encoder_dropout=0.1,
                 n_heads=8):
        super(Tess, self).__init__()
        
        self.ts_dim = ts_dim

        self.dataset = dataset
        
        self.ts_encoder = MLPTSencoder(
            input_size=ts_dim, 
            hidden_size=ts_encoder_hidden_size, 
            output_size=time_embedding_dim,
            num_layers=ts_encoder_num_layers, 
            dropout=0.1
        )

        self.time_embedding = TimeEmbedding(dim=ts_encoder_hidden_size)

        self.mha = make_mha(
            n_layers=4,
            input_size=ts_encoder_hidden_size,
            rep_size=ts_encoder_hidden_size,
            n_heads=n_heads,
            dropout=0.1
        )
        



    def forward(self, x, t, is_masked=None, mask=None):
        # x : B x T x 2 x D
        # t : B x T x 1
        # is_masked: 1. and 0. tensor of shape B x T x 1

        # encode time series
        x = self.ts_encoder(x)
        t = self.time_embedding(t)

        x = x + t

        if is_masked is not None:
            #mask = torch.full_like(x, -2, device=self.dataset.device)
            x = (1 - is_masked) * x + is_masked @ mask.T

        # apply multihead attention
        x = self.mha(x)

        return x
    

