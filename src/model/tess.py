from torch import nn
from src.model.ts_encoder import MLPTSencoder, TimeEmbedding
import torch
from src.model.model_util import make_dense, make_mha
from torch.nn.functional import sigmoid,tanh



class PredictHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PredictHead, self).__init__()
        self.dense_obs = make_dense(shape=[in_channels, 128,128,128, out_channels], last_layer_activation=nn.Tanh)
        self.dense_mask = make_dense(shape=[in_channels, 128,128,128, out_channels], last_layer_activation=nn.Identity)


    def forward(self, z):
        y = self.dense_obs(z)
        #y = tanh(y)

        m = self.dense_mask(z)
        return y, m




class Tess(nn.Module):
    def __init__(self, dataset, 
                 ts_dim, time_embedding_dim, static_dim,
                 ts_encoder={'shape': [], 'dropout': 0.1}, 
                 time_embedding={'shape' : []}, 
                 static_feature_encoder={'shape' : [], 'dropout': 0.1, 'last_layer_activation': nn.Identity}, 
                 mha={'num_layers' : 4, 'n_heads': 8, 'dropout': 0.1}):

        super(Tess, self).__init__()
        
        self.ts_dim = ts_dim

        self.dataset = dataset
        
        self.ts_encoder = MLPTSencoder(
            shape=ts_encoder['shape'],
            dropout=ts_encoder['dropout']
        )

        self.time_embedding = TimeEmbedding(
            shape=time_embedding['shape']
        )

        self.static_feature_encoder = make_dense(
            shape=static_feature_encoder['shape'],
        )

        self.mha = make_mha(
            n_layers=mha['num_layers'],
            input_size=time_embedding_dim,
            rep_size=time_embedding_dim,
            n_heads=mha['n_heads'],
            dropout=mha['dropout']
        )


    def forward(self, lab, pid, t, is_masked=None, mask=None, rep=None):
        # x : B x T x 2 x D
        # t : B x T x 1
        # is_masked: 1. and 0. tensor of shape B x T x 1
        
        B = lab.shape[0]
        
        # encode time series
        x = self.ts_encoder(lab)
        t = self.time_embedding(t)

        x = x + t

        if is_masked is not None:
            #mask = torch.full_like(x, -2, device=self.dataset.device)
            x = (1 - is_masked) * x + is_masked @ mask.T
        
        # encode static features
        s = self.static_feature_encoder(pid)

        # concatenate time series and static features
        x = torch.cat([x, s.unsqueeze(-2)], dim=-2)

        if rep is not None:
            rep = rep.unsqueeze(0).repeat(B,1,1)
            x = torch.cat([x, rep], dim=-2)

        # apply multihead attention
        x = self.mha(x)

        return x
    

