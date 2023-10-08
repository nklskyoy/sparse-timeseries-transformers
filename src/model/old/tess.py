from torch import nn
from src.model.ts_encoder import MLPTSencoder, TimeEmbedding
import torch
from src.model.model_util import make_dense, make_mha
from torch.nn.functional import sigmoid,tanh





def create_mask(seq_len, T, device):
        seq_len = torch.tensor(seq_len)
        seq_len = seq_len.to(device)
        seq_len = seq_len.long()

        B = seq_len.size(0)

        # Expand sequence lengths to a 2D tensor (B, T) where each row contains the sequence length repeated
        expanded_lengths = seq_len.unsqueeze(1).expand(-1, T)

        # Create a tensor of range values (0 to T-1) for each sequence in the batch
        range_tensor = torch.arange(T, device=seq_len.device).expand_as(expanded_lengths)

        # Create a mask to identify valid positions in each sequence (before padding)
        valid_positions = range_tensor < expanded_lengths

        # Generate random values for each sequence
        random_vals = torch.rand_like(range_tensor.float())
        max_vals = random_vals.amax(-1)
        max_vals = max_vals.unsqueeze(1).expand(-1, T)
        fill = torch.arange(T, device=seq_len.device).unsqueeze(1).expand(-1, B).T  < expanded_lengths
        fill = ~fill
        random_vals[fill] = max_vals[fill]

        # Find the threshold value for each sequence that will mask 50% of its values
        _, sorted_indices = random_vals.sort(dim=-1, descending=False)
        half_lengths = (seq_len / 2).int()
        threshold_indices = half_lengths.unsqueeze(1).expand(-1, T) - 1
        thresholds = torch.gather(random_vals, 1, sorted_indices).gather(1, threshold_indices.long()).expand(-1, T)

        valid_positions = torch.gather(valid_positions, 1, sorted_indices)

        # Generate mask using the computed thresholds
        mask = (random_vals <= thresholds)

        return mask





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
        x = torch.cat([s.unsqueeze(-2), x], dim=-2)

        if rep is not None:
            rep = rep.unsqueeze(0).repeat(B,1,1)
            x = torch.cat([rep, x], dim=-2)

        # apply multihead attention
        x = self.mha(x)

        return x
    

