from torch import nn
import torch
from src.model.model_util import make_dense, make_mha
from torch.nn.functional import sigmoid,tanh
import math


def create_mask(seq_len, T, p_mask,device):
        
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
        half_lengths = (seq_len * p_mask).int()
        threshold_indices = half_lengths.unsqueeze(1).expand(-1, T) - 1
        thresholds = torch.gather(random_vals, 1, sorted_indices).gather(1, threshold_indices.long()).expand(-1, T)

        valid_positions = torch.gather(valid_positions, 1, sorted_indices)

        # Generate mask using the computed thresholds
        mask = (random_vals <= thresholds)

        return mask





class Encoder(nn.Module):
    def __init__(self, 
                 device,
                 d, s, z,
                 emb_time_bin={'shape': [], 'dropout': 0.1}, 
                 emb_demographic={'shape' : [], 'dropout': 0.1, 'last_layer_activation': nn.Identity}, 
                 pos_encoding={'shape' : [], 'dropout': 0.1},
                 mha={'num_layers' : 4, 'n_heads': 8, 'dropout': 0.1},
                 p_mask=0.2,
                 with_rep=True, with_mask=True
                 ):

        super(Encoder, self).__init__()
        
        self.device = device

        self.d = d
        self.s = s
        self.z = z
        self.p_mask = p_mask

        self.with_mask = with_mask
        self.with_rep = with_rep

        self.emb_time_bin = make_dense(
            **emb_time_bin
        )

        self.emb_demographic = make_dense(
            **emb_demographic
        )

        self.pos_encoding = make_dense(
            **pos_encoding
        )

        self.mha = make_mha(
            **mha
        )
        stdv = math.sqrt(2. / self.z)

        if with_mask:
            self.mask = nn.Parameter(torch.randn(z, 1) * stdv, requires_grad=True).to(device)
            #nn.init.kaiming_normal_(self.mask, nonlinearity='relu').to(device)
        else:
            self.mask = None

        if with_rep:
            self.rep = nn.Parameter(torch.randn(1, z) * stdv)
            #nn.init.kaiming_normal_(self.rep, nonlinearity='relu')
        else:
            self.rep = None




    def forward(self, lab, pid, t, seq_len=None):
        # x : B x T x 2 x D
        # t : B x T x 1
        # is_masked: 1. and 0. tensor of shape B x T x 1
        
        B,T,_,_ = lab.shape        
        
        # encode time series
        x = self.emb_time_bin(lab.reshape(B, T, -1))
        t = self.pos_encoding(t)
        s = self.emb_demographic(pid)

        x = x + t


        if self.with_mask:
            is_masked = create_mask(seq_len, T,self.p_mask ,self.device)
            is_masked = is_masked.float().unsqueeze(-1)
            x = (1 - is_masked) * x + is_masked @ self.mask.T
        
        # concatenate time series and static features
        x = torch.cat([s.unsqueeze(-2), x], dim=-2)

        if self.rep is not None:
            #rep = rep.unsqueeze(0).repeat(B,1,1)
            x = torch.cat([self.rep.repeat(B,1).unsqueeze(-2), x], dim=-2)
            #torch.cat([self.rep, x], dim=-2)

        # apply multihead attention
        x = self.mha(x)

        if self.with_mask:
            return x, is_masked
        else:
            return x
    

