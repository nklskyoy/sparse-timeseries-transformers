from torch import nn
import torch

def make_dense(shape, dropout=0., activation=nn.ReLU, last_layer_activation=nn.ReLU):
    if len(shape) < 2:
        return None
    
    layers = []

    num_layers = len(shape) - 1
    source_size = shape[0]

    for i in range(num_layers):
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(shape[i], shape[i+1]))
        if i != num_layers - 1:
            layers.append(activation())
        else:
            layers.append(last_layer_activation())

    return nn.Sequential(*layers)



def mask_token(dim):
    return torch.full(dim, -10.)



def make_mha(n_layers, input_size, rep_size, n_heads, dropout=0.1):
    layers = []
    for _ in range(n_layers):
        if _ == 0:
            layers.append(MHA(input_size, rep_size, n_heads, dropout=dropout))
        else:
            layers.append(MHA(rep_size, rep_size, n_heads, dropout=dropout))

    return nn.Sequential(*layers)


class MHA(nn.Module):
    def __init__(self, input_size, rep_size, n_heads, dropout=0.1):
        super(MHA, self).__init__()
        
        self.q = make_dense(shape=[input_size, rep_size], dropout=dropout)
        self.k = make_dense(shape=[input_size, rep_size], dropout=dropout)
        self.v = make_dense(shape=[input_size, rep_size], dropout=dropout)

        self.mha = nn.MultiheadAttention(
            embed_dim=rep_size,
            num_heads=n_heads,
            dropout=0.1, 
            batch_first=True
        )

        self.o = make_dense( shape=[rep_size, rep_size], dropout=dropout, last_layer_activation=nn.Identity)

        self.layer_norm1 = nn.LayerNorm(rep_size)
        self.layer_norm2 = nn.LayerNorm(rep_size)

        self.ff = make_dense(shape=[rep_size, rep_size], dropout=dropout, last_layer_activation=nn.ReLU)


    def forward(self, x, mask=None):
        # x : B x T x D
        # mask : B x T x 1 tensor with 0. and 1.
        # x : B x T x D
        # mask : B x T x 1 tensor with 0. and 1.

        B, T, D = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # B x T x D
        step1, _ = self.mha(q, k, v)

        step1 = self.o(step1)
        step1 = step1 + x         # residual
        step1 = self.layer_norm1(step1)

        step2 = self.ff(step1)
        step2 = step2 + step1     # residual
        step2 = self.layer_norm2(step2)

        return step2



class PredictHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PredictHead, self).__init__()
        pass


    def forward(self, z):
        y = self.dense_obs(z)
        m = self.dense_mask(z)
        return y, m
    

