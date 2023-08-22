from torch import nn
import torch

def make_dense(input_size, hidden_size, output_size, num_layers, dropout=0., activation=nn.ReLU, last_layer_activation=nn.ReLU):
    layers = []
    input_dim = input_size

    for _ in range(num_layers):

        if _ == 0:
            source_size = input_size
        else:
            source_size = hidden_size

        if _ == num_layers - 1:
            target_size = output_size
        else:
            target_size = hidden_size


        layers.append(nn.Linear(source_size, target_size))
        if _ != num_layers - 1:
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
        
        self.q = make_dense(input_size, rep_size, rep_size, 1, dropout=dropout)
        self.k = make_dense(input_size, rep_size, rep_size, 1, dropout=dropout)
        self.v = make_dense(input_size, rep_size, rep_size, 1, dropout=dropout)

        self.mha = nn.MultiheadAttention(
            embed_dim=rep_size,
            num_heads=n_heads,
            dropout=0.1, 
            batch_first=True
        )

        self.o = make_dense(rep_size, rep_size, rep_size, 1, dropout=dropout, last_layer_activation=nn.Identity)

        self.layer_norm = nn.LayerNorm(rep_size)


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
        x, _ = self.mha(q, k, v)

        x = self.o(x)
        x = self.layer_norm(x)

        return x



class PredictHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PredictHead, self).__init__()
        pass


    def forward(self, z):
        y = self.dense_obs(z)
        m = self.dense_mask(z)
        return y, m
    

