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




class PredictHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PredictHead, self).__init__()
        pass


    def forward(self, z):
        y = self.dense_obs(z)
        m = self.dense_mask(z)
        return y, m
    

