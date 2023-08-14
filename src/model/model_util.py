from torch import nn


def make_dense(input_size, hidden_size, output_size, num_layers, dropout=0., activation=nn.ReLU()):
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
        layers.append(nn.ReLU())

    return nn.Sequential(*layers)