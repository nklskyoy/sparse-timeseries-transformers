from torch import nn


class MLPTSencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):   
        super(MLPTSencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        layers = []
        input_dim = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_size))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
            input_dim = hidden_size  # Update input_dim for the next layer

        self.dense = nn.ModuleList(layers)

        def forward(self, x):
            # X : B x T x D x 2
            B, T, D, _ = x.shape
            x = x.view(B, T, D * 2)

            for layer in self.dense:
                x = layer(x)
            return x



class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()
        self.input_size = 1
        self.hidden_size = int(dim ** 0.5)
        self.num_layers = 1
        self.output_size = dim

        layers = []
        input_dim = self.input_size
        for _ in range(self.num_layers):

            if _ == 0:
                source_size = self.input_size
            else:
                source_size = self.hidden_size

            if _ == self.num_layers - 1:
                target_size = self.output_size
            else:
                target_size = self.hidden_size


            layers.append(nn.Linear(source_size, target_size))
            layers.append(nn.Tanh())


        def forward(self, x):
            # X : B x 1
            B, T, D, _ = x.shape
            x = x.view(B, T, D * 2)

            for layer in self.dense:
                x = layer(x)
            return x
    
        