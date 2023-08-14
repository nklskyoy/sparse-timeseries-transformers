from torch import nn
from src.model.model_util import make_dense

class MLPTSencoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):   
        super(MLPTSencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.dense = make_dense(
            input_size*2, 
            hidden_size, 
            output_size, 
            num_layers, 
            dropout=dropout, 
            activation=nn.ReLU()
        )


    def forward(self, x):
        # X : B x T x 2 x sD
        B, T, D, _ = x.shape
        x = x.reshape(B, T, -1)

        for layer in self.dense:
            x = layer(x)

        # X : B x T x sD
        return x



class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()
        self.input_size = 1
        self.hidden_size = int(dim ** 0.5)
        self.num_layers = 1
        self.output_size = dim

        self.dense = make_dense(
            self.input_size, 
            self.hidden_size, 
            self.output_size, 
            self.num_layers, 
            dropout=0., 
            activation=nn.Tanh()
        )


    def forward(self, x):
        # X : B x T x 1
        #B, T, D = x.shape
        #x = x.view(B, T, D * 2)

        return self.dense(x)
    
