from torch import nn
from src.model.model_util import make_dense

class MLPTSencoder(nn.Module):
    def __init__(self, shape, dropout):   
        super(MLPTSencoder, self).__init__()
        self.shape = shape
        self.dropout = dropout

        self.dense = make_dense(
            shape,
            dropout=dropout, 
            activation=nn.ReLU
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
    def __init__(self, shape):
        super(TimeEmbedding, self).__init__()
        self.shape = shape

        self.dense = make_dense(
            shape,
            dropout=0., 
            activation=nn.Tanh
        )


    def forward(self, x):
        # X : B x T x 1
        #B, T, D = x.shape
        #x = x.view(B, T, D * 2)

        return self.dense(x)
    
