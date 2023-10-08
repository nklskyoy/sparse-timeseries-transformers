from src.model.model_util import make_dense
import torch.nn as nn



class PretrainHead(nn.Module):
    def __init__(self, obs, mask):
        super(PretrainHead, self).__init__()
        self.dense_obs = make_dense(**obs)
        self.dense_mask = make_dense(**mask)

        #self.dense_obs = make_dense(shape=[in_channels, 128,128,128, out_channels], last_layer_activation=nn.Tanh)
        #self.dense_mask = make_dense(shape=[in_channels, 128,128,128, out_channels], last_layer_activation=nn.Identity)


    def forward(self, z):
        y = self.dense_obs(z)
        m = self.dense_mask(z)
        return y, m