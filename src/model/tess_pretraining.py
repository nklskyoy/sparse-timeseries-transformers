from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch 
from torch import nn
from torch.nn import BCELoss
from src.model.tess import Tess, PredictHead


class PreTESS(pl.LightningModule):
    def __init__(
            self, dataset,
            ts_dim, time_embedding_dim,
            ts_encoder_hidden_size=128, ts_encoder_num_layers=2, ts_encoder_dropout=0.1,
            n_heads=8,
            prob_mask=0.15,
            alpha = 0.1,
    ) -> None:
        
        super(PreTESS, self).__init__()
        
        self.dataset = dataset

        self.tess = Tess(
            dataset,
            ts_dim, time_embedding_dim,
            ts_encoder_hidden_size, ts_encoder_num_layers, ts_encoder_dropout,
            n_heads
        )

        self.prob_mask = prob_mask
        self.alpha = alpha

        self.head = PredictHead(
            time_embedding_dim, ts_dim
        )

        self.bce = BCELoss()



    def get_masked(self,shape):
        return torch.bernoulli(torch.full(shape, 1- self.prob_mask)).to(self.dataset.device)


    def training_step(self, batch, batch_idx):
        # batch: B x T x 2 x D
        lab = batch[0]
        pid = batch[1]
        timesteps = batch[2]

        m = lab[:,:,1,:]
        vals = lab[:,:,0,:]

        B,T,_,_ = lab.shape

        mask = self.get_masked((B,T,1))
        #mask_bool = mask.int().bool()

        # B x T x D_emb
        x_hat = self.tess(lab,timesteps,mask)
        x_rec, m_rec = self.head(x_hat)
        x_rec = x_rec * mask
        m_rec = m_rec * mask


        # TODO : is this averaging correct?
        bce = self.bce(m,m_rec)
        mse = (vals - x_rec)**2 * m

        loss = bce + self.alpha * mse.mean()
        #loss = self.head(x_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer