from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch 
from torch import nn
from torch.nn import BCELoss, BCEWithLogitsLoss
from src.model.tess import Tess, PredictHead


class PreTESS(pl.LightningModule):
    def __init__(
            self, dataset,
            ts_dim, time_embedding_dim,
            ts_encoder_hidden_size=128, ts_encoder_num_layers=2, ts_encoder_dropout=0.1,
            n_heads=8,
            prob_mask=0.15,
            alpha = 0.5,
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

        self.bce = BCEWithLogitsLoss(reduction='none')



    def select_random_timesteps(self,shape):
        B, T = shape
        # How many timesteps to select
        t = int(T * self.prob_mask) 

        # Select t random timesteps
        # Generate a random tensor of shape B x T
        random_values = torch.rand(B, T)

        # Argsort along the time dimension to get the permutation indices
        _, idx = random_values.sort(dim=1)
        idx = idx[:, :t]
        idx,_ = idx.sort(dim=1)
        is_masked = torch.full((B,T), False)
        rows = torch.arange(B).unsqueeze(-1).expand_as(idx)
        is_masked[rows, idx] = True
        # Use the first t columns as the indices

        return is_masked.to(self.dataset.device)


    def training_step(self, batch, batch_idx):
        # batch: B x T x 2 x D
        lab = batch[0]
        pid = batch[1]
        timesteps = batch[2]

        spm = lab[:,:,1,:]
        vals = lab[:,:,0,:]

        B,T,_,_ = lab.shape

        # True if masked
        is_masked = self.select_random_timesteps((B,T))
        is_masked_float = is_masked.float().unsqueeze(-1)

        # B x T x D_emb
        x_hat = self.tess(lab, timesteps, is_masked_float)
        vals_pred, spm_pred = self.head(x_hat)
        
        #x_rec = x_rec * mask
        #m_rec = m_rec * mask

        bce = self.bce(spm, spm_pred) * is_masked_float
        mse = (vals - vals_pred)**2 * spm * is_masked_float

        loss = mse + bce * self.alpha 
        loss_per_bin = loss.mean(dim=-1)
        loss_per_pat = loss_per_bin.sum(dim=-1) /is_masked.sum(-1)

        loss = loss_per_pat.mean()
        
        #loss = self.head(x_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer 