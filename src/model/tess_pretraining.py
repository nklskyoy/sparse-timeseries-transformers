from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch 
from torch import nn
from torch.nn import BCELoss, BCEWithLogitsLoss
from src.model.tess import Tess, PredictHead
from pytorch_lightning.utilities import grad_norm




class PreTESS(pl.LightningModule):
    def __init__(
            self, dataset,
            ts_dim, time_embedding_dim, static_dim,
            ts_encoder={'shape': [], 'dropout': 0.1}, 
            time_embedding={'shape' : []}, 
            static_feature_encoder={'shape' : [], 'dropout': 0.1, 'last_layer_activation': nn.Identity}, 
            mha={'num_layers' : 4, 'n_heads': 8, 'dropout': 0.1},
            prob_mask=0.15,
            alpha = 0.2,
    ) -> None:
        
        super(PreTESS, self).__init__()
        
        self.dataset = dataset

        self.tess = Tess(
            dataset,
            ts_dim, time_embedding_dim, static_dim,
            ts_encoder=ts_encoder, 
            time_embedding=time_embedding,
            static_feature_encoder=static_feature_encoder, 
            mha=mha
        )

        D = self.dataset.ts_dim
        self.mask = nn.Parameter(torch.ones(time_embedding_dim, 1))
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

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.tess, norm_type=2)
        self.log_dict(norms)
        norms = grad_norm(self.head, norm_type=2)
        self.log_dict(norms)

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

        # B x (T + 1) x D_emb
        x_hat = self.tess(lab, pid, timesteps, is_masked_float, self.mask)
        # B x T x D
        x_hat = x_hat[:,:-1,:] # we do not predict static features

        vals_pred, spm_pred = self.head(x_hat)
        
        #x_rec = x_rec * mask
        #m_rec = m_rec * mask

        bce = self.bce(spm_pred,spm) #* is_masked_float
        mse = (vals - vals_pred)**2 #* spm #* is_masked_float

        loss =  mse + bce * self.alpha 
        loss_per_bin = loss.mean(dim=-1)
        loss_per_pat = loss_per_bin.sum(dim=-1) /is_masked.sum(-1)

        loss = loss_per_pat.mean()
        
        #loss = self.head(x_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-6)
        return optimizer 