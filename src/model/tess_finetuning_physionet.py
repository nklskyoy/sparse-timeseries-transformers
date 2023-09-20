from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch 
from torch import nn
from torch.nn import BCELoss, BCEWithLogitsLoss
from src.model.tess import Tess, PredictHead
from src.model.model_util import make_dense
from pytorch_lightning.utilities import grad_norm
from src.model.tess_pretraining import PreTESS
from torcheval.metrics import BinaryAUROC
from src.util.lr_scheduler import CustomLRSchedule

# model for finetuning

class TESSFinePhys(pl.LightningModule):
    def __init__(
            self, dataset_train, dataset_val,
            sst_model_path, sst_config, 
            supervised_predict_head= {
                "shape": [256, 128, 128 , 128, 1],
                "last_layer_activation": nn.Identity
            },
            prob_mask=0.15,
            scheduler = CustomLRSchedule,
            start_lr=1e-5, max_lr=1e-4, ramp_up_epochs=200,
        ) -> None:
        
        super(TESSFinePhys, self).__init__()
        
        self.time_embedding_dim = sst_config['time_embedding_dim']
        self.prob_mask = prob_mask

        sst_model_path = 'checkpoints/pretrain_physionet_m_adamw_nosche-epoch=299-val_loss=1.59.ckpt'
        state_dict = torch.load(sst_model_path)
        state_dict = state_dict['state_dict']
        model = PreTESS( **sst_config)
        model.load_state_dict(state_dict)

        self.tess = model.tess

        for name, module in self.tess.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.5


        self.mask = model.mask

        self.rep = nn.Parameter(torch.ones(1, self.time_embedding_dim))

        self.head = make_dense(**supervised_predict_head)
        self.bce = BCEWithLogitsLoss(reduction='none')

        self.auroc = BinaryAUROC()

        self.dataset_train = dataset_train
        self.dataset_val = dataset_val

        self.scheduler = scheduler

        self.start_lr=1e-5 
        self.max_lr=1e-4
        self.ramp_up_epochs=1000




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

        return is_masked.to(self.dataset_train.device)


    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.tess, norm_type=2)
        self.log_dict(norms)
        norms = grad_norm(self.head, norm_type=2)
        self.log_dict(norms)


    def step(self, batch):
        lab = batch[0]
        pid = batch[1]
        timesteps = batch[2]
        target = batch[3]

        B,T,_,_ = lab.shape

        # True if masked
        is_masked = self.select_random_timesteps((B,T))
        is_masked_float = is_masked.float().unsqueeze(-1)

        # B x (T + 1) x D_emb
        x_hat = self.tess(lab, pid, timesteps, is_masked_float, self.mask, self.rep)
        # B x 1 x D
        rep_hat = x_hat[:,-1,:]

        # B x 1
        pred = self.head(rep_hat)

        return pred
        # B x 1



    def training_step(self, batch, batch_idx):
        target = batch[3]

        pred = self.step(batch)
        bce = self.bce(pred,target) #* is_masked_float

        loss = bce.mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    

    def validation_step(self, batch, batch_idx):
        target = batch[3]

        pred = self.step(batch)
        bce = self.bce(pred,target) #* is_masked_float

        loss = bce.mean()
        self.log('valid_loss', loss, on_epoch=True, prog_bar=True, logger=True)

        pred = torch.sigmoid(pred)
        self.auroc.update(pred.reshape(-1), target.reshape(-1))
        auroc = self.auroc.compute()
        self.log('valid_auroc', auroc, on_epoch=True, prog_bar=True, logger=True)   

        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.start_lr, weight_decay=1e-2)
        
        if self.scheduler is not None:
            return optimizer 
        else:
            scheduler = {
                'scheduler': CustomLRSchedule(optimizer, self.start_lr, self.max_lr, self.ramp_up_epochs),
                'interval': 'epoch',
                'frequency': 1,
            }
            return [optimizer], [scheduler]