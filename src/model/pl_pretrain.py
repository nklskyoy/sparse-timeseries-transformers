from src.model.pl_base import PLBase
from src.model.head import PretrainHead
from torch.nn import BCEWithLogitsLoss



class PLPretrain(PLBase):
    def __init__(self,config_name):
        super(PLPretrain, self).__init__(config_name)

        self.head = PretrainHead(**self._model_params['head'])

        self.bce = BCEWithLogitsLoss(
            reduction='none'
        )


    def step(self, batch):
        # batch: B x T x 2 x D
        lab = batch[0]
        seq_len = batch[1]
        pid = batch[2]
        timesteps = batch[3]

        spm = lab[:,:,1,:]
        vals = lab[:,:,0,:]

        # True if masked
        x_hat, is_masked = self.encoder(lab, pid, timesteps, seq_len)
        #is_masked_float = is_masked.float().unsqueeze(-1)
        
        # B x T x D
        x_hat = x_hat[:,2:,:] # we do not predict static features and rep token

        vals_pred, spm_pred = self.head(x_hat)
        
        bce = self.bce(spm_pred,spm) * is_masked
        mse = (vals - vals_pred)**2 * spm * is_masked

        bce_per_bin = bce.mean(dim=-1)
        mse_per_bin = mse.mean(-1)
        loss_per_bin = mse_per_bin + bce_per_bin * self.alpha

        loss_per_pat = loss_per_bin.sum(dim=-1) / is_masked.sum(-1).sum(-1)

        return loss_per_pat.mean()
    
    
    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
