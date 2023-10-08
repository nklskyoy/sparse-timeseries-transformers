from src.model.pl_base import PLBase
from src.model.model_util import make_dense
from torch.nn import BCEWithLogitsLoss
from torcheval.metrics import BinaryAUROC
import torch



class PLFinetune(PLBase):
    def __init__(self, config_name):
        super(PLFinetune, self).__init__(config_name)

        self.head = make_dense(**self._model_params['head'])

        self.bce = BCEWithLogitsLoss(
            reduction='none', 
            pos_weight=torch.tensor([7.14]).to(self.device)
        )

        self.auroc = BinaryAUROC()



    def step(self, batch):
        lab = batch[0]
        seq_len = batch[1]
        pid = batch[2]
        timesteps = batch[3]

        B,T,_,_ = lab.shape

        # B x (T + 1) x D_emb
        x_hat, _ = self.encoder(lab, pid, timesteps, seq_len)        # B x 1 x D
        rep_hat = x_hat[:,0,:]

        # B x 1
        pred = self.head(rep_hat)

        return pred
        # B x 1



    def training_step(self, batch, batch_idx):
        target = batch[4]

        pred = self.step(batch)
        bce = self.bce(pred,target) #* is_masked_float

        loss = bce.mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    

       

    def validation_step(self, batch, batch_idx):
        target = batch[4]

        pred = self.step(batch)
        bce = self.bce(pred,target) #* is_masked_float

        loss = bce.mean()
        self.log('valid_loss', loss, on_epoch=True, prog_bar=True, logger=True)

        pred = torch.sigmoid(pred)
        self.auroc.update(pred.reshape(-1), target.reshape(-1))
        auroc = self.auroc.compute()
        self.log('valid_auroc', auroc, on_epoch=True, prog_bar=True, logger=True)   

        return loss
