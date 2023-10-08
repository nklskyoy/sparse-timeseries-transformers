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
from src.util.lr_scheduler import CustomLRSchedule, CyclicLRWithRestarts

# model for finetuning

class TESSFinePhys(pl.LightningModule):
    def __init__(
            self, cofig_optimizer_fn,
            batch_size,
            dataset_train, dataset_val,
            sst_model_path, sst_config, 
            supervised_predict_head= {
                "shape": [256, 128, 128 , 128, 1],
                "last_layer_activation": nn.Identity
            },
            prob_mask=0.15,
            scheduler = CyclicLRWithRestarts,
            start_lr=1e-5, max_lr=1e-4, ramp_up_epochs=200,
        ) -> None:
        
        super(TESSFinePhys, self).__init__()
        
        self.time_embedding_dim = sst_config['time_embedding_dim']
        self.prob_mask = prob_mask

        #sst_model_path = 'checkpoints/pretrain_physionet_m_adamw_nosche-epoch=299-val_loss=1.59.ckpt'
        state_dict = torch.load(sst_model_path)
        state_dict = state_dict['state_dict']
        model = PreTESS( **sst_config)
        model.load_state_dict(state_dict)


        self.batch_size = batch_size

        self.tess = model.tess

        for name, module in self.tess.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.5


        self.mask = model.mask

        self.rep = model.rep

        self.head = make_dense(**supervised_predict_head)
        self.bce = BCEWithLogitsLoss(
            reduction='none', 
            pos_weight=torch.tensor([7.14]).to(dataset_train.device)
        )

        self.auroc = BinaryAUROC()

        self.dataset_train = dataset_train
        self.dataset_val = dataset_val

        self.scheduler = scheduler

        self.start_lr=1e-5 
        self.max_lr=1e-4
        self.ramp_up_epochs=200

        self.mask_prob = 0.2


        self.cofig_optimizer_fn = cofig_optimizer_fn



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


    def create_mask(self,seq_len, T):
            
            seq_len = torch.tensor(seq_len)
            seq_len = seq_len.to(self.dataset_train.device)
            seq_len = seq_len.long()

            B = seq_len.size(0)

            # Expand sequence lengths to a 2D tensor (B, T) where each row contains the sequence length repeated
            expanded_lengths = seq_len.unsqueeze(1).expand(-1, T)

            # Create a tensor of range values (0 to T-1) for each sequence in the batch
            range_tensor = torch.arange(T, device=seq_len.device).expand_as(expanded_lengths)

            # Create a mask to identify valid positions in each sequence (before padding)
            valid_positions = range_tensor < expanded_lengths

            # Generate random values for each sequence
            random_vals = torch.rand_like(range_tensor.float())
            max_vals = random_vals.amax(-1)
            max_vals = max_vals.unsqueeze(1).expand(-1, T)
            fill = torch.arange(T, device=seq_len.device).unsqueeze(1).expand(-1, B).T  < expanded_lengths
            fill = ~fill
            random_vals[fill] = max_vals[fill]

            # Find the threshold value for each sequence that will mask 50% of its values
            _, sorted_indices = random_vals.sort(dim=-1, descending=False)
            half_lengths = (seq_len * self.mask_prob).int()
            threshold_indices = half_lengths.unsqueeze(1).expand(-1, T) - 1
            thresholds = torch.gather(random_vals, 1, sorted_indices).gather(1, threshold_indices.long()).expand(-1, T)

            valid_positions = torch.gather(valid_positions, 1, sorted_indices)

            # Generate mask using the computed thresholds
            mask = (random_vals <= thresholds)

            return mask


    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.tess, norm_type=2)
        self.log_dict(norms)
        norms = grad_norm(self.head, norm_type=2)
        self.log_dict(norms)


    def step(self, batch):
        lab = batch[0]
        seq_len = batch[1]
        pid = batch[2]
        timesteps = batch[3]

        B,T,_,_ = lab.shape

        # True if masked
        is_masked = self.create_mask(seq_len, T)
        is_masked_float = is_masked.float().unsqueeze(-1)

        # B x (T + 1) x D_emb
        x_hat = self.tess(lab, pid, timesteps, is_masked_float, self.mask, self.rep)
        # B x 1 x D
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


    def configure_optimizers(self):
        return self.cofig_optimizer_fn(self.parameters())


        optimizer = torch.optim.AdamW(self.parameters(), lr=self.start_lr, weight_decay=1e-2)
        
        if self.scheduler is not None:
            return optimizer 
        else:
            scheduler = {
                'scheduler': CyclicLRWithRestarts(
                    self.batch_size,
                    optimizer, self.start_lr, self.max_lr, self.ramp_up_epochs),
                'interval': 'epoch',
                'frequency': 1,
            }
            return [optimizer], [scheduler]