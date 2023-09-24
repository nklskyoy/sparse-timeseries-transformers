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
            prob_mask=0.5,
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
        self.rep = nn.Parameter(torch.ones(1, time_embedding_dim))

        self.prob_mask = prob_mask
        
        self.alpha = alpha

        self.head = PredictHead(
            time_embedding_dim, ts_dim
        )

        self.bce = BCEWithLogitsLoss(reduction='none')




    def create_mask_fix(self, seq_len, T):
        # Create a tensor of range values (0 to T-1) for each sequence in the batch
        range_tensor = torch.arange(T, device=seq_len.device).expand(seq_len.size(0), -1)
        
        # Create a mask to identify valid positions in each sequence (before padding)
        valid_positions = range_tensor < seq_len.unsqueeze(1)
        
        # Determine the number of events to mask for each sequence
        half_lengths = (seq_len / 2).int()
        
        # Create a tensor of ones and zeros for masking
        mask_values = torch.cat([torch.ones(half_lengths.sum().item(), dtype=torch.bool, device=seq_len.device),
                                torch.zeros((seq_len - half_lengths).sum().item(), dtype=torch.bool, device=seq_len.device)])

        # Determine how many times each sequence should be repeated for reshuffling
        repeats = torch.repeat_interleave(seq_len, dim=0)
        
        # Shuffle each sequence's mask values independently
        shuffled_mask = torch.repeat_interleave(mask_values[torch.randperm(mask_values.size(0))], repeats=repeats).view(seq_len.size(0), T)

        # Ensure the mask is applied only to valid positions (before padding)
        mask = shuffled_mask & valid_positions

        return mask



    def create_mask(self,seq_len, T):
        
        seq_len = torch.tensor(seq_len)
        seq_len = seq_len.to(self.dataset.device)
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
        half_lengths = (seq_len / 2).int()
        threshold_indices = half_lengths.unsqueeze(1).expand(-1, T) - 1
        thresholds = torch.gather(random_vals, 1, sorted_indices).gather(1, threshold_indices.long()).expand(-1, T)

        valid_positions = torch.gather(valid_positions, 1, sorted_indices)

        # Generate mask using the computed thresholds
        mask = (random_vals <= thresholds)

        return mask



    def select_random_timesteps(self,shape, seq_len):
        B, T = shape
        # How many timesteps to select
        t = int(T * self.prob_mask) 
        
        seq_len = torch.tensor(seq_len)
        seq_len = seq_len.to(self.dataset.device)
        seq_len = seq_len.long()

        t_ = seq_len * self.prob_mask
        t_ = t_.int()
        t_ = t - t_
        
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

        return is_masked.to(self.dataset.device)


    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.tess, norm_type=2)
        self.log_dict(norms)
        norms = grad_norm(self.head, norm_type=2)
        self.log_dict(norms)


    def step(self, batch):
        # batch: B x T x 2 x D
        lab = batch[0]
        seq_len = batch[1]
        pid = batch[2]
        timesteps = batch[3]

        spm = lab[:,:,1,:]
        vals = lab[:,:,0,:]

        B,T,_,_ = lab.shape

        # True if masked
        #is_masked = self.select_random_timesteps((B,T),seq_len)
        is_masked = self.create_mask(seq_len, T)
        is_masked_float = is_masked.float().unsqueeze(-1)

        # B x (T + 1) x D_emb
        x_hat = self.tess(lab, pid, timesteps, is_masked_float, self.mask, self.rep)
        # B x T x D
        x_hat = x_hat[:,1:-1,:] # we do not predict static features and rep token

        vals_pred, spm_pred = self.head(x_hat)
        
        #x_rec = x_rec * mask
        #m_rec = m_rec * mask

        bce = self.bce(spm_pred,spm) #* is_masked_float
        mse = (vals - vals_pred)**2 #* spm #* is_masked_float

        loss =  mse + bce * self.alpha 
        loss_per_bin = loss.mean(dim=-1)
        loss_per_pat = loss_per_bin.sum(dim=-1) /is_masked.sum(-1)

        loss = loss_per_pat.mean()
        if torch.isinf(loss):
            print("inf loss")

        return loss


    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-6)
        return optimizer 