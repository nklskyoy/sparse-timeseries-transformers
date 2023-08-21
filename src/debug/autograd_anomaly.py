# %%
from src.physionet_dataset import PhysioNetDataset, CollateFn
from torch.utils.data import DataLoader
import torch 
from torch import autograd
from torch import nn
from torch.nn import BCELoss, BCEWithLogitsLoss
from src.model.tess import Tess, PredictHead
import os


class PreTESS(nn.Module):
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




    def get_masked(self,shape):
        return torch.bernoulli(torch.full(shape, 1- self.prob_mask)).to(self.dataset.device)


    def forward(self, batch, batch_idx):
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
        return x_rec, m_rec, vals, m

        # TODO : is this averaging correct?
        bce = self.bce(m,m_rec)
        mse = (vals - x_rec)**2 * m

        loss = bce.mean() + self.alpha * mse.mean()
        #loss = self.head(x_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer


# %%
freq='1H'

lab = PhysioNetDataset(
    root_path={
        'raw' : os.path.join('data','physionet.org','files', 'set-a'),
        'data' : os.path.join('data','physionet.org'),
    },
    dataset_name='set-a',
    freq=freq,
    write_to_disc=False
)


model = PreTESS(
    dataset=lab,
    ts_dim=36, time_embedding_dim=2048,
    ts_encoder_hidden_size=2048, ts_encoder_num_layers=2, ts_encoder_dropout=0.1,
    n_heads=8,
    prob_mask=0.15
)

collate_fn = CollateFn(device=lab.device)
dataloader = DataLoader(lab, batch_size=128, shuffle=True, num_workers=0, collate_fn=collate_fn)


# %%
bce = BCELoss(reduction='none')
bcell = BCEWithLogitsLoss(reduction='none')
mse = nn.MSELoss(reduction='none')
batch = next(iter(dataloader))




# %%
import torchviz
x_rec, m_rec, vals, m = model(batch,0)
torchviz.make_dot(bce(m,m_rec).mean())

# %%
with autograd.detect_anomaly():
    x_rec, m_rec, vals, m = model(batch,0)
    bce(m,m_rec).mean().backward()





# %%
def register_hooks(module, grad_list):
    for name, layer in module.named_children():
        layer.register_backward_hook(lambda module, grad_input, grad_output: grad_list.append((module, grad_input, grad_output)))
        register_hooks(layer, grad_list)

grad_list = []
register_hooks(model, grad_list)

x_rec, m_rec, vals, m = model(batch,0)
loss = bcell(m,m_rec).mean()
loss.backward()

for module, grad_input, grad_output in grad_list:
    print(f"Module: {module}")
    print(f"Gradient Input: {grad_input}")
    print(f"Gradient Output: {grad_output}")
# %%

# %%
