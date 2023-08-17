# %%
from src.physionet_dataset import PhysioNetDataset, collate_fn
from src.model.tess_pretraining import PreTESS
from torch.utils.data import DataLoader
import os
#import pytorch_lightning as pl
from pytorch_lightning import Trainer


# %%
freq='1H'

# %%

model = PreTESS(
    ts_dim=36, time_embedding_dim=128,
    ts_encoder_hidden_size=128, ts_encoder_num_layers=2, ts_encoder_dropout=0.1,
    n_heads=8,
    prob_mask=0.15
)


lab = PhysioNetDataset(
    root_path={
        'raw' : os.path.join('data','physionet.org','files', 'set-a'),
        'data' : os.path.join('data','physionet.org'),
    },
    dataset_name='set-a',
    freq=freq,
    write_to_disc=False
)

# %%
dataloader = DataLoader(lab, batch_size=128, shuffle=True, num_workers=1, collate_fn=collate_fn)
#i = iter(dataloader)
#x,t = next(i)
# %%
trainer = Trainer(accelerator="mps", devices=1)
trainer.fit(model, dataloader)