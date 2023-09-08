# %%
from src.physionet_dataset import PhysioNetDataset
from src.model.tess_pretraining import PreTESS
from torch.utils.data import DataLoader
import torch.nn as nn
import os


# %%
freq='1H'

# %%

model = PreTESS(
    ts_dim=36, time_embedding_dim=512, static_dim=5,
    ts_encoder={'shape': [36, 256, 256, 512], 'dropout': 0.1},
    time_embedding={'shape' : [1, int(512 ** 0.5), 512]},
    static_feature_encoder={'shape' : [5,512], 'dropout': 0.1, 'last_layer_activation': nn.Identity},
    mha={'num_layers' : 4, 'n_heads': 8, 'dropout': 0.1},
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
x, pid, T = lab[0]
batch_x = x.unsqueeze(0)
batch_t = T.unsqueeze(0)
model.training_step((batch_x, batch_t), 1)
# %%
