# %%
from src.physionet_dataset import PhysioNetDataset
from src.model.tess import Tess
from torch.utils.data import DataLoader
import os


# %%
freq='1H'

# %%

model = Tess(
    ts_dim=36, time_embedding_dim=128,
    ts_encoder_hidden_size=128, ts_encoder_num_layers=2, ts_encoder_dropout=0.1,
    n_heads=8
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
model(batch_x, batch_t)
# %%
