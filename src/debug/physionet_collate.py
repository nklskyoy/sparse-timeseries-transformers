# %%
from src.physionet_dataset import PhysioNetDataset, collate_on_device
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import os


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

collate_fn = collate_on_device(device=lab.device)

dl = DataLoader(lab, collate_fn=collate_fn, batch_size=5)
t = next(iter(dl))
# %%
t
# %%
