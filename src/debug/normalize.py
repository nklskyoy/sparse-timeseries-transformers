# %%
from src.physionet_dataset import PhysioNetDataset
import matplotlib.pyplot as plt
import numpy as np
import os

# The data will be normalized on dataset creation
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
