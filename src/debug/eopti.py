# %%
import numpy as np
import torch
import os


# %%
path_to_data = 'data/eopti/interim'
disease = 'delir'
label = 'train'

data_root = os.path.join(path_to_data, disease, label)
# %%
# get list of all npx files
npz_files = [f for f in os.listdir(data_root) if f.endswith('.npz')]

# 
# %%
