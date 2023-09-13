# %%
from src.physionet_dataset import PhysioNetDataset, CollateFn
from src.model.tess_pretraining import PreTESS
from torch.utils.data import DataLoader
import os
#import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
from torch import nn
from pytorch_lightning.loggers import TensorBoardLogger
from src.util.general import parse_config
from pytorch_lightning.callbacks import ModelCheckpoint
from src.model.tess_pretraining import PreTESS
# %%
_, dataset_params, model_params, _, _ = parse_config('finetune_physionet')
device_name = os.getenv('DEVICE', 'cpu')
train_data_params = dataset_params['train']
train_dataset = PhysioNetDataset(**train_data_params)
# %%
x, pid, T, _ = train_dataset[0]
# %%

supervised_head_params = model_params['supervised_predict_head']
model_params['dataset'] = train_dataset 
del model_params['supervised_predict_head']

model = PreTESS( **model_params)
# %%
sst_model_path = 'checkpoints/pretrain_physionet_m_adamw_nosche-epoch=299-val_loss=1.59.ckpt'
state_dict = torch.load(sst_model_path)
state_dict = state_dict['state_dict']
model.load_state_dict(state_dict)

# %%



# %%

PreTESS.load_from_checkpoint(
    sst_model_path, 
    dataset=train_dataset, ts_dim=36, time_embedding_dim=256, static_dim=5
)
# %%
