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

logger = TensorBoardLogger("tb_logs", name="my_model")

# %%
# %%
if __name__ == "__main__":

    device_name = os.getenv('DEVICE', 'cpu')

    dataset_params, model_params, optimizer_params, trainer_params = parse_config('pretrain_physionet')

    batch_size = optimizer_params['batch_size']

    dataset_params['device'] = torch.device(device_name) 
    lab = PhysioNetDataset(**dataset_params)
    model_params['dataset'] = lab 
    model = PreTESS( **model_params)

    collate_fn = CollateFn(device=lab.device)
    dataloader = DataLoader(lab, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

    trainer = Trainer(
        accelerator=device_name, 
        devices=1, 
        max_epochs=300, 
        log_every_n_steps=1, 
        logger=logger, 
        default_root_dir=trainer_params['default_root_dir']
    )

    
    trainer.fit(model, dataloader)