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
if __name__ == "__main__":

    device_name = os.getenv('DEVICE', 'cpu')

    dataset_params, model_params, optimizer_params, trainer_params = parse_config('pretrain_physionet')
    
    train_data_params = dataset_params['train']
    val_data_params = dataset_params['val']
    train_data_params['device'] = torch.device(device_name) 
    val_data_params['device'] = torch.device(device_name)
    train_dataset = PhysioNetDataset(**train_data_params)
    val_dataset = PhysioNetDataset(**val_data_params)
    
    collate_fn = CollateFn(device=torch.device(device_name) )

    batch_size = optimizer_params['batch_size']

    loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    loader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    
    model_params['dataset'] = train_dataset 
    model = PreTESS( **model_params)

    trainer = Trainer(
        accelerator=device_name, 
        devices=1, 
        max_epochs=300, 
        log_every_n_steps=1, 
        logger=logger, 
        default_root_dir=trainer_params['default_root_dir']
    )
    
    trainer.fit(model, loader_train, loader_val)