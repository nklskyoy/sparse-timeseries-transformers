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

# %%
if __name__ == "__main__":

    name, dataset_params, model_params, optimizer_params, trainer_params = parse_config('pretrain_physionet')

    device_name = os.getenv('DEVICE', 'cpu')
    logger = TensorBoardLogger("tb_logs", name=name)
    
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

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename=name+'-{epoch:02d}-{val_loss:.2f}',
        dirpath='checkpoints',
        save_top_k=-1
    )

    trainer = Trainer(
        accelerator=device_name, 
        devices=1, 
        max_epochs=3, 
        log_every_n_steps=1, 
        logger=logger, 
        enable_checkpointing=True,
        callbacks=[checkpoint_callback]
    )
    
    trainer.fit(model, loader_train, loader_val)