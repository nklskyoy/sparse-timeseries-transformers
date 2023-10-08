# %%
from src.model.pl_pretrain import PLPretrain
from torch.utils.data import DataLoader
import os
from pytorch_lightning import Trainer
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from src.util.general import parse_config
from pytorch_lightning.callbacks import ModelCheckpoint
from src.dataset.physionet import CollateFn

# %%
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')# good solution !!!!

    model = PLPretrain('pretrain_physnet_cosine')

    logger = TensorBoardLogger("tb_logs", name=model.name)
    
    device = torch.device(model.device_name)

    collate_fn = CollateFn(device=device )

    loader_train = DataLoader(
        model.train_dataset, 
        batch_size=model.batch_size, 
        shuffle=True, num_workers=1, collate_fn=collate_fn
    )

    loader_val = DataLoader(
        model.val_dataset, 
        batch_size=model.batch_size, 
        shuffle=True, num_workers=1, collate_fn=collate_fn
    )
    

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename=model.name+'-{epoch:02d}-{val_loss:.2f}',
        dirpath='checkpoints/physnet_pretrain',
        save_top_k=10
    )

    trainer = Trainer(
        deterministic=True,
        accelerator='cuda', 
        devices=1, 
        max_epochs=350, 
        log_every_n_steps=1, 
        logger=logger, 
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        overfit_batches=10
    )
    
    trainer.fit(model, loader_train, loader_val)