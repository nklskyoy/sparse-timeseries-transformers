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

logger = TensorBoardLogger("tb_logs", name="my_model")

# %%
freq='1H'

# %%
if __name__ == "__main__":

    lab = PhysioNetDataset(
        root_path={
            'raw' : os.path.join('data','physionet.org','files', 'set-a'),
            'data' : os.path.join('data','physionet.org'),
        },
        dataset_name='set-a',
        freq=freq,
        write_to_disk=False,
        device=torch.device('cpu')
    )
                
    model = PreTESS(
        dataset=lab,
        ts_dim=36, time_embedding_dim=256, static_dim=5,
        ts_encoder={'shape': [72, 128, 128, 256], 'dropout': 0.1},
        time_embedding={'shape' : [1, int(256 ** 0.5), 256]},
        static_feature_encoder={'shape' : [5,256], 'dropout': 0.1, 'last_layer_activation': nn.Identity},
        mha={'num_layers' : 4, 'n_heads': 8, 'dropout': 0.1},
    )


    collate_fn = CollateFn(device=lab.device)
    dataloader = DataLoader(lab, batch_size=128, shuffle=True, num_workers=0, collate_fn=collate_fn)

    trainer = Trainer(accelerator="cpu", devices=1, max_epochs=300, log_every_n_steps=1, logger=logger)
    trainer.fit(model, dataloader)