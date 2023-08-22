# %%
from src.physionet_dataset import PhysioNetDataset, CollateFn
from src.model.tess_pretraining import PreTESS
from torch.utils.data import DataLoader
import os
#import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
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
        write_to_disc=False,
        device=torch.device('mps')
    )


                
    model = PreTESS(
        dataset=lab,
        ts_dim=36, time_embedding_dim=2048,
        ts_encoder_hidden_size=2048, ts_encoder_num_layers=2, ts_encoder_dropout=0.1,
        n_heads=8,
        prob_mask=0.5
    )

    # %%
    collate_fn = CollateFn(device=lab.device)
    dataloader = DataLoader(lab, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate_fn)


    # %%
    trainer = Trainer(accelerator="mps", devices=1, max_epochs=300, log_every_n_steps=1, logger=logger)
    trainer.fit(model, dataloader)