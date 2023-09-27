from torch.utils.data import Dataset, DataLoader
from torch import from_numpy
import numpy as np
import pandas as pd
import glob 
import os
import torch 
import pickle



class CollateFn:
    def __init__(self, device=torch.device('cpu'), supervised=False) -> None:
        self.device = device
        self.supervised = supervised

    def __call__(self, batch):
        device = self.device
        # Get the max length in this batch
        max_length = max([x[0].shape[0] for x in batch])
        D = batch[0][0].shape[2]
        
        n_static_features = batch[0][1].shape[1]

        # Create a tensor filled with NaNs
        lab = torch.full(
            (len(batch), max_length,2, D), 
            0., device=device)
        pid = torch.full(
            (len(batch), n_static_features), 0., 
            device=device)

        if self.supervised == True:
            target = torch.full(
                (len(batch), 1), 0., 
                device=device)

        T = torch.arange(max_length, device=device).unsqueeze(-1).float()
        T = T.repeat(len(batch), 1,1)

        # Copy over the actual sequences
        for i, sequence in enumerate(batch):
            cur_lab = sequence[0]
            cur_pid = sequence[1]

            lab[i, :cur_lab.shape[0],: , :] = cur_lab
            pid[i, :] = cur_pid

            if self.supervised == True:
                cur_target = sequence[3]
                target[i, :] = cur_target
        if not self.supervised:
            return lab, pid, T
        else:
            return lab, pid, T, target



class EoptiDataset(Dataset):
    def __init__(self, root_path, dataset_name, freq='10H', write_to_disk=False, device=torch.device('cpu'), supervised=False) -> None:
        self.root_path = root_path
        self.device = device
        self.supervised = supervised
        self.name = dataset_name

        data_path = os.path.join(root_path['data'], "{name}_{freq}".format(name=dataset_name, freq=freq))
        if os.path.exists(data_path) and not write_to_disk:
            # if the data is already saved, load it
            print('use saved data')
            data = np.load(os.path.join(data_path, 'data.npz'), allow_pickle=True)
            
            self.lab_x=data['lab_x']
            self.lab_index=data['lab_index']
            self.lab_features=data['lab_features']
            self.pid_x=data['pid_x']
            self.pid_index=data['pid_index']
            self.pid_features=data['pid_features']

        # load targets for supervised training

        # for convenience   
        self.unique_id = np.unique(self.lab_index)
        self.ts_dim = self.lab_x.shape[1]



    def __getitem__(self, idx):
        lab = self.lab_x[self.lab_index == self.unique_id[idx]] 
        lab = from_numpy(lab.astype(np.float32))
        mask = ~lab.isnan()
        lab[~mask] = 0

        pid = self.pid_x[self.pid_index == self.unique_id[idx]]
        pid = from_numpy(pid.astype(np.float32))

        # T x 1
        T = lab.shape[0]
        # T x 2 x D
        masked_lab = torch.stack((lab, mask), dim=2).transpose(-2,-1)

        if self.supervised:
            target = self.target[self.target[:, 0] == int(self.unique_id[idx]), 1]
            target = from_numpy(target.astype(np.float32))

            return  \
                masked_lab.to(self.device),\
                pid.to(self.device),\
                torch.arange(T, device=self.device).unsqueeze(-1).float(),\
                target.to(self.device)

        else:
            return  \
                masked_lab.to(self.device),\
                pid.to(self.device),\
                torch.arange(T, device=self.device).unsqueeze(-1).float()

        
    def __len__(self):
        return self.unique_id.shape[0]


