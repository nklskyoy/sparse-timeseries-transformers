from torch.utils.data import Dataset, DataLoader
from torch import from_numpy
import numpy as np
import pandas as pd
import glob 
import os
import torch 
import pickle



class CollateFn:
    def __init__(self, device=torch.device('cpu')) -> None:
        self.device = device

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

        T = torch.arange(max_length, device=device).unsqueeze(-1).float()
        T = T.repeat(len(batch), 1,1)

        # Copy over the actual sequences
        for i, sequence in enumerate(batch):
            cur_lab = sequence[0]
            cur_pid = sequence[1]

            lab[i, :cur_lab.shape[0],: , :] = cur_lab
            pid[i, :] = cur_pid

        return lab, pid, T



class PhysioNetDataset(Dataset):
    def __init__(self, root_path, dataset_name, freq='10H', write_to_disk=False, device=torch.device('cpu')) -> None:
        self.root_path = root_path
        self.device = device

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
        else:
            txt_files = glob.glob(os.path.join(root_path['raw'], '*.txt'))
            df_list = []
            # loop through each file and parse it into a pandas dataframe
            for file in txt_files:
                # read the txt file into a pandas dataframe
                df = pd.read_csv(file, delimiter=',')
                df.loc[:, 'ID'] = os.path.basename(file).split('.')[0]
                df.loc[:, 'Age'] = df[df.Parameter == 'Age']['Value'].values[0]
                df.loc[:, 'Gender'] = df[df.Parameter == 'Gender']['Value'].values[0]
                df.loc[:, 'Height'] = df[df.Parameter == 'Height']['Value'].values[0]
                df.loc[:, 'ICUType'] = df[df.Parameter == 'ICUType']['Value'].values[0]
                df.loc[:, 'Weight'] = df[df.Parameter == 'Weight']['Value'].values[0]

                # add a new column with the filename as id
                # append the dataframe to the list
                df_list.append(df)

            df = pd.concat(df_list)

            df.loc[:,'Time'] = df.Time + ':00'
            df.loc[:,'Time'] = pd.to_timedelta(df.Time)

            self.name = dataset_name
            pid = df[df.Parameter.isin(['Age', 'Gender', 'Height', 'ICUType', 'Weight', 'RecordID'])].drop_duplicates()
            pid = pid.drop(['Time','Parameter','Value'], axis=1).drop_duplicates()
            
            lab = df[~df.Parameter.isin(['RecordID','Age', 'Gender', 'Height', 'ICUType', 'Weight'])].drop_duplicates()
            
            # long to wide
            lab = pd.DataFrame(lab.groupby([
                pd.Grouper(key='Time', freq=freq),
                pd.Grouper(key='Parameter'),
                pd.Grouper('ID')    
            ])['Value'].mean())\
                .reset_index()\
                .pivot(index=['ID', 'Time'], columns='Parameter', values='Value')
            
            # resamples
            lab =  lab \
                .reset_index() \
                .set_index('Time') \
                .groupby('ID') \
                .resample(freq) \
                .aggregate(np.mean) \
                .reset_index()
            
            # adjust Time column such that every timeseries starts with 0:00:00
            lab['Time'] = lab.groupby('ID')['Time'].transform(lambda x: x - x.min())

            columns_to_normalize = [col for col in lab.columns if col not in ['ID', 'Time']]

            mean_std_values = {}  # Dictionary to save mean and std for each column

            for col in columns_to_normalize:
                # Compute mean and std ignoring NaNs
                mean_value = lab[col].mean(skipna=True)
                std_value = lab[col].std(skipna=True)

                # Store mean and std values
                mean_std_values[col] = {'mean': mean_value, 'std': std_value}

                # Normalize the column
                lab[col] = (lab[col] - mean_value) / std_value


            lab = lab.sort_values(['ID', 'Time'])

            # try impute nan values via ffill
            for col in columns_to_normalize:
                # Compute mean and std ignoring NaNs
                lab[col] = lab\
                    .groupby('ID')[col]\
                    .fillna(method='ffill')

            # save attributes
            self.lab_x = lab.drop(['ID', 'Time'], axis=1).to_numpy()
            self.lab_index = lab[['ID']].to_numpy().reshape(-1)
            self.lab_features = lab.columns.values[2:]
            self.pid_x = pid.drop(['ID'], axis=1).to_numpy()
            self.pid_index = pid[['ID']].to_numpy().reshape(-1)
            self.pid_features = pid.columns.values[1:]

            os.makedirs(data_path)

            # save mean and std values
            with open(os.path.join(data_path, 'mean_std_values.pkl'), 'wb') as file:
                pickle.dump(mean_std_values, file)  

            # save data as npz
            np.savez(
                os.path.join(data_path, 'data.npz'), 
                lab_x=self.lab_x,
                lab_index=self.lab_index,
                lab_features=self.lab_features,
                pid_x=self.pid_x,
                pid_index=self.pid_index,
                pid_features=self.pid_features,
            )
        
        # for convenience   
        self.unique_id = np.unique(self.lab_index)
        self.ts_dim = self.lab_x.shape[1]


    def to_pandas(self):
        df_lab = pd.DataFrame(self.lab_x, index=self.lab_index,columns=self.lab_features)
        df_pid = pd.DataFrame(self.pid_x, index=self.pid_index,columns=self.pid_features)
        return df_lab, df_pid


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

        return  \
            masked_lab.to(self.device),\
            pid.to(self.device),\
            torch.arange(T, device=self.device).unsqueeze(-1).float()

        
    def __len__(self):
        return self.unique_id.shape[0]


