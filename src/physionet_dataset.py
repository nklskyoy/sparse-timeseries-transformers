from torch.utils.data import Dataset, DataLoader
from torch import from_numpy
import numpy as np
import pandas as pd
import glob 
import os

class PhysioNetDataset(Dataset):
    def __init__(self, root_path, dataset_name, freq='10H', write_to_disc=False) -> None:
        self.root_path = root_path
        
        data_path = os.path.join(root_path['data'], "{name}_{freq}".format(name=dataset_name, freq=freq))
        if os.path.exists(data_path) and not write_to_disc:
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

            # save attributes
            self.lab_x = lab.drop(['ID', 'Time'], axis=1).to_numpy()
            self.lab_index = lab[['ID']].to_numpy().reshape(-1)
            self.lab_features = lab.columns.values[2:]
            self.pid_x = pid.drop(['ID'], axis=1).to_numpy()
            self.pid_index = pid[['ID']].to_numpy().reshape(-1)
            self.pid_features = pid.columns.values[1:]

            os.makedirs(data_path)
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
        
        return lab, mask, pid

        
    def __len__(self):
        return len(self.lab.index.get_level_values(0).unique())

