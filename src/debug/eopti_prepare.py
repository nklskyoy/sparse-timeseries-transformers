# %%
import pandas as pd
import numpy as np
import torch
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# %%
path_to_data = '/home/alexander/kidfail/data/EOPTI_ALLE/processed'

hash_dict = {
    'delir':'74377a0b',#
    'mangel':'a11f2fd7',
    'sepsis':'30f64355',
}

# %%



# %%
def prepare_and_wite_lab(disease, label, freq, base_path):
    h = hash_dict[disease]
    data_root = os.path.join(path_to_data, h)
    lab = os.path.join(data_root, "{d}_lab_{l}.csv".format(d=disease, l=label))
    lab = pd.read_csv(lab, sep=',')

    lab = lab.drop(['y_date'],axis=1)
    lab = lab.rename(columns={'x_date':'Time'})
                     
    lab.loc[:,'Time'] = pd.to_datetime(lab['Time'])

    grouped = lab.groupby('id').agg(timeseries_length=('Time', lambda x: x.max() - x.min()))
    grouped['hours'] = grouped['timeseries_length'].dt.total_seconds() / (60 * 60)

    lab =  lab[lab['id'].isin(grouped[grouped.hours > 20].index)]
    print(lab.id.unique().shape)
    lab['Time'] = lab.groupby('id')['Time'].transform(lambda x: x - x.min())

    y = os.path.join(data_root, "{d}_classified_ids_{l}.csv".format(d=disease, l=label))
    y = pd.read_csv(y, sep=',')   

    lab =  lab \
        .reset_index() \
        .set_index('Time') \
        .groupby('id') \
        .resample(freq) \
        .aggregate(np.mean) \
        .reset_index()

    lab_mean_std_values = {}  # Dictionary to save mean and std for each column
    pid_mean_std_values = {}

    columns_to_normalize = [col for col in lab.columns if col not in ['id', 'Time']]

    for col in columns_to_normalize:
        # Compute mean and std ignoring NaNs
        mean_value = lab[col].mean(skipna=True)
        std_value = lab[col].std(skipna=True)

        # Store mean and std values
        lab_mean_std_values[col] = {'mean': mean_value, 'std': std_value}

        # Normalize the column
        lab[col] = (lab[col] - mean_value) / std_value


    columns_to_normalize = [col for col in pid.columns if col not in ['id']]


    for col in columns_to_normalize:
        # Compute mean and std ignoring NaNs
        mean_value = lab[col].mean(skipna=True)
        std_value = lab[col].std(skipna=True)

        # Store mean and std values
        pid_mean_std_values[col] = {'mean': mean_value, 'std': std_value}

        # Normalize the column
        lab[col] = (lab[col] - mean_value) / std_value


    lab = lab.sort_values(['id', 'Time'])

    disease_dir = os.path.join(base_path, disease)
    if not os.path.exists(disease_dir):
        os.makedirs(disease_dir)

    labeled_dir = os.path.join(disease_dir, label)
    if not os.path.exists(labeled_dir):
        os.makedirs(labeled_dir)

    with open(os.path.join(labeled_dir, 'lab_mean_std_values.pkl'), 'wb') as file:
        pickle.dump(lab_mean_std_values, file)  

    with open(os.path.join(labeled_dir, 'pid_mean_std_values.pkl'), 'wb') as file:
        pickle.dump(pid_mean_std_values, file)  

    pid = os.path.join(data_root, "{d}_fall_{l}.csv".format(d=disease, l=label))
    pid = pd.read_csv(pid, sep=',')
    pid = pid[pid.id.isin(lab.id.unique())]
    pid.loc[:,'sex'] = -1
    pid[pid.Geschlecht == 'w']['sex'] = 0
    pid[pid.Geschlecht == 'W']['sex'] = 0
    pid[pid.Geschlecht == 'm']['sex'] = 1
    pid[pid.Geschlecht == 'M']['sex'] = 1

    pid = pid[['id', 'Alter-in-Jahren-am-Aufnahmetag', 'sex']]


    # save data as npz
    np.savez(
        os.path.join(labeled_dir, 'data.npz'), 
        lab_index=lab[['id']].to_numpy().reshape(-1),
        lab_x = lab.drop(['id', 'Time'], axis=1).to_numpy(),
        lab_features=lab.columns.values[2:],
        pid_x=pid.drop(['id'], axis=1).to_numpy(),
        pid_index= pid[['id']].to_numpy().reshape(-1),
        pid_features=pid.columns.values[1:]
    )
    
# %%
labels = ['train', 'test', 'dev']
freq = '5H'


for disease in hash_dict.keys():
    for label in labels:
        prepare_and_wite_lab(disease, label, freq, 'data/eopti')




















# %%


# %%



