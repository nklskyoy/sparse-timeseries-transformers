# %%
import pandas as pd
import numpy as np
import torch
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing.pool import ThreadPool as Pool
import json


n_jobs = 6
N=int(5e3)
freq = '5H'
path_to_data = '/home/alexander/kidfail/data/EOPTI_ALLE/processed'



hash_dict = {
    'delir':'74377a0b',#
    'mangel':'a11f2fd7',
    'sepsis':'30f64355',
}






def chunk_fn(chunk_root,freq='1H'):
    def process_chunk(args):
        chunk, chunk_num = args

        chunk =  chunk \
            .reset_index() \
            .set_index('Time') \
            .groupby('id') \
            .resample(freq) \
            .aggregate(np.mean) \
            .reset_index()

        chunk['Time'] = chunk.groupby('id')['Time'].transform(lambda x: x - x.min())
        chunk_path = os.path.join(chunk_root, f"chunk_{chunk_num}.npz")
        
        unique_ids = chunk['id'].unique()
        for uid in unique_ids:
            subset = chunk[chunk['id'] == uid].sort_values(by='Time')
            subset = subset.drop(['id', 'Time'],axis=1)
            chunk_path = os.path.join(chunk_root, f"{uid}.npz")
            np.savez(chunk_path, subset.to_numpy())

        return chunk

    return process_chunk



def get_id_map(df,n):
    ids = df['id'].unique()
    id_map = {}
    for i in range(0, len(ids), n):
        chunk_num = i // n
        chunk_ids = ids[i:i+n]
        id_map[str(chunk_num)] = chunk_ids.tolist()
    return id_map


def split_dataframe(df, n):
    ids = df['id'].unique()

    id_map = {}
    print(ids)
    for i in range(0, len(ids), n):
        chunk_num = i // n
        chunk_ids = ids[i:i+n]
        yield df[df['id'].isin(chunk_ids)], chunk_num
        id_map[str(chunk_num)] = chunk_ids.tolist()
    return id_map





def prepare_and_wite_interim_lab(disease, label, freq, base_path):
    h = hash_dict[disease]
    data_root = os.path.join(path_to_data, h)
    lab = os.path.join(data_root, "{d}_lab_{l}.csv".format(d=disease, l=label))
    lab = pd.read_csv(lab, sep=',')

    lab = lab.drop(['y_date'],axis=1)
    lab = lab.rename(columns={'x_date':'Time'})

    columns_to_normalize = [col for col in lab.columns if col not in ['id', 'Time']]
    print(columns_to_normalize)
    mean_std_values = {
        'lab' : {},
    }  # Dictionary to save mean and std for each column


    for col in columns_to_normalize:
        # Compute mean and std ignoring NaNs
        mean_value = lab[col].mean(skipna=True)
        std_value = lab[col].std(skipna=True)

        # Store mean and std values
        mean_std_values['lab'][col] \
            = {'mean': mean_value, 'std': std_value}

        # Normalize the column
        lab[col] = (lab[col] - mean_value) / std_value
                    
    lab.loc[:,'Time'] = pd.to_datetime(lab['Time'])

    grouped = lab.groupby('id').agg(timeseries_length=('Time', lambda x: x.max() - x.min()))
    grouped['hours'] = grouped['timeseries_length'].dt.total_seconds() / (60 * 60)

    lab =  lab[lab['id'].isin(grouped[grouped.hours > 20].index)]
    
    print(lab.id.unique().shape)

    chunks = list(split_dataframe(lab, N))
    id_map = get_id_map(lab, N)

    disease_dir = os.path.join(base_path, disease)
    if not os.path.exists(disease_dir):
        os.makedirs(disease_dir)

    labeled_dir = os.path.join(disease_dir, label)
    if not os.path.exists(labeled_dir):
        os.makedirs(labeled_dir)

    with open(os.path.join(labeled_dir, 'lab_mean_std_values.pkl'), 'wb') as file:
            pickle.dump(mean_std_values, file)  

    # save id_map
    with open(os.path.join(labeled_dir, 'id_map.json'), 'w') as fp:
        json.dump(id_map, fp)

    process_chunk  = chunk_fn(labeled_dir, freq=freq)

    # Use all available CPU cores
    with Pool(6) as pool:
        _ = pool.map(process_chunk, chunks)

    
# %%
labels = ['train', 'test', 'dev']



for disease in hash_dict.keys():
    for label in labels:
        prepare_and_wite_interim_lab(disease, label, freq, 'data/eopti/interim')




















# %%


# %%



