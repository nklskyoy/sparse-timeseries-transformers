# Create chunks of the dataframe based on 'ID'
import pandas as pd
import os
import numpy as np
from multiprocessing.pool import ThreadPool as Pool
n_jobs = 6
import json


def get_id_map(df,n):
    ids = df['ID'].unique()
    id_map = {}
    for i in range(0, len(ids), n):
        chunk_num = i // n
        chunk_ids = ids[i:i+n]
        id_map[str(chunk_num)] = chunk_ids.tolist()
    return id_map


def split_dataframe(df, n):
    ids = df['ID'].unique()
    id_map = {}
    for i in range(0, len(ids), n):
        chunk_num = i // n
        chunk_ids = ids[i:i+n]
        yield df[df['ID'].isin(chunk_ids)], chunk_num
        id_map[str(chunk_num)] = chunk_ids.tolist()
    return id_map

# Processing function to be applied to each chunk
def chunk_fn(chunk_root,freq='1H'):
    def process_chunk(args):
        chunk, chunk_num = args

        chunk = pd.DataFrame(chunk.groupby([
            pd.Grouper(key='Time', freq=freq),
            pd.Grouper(key='Parameter'),
            pd.Grouper('ID')    
        ])['Value'].mean())\
            .reset_index()\
            .pivot(index=['ID', 'Time'], columns='Parameter', values='Value')

        chunk =  chunk \
            .reset_index() \
            .set_index('Time') \
            .groupby('ID') \
            .resample(freq) \
            .aggregate(np.mean) \
            .reset_index()

        chunk['Time'] = chunk.groupby('ID')['Time'].transform(lambda x: x - x.min())
        chunk_path = os.path.join(chunk_root, f"chunk_{chunk_num}.npz")

        np.savez(chunk_path, chunk.to_numpy())
        return chunk

    return process_chunk

if __name__ == "__main__":
    # Number of IDs in each chunk
    N = 20000  # or whatever number you prefer

    raw_data_root = '/home/alexander/kidfail/data/EOPTI_ALLE'
    data_root = os.path.join('data/eopti/lab')

    lab = os.path.join(raw_data_root, 'LAB_FILTERED.csv')
    pid = os.path.join(raw_data_root, 'FALL.csv')
    lab = pd.read_csv(lab, sep=';')
    pid = pd.read_csv(pid, sep=';')

    lab.loc[:,'STAMP'] = pd.to_datetime(lab.STAMP)
    lab = lab.rename(columns={
        'KH-internes-Kennzeichen': 'ID',
        'Analyse': 'Parameter',
        'Ergebnis': 'Value',
        'STAMP': 'Time'
    })
    lab = lab[['ID', 'Parameter', 'Value', 'Time']]

    # Split the dataframe into chunks
    chunks = list(split_dataframe(lab, N))
    id_map = get_id_map(lab, N)

    # save id_map
    with open(os.path.join(data_root, 'id_map.json'), 'w') as fp:
        json.dump(id_map, fp)

    process_chunk  = chunk_fn(data_root)

    # Use all available CPU cores
    with Pool(6) as pool:
        results = pool.map(process_chunk, chunks)

    # Combine results
    processed_lab = pd.concat(results)