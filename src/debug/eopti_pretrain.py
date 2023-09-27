# %%
import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

# %%
path_to_data = '/home/alexander/kidfail/data/EOPTI_ALLE'

data_root = os.path.join(path_to_data)
lab = os.path.join(data_root, "LAB_FILTERED.csv")

lab = pd.read_csv(lab, sep=';')


lab.loc[:,'STAMP'] = pd.to_datetime(lab.STAMP)
lab = lab.rename(columns={
    'KH-internes-Kennzeichen': 'ID',
    'Analyse': 'Parameter',
    'Ergebnis': 'Value',
    'STAMP': 'Time'
})

# %%

ts_len = lab.groupby('ID').size()
ts_len = pd.DataFrame(ts_len)
ts_len = ts_len.rename(columns={0:'cnt'})
ts_len = ts_len[ts_len.cnt > 15]

# %%
ts_len.hist(bins=400, edgecolor='black', alpha=0.3, label=f"distribution")

plt.title('Distribution of Counts per ID')
plt.xlim(10, 250)
plt.xlabel('Counts')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()



# %%
lab = lab[lab['ID'].isin(ts_len.index)]





# %%
