# %%
import pandas as pd
import numpy as np
import torch
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
disease = 'delir'
h = hash_dict[disease]


data_root = os.path.join(path_to_data, h)
lab_train = os.path.join(data_root, "{d}_lab_train.csv".format(d="delir"))
lab_train = pd.read_csv(lab_train, sep=',')
y_train = os.path.join(data_root, "{d}_classified_ids_train.csv".format(d="delir"))
y_train = pd.read_csv(y_train, sep=',')

classes = y_train.columns.to_list()
classes.remove('id')
classes.remove('y_date')


data = pd.merge(lab_train, y_train.drop(['y_date'],axis=1), on='id')
data.loc[:,'healthy'] = True

for c in classes:
    data.loc[data[c], 'healthy'] = False

classes_with_healthy = classes + ['healthy']
data.loc[:,'x_date'] = pd.to_datetime(data.x_date)
grouped = data.groupby('id').agg(timeseries_length=('x_date', lambda x: x.max() - x.min()))
grouped['hours'] = grouped['timeseries_length'].dt.total_seconds() / (60 * 60)


# %%
data_filtered = data
class_ts_len = {}
for c in classes_with_healthy:
    class_ts_len[c] = data_filtered[data_filtered[c]].groupby('id').size()


binary_class_ts_len = {}
binary_class_ts_len['healthy'] = data_filtered[data_filtered['healthy']].groupby('id').size()
binary_class_ts_len['F05'] = data_filtered[~data_filtered['healthy']].groupby('id').size()

binary_classes = ['healthy', 'F05']

# %%





# %%
# Plotting
plt.figure(figsize=(10, 6))
sns.histplot(grouped['hours'], bins=1000, kde=True)
plt.xlim([100, 1000])

plt.xlabel('Timeseries Length (hours)')
plt.ylabel('Count')
plt.title('Distribution of Timeseries Length')
plt.show()



# %%
for c in classes:
    #class_ts_len[c].plot.kde(label=f"Class {c}", lw=1.5)
    class_ts_len[c].hist(bins=30, edgecolor='black', alpha=0.3, label=f"Class {c}")

plt.title('Distribution of Counts per ID')
plt.xlim(10, 250)
plt.xlabel('Counts')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# %%
for c in binary_classes:
    #class_ts_len[c].plot.kde(label=f"Class {c}", lw=1.5)
    binary_class_ts_len[c].hist(bins=100, edgecolor='black', alpha=0.3, label=f"Class {c}")

plt.title('Distribution of Counts per ID')
plt.xlim(10, 250)
plt.xlabel('Counts')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()





# %%
lab = os.path.join(path_to_data, 'LAB_FILTERED.csv')
pid = os.path.join(path_to_data, 'FALL.csv')
lab = pd.read_csv(lab, sep=';')
pid = pd.read_csv(pid, sep=';')

freq = '1H'

# %%
lab.loc[:,'STAMP'] = pd.to_datetime(lab.STAMP)
lab = lab.rename(columns={
    'KH-internes-Kennzeichen': 'ID',
    'Analyse': 'Parameter',
    'Ergebnis': 'Value',
    'STAMP': 'Time'
})
lab = lab[['ID', 'Parameter', 'Value', 'Time']]


# %%
lab = pd.DataFrame(lab.groupby([
    pd.Grouper(key='Time', freq=freq),
    pd.Grouper(key='Parameter'),
    pd.Grouper('ID')    
])['Value'].mean())\
    .reset_index()\
    .pivot(index=['ID', 'Time'], columns='Parameter', values='Value')

lab =  lab.reset_index()
# %%
min_len = 4

ts_len = lab.groupby('ID').size()
ts_len = pd.DataFrame(ts_len)
ts_len = ts_len.rename(columns={0:'cnt'})
id_min_len = ts_len[ts_len.cnt > 4]

# %%



lab['Time'] = lab.groupby('ID')['Time'].transform(lambda x: x - x.min())
# %%
# check for multiple units

result = lab.groupby('Analyse')['Einheit'].unique()
multiple_units = result[result.apply(len) > 1]

print(multiple_units)
# %%


