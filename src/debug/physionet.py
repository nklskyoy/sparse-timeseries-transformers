# %%
from src.physionet_dataset import PhysioNetDataset
import matplotlib.pyplot as plt
import numpy as np
import os

# %%
freq='1H'
REPORTS_ROOT = os.path.join('reports', 'physionet'+freq)

if not os.path.exists(REPORTS_ROOT):
    # If it doesn't exist, create it
    os.makedirs(REPORTS_ROOT)
else:
    # If it already exists, do nothing or handle as per your requirement
    print(f"Directory '{REPORTS_ROOT}' already exists.")

# %%
lab = PhysioNetDataset(
    root_path={
        'raw' : os.path.join('data','physionet.org','files', 'set-a'),
        'data' : os.path.join('data','physionet.org'),
    },
    dataset_name='set-a',
    freq=freq,
    write_to_disc=False
)

# %%
df_lab, df_pid = lab.to_pandas()

lab, mask, pid = lab[0]

# expolre the data

# %%
def get_nan_count_per_patient(df,feature):
    return df[feature].isna().reset_index().groupby('index').sum()


def get_length_nan_window_per_patient(df,feature):
    vals = df[feature]

# %%
features = df_lab.columns

plot_obj = {
    'nan_per_patient' : {}
}

for feature in features:
    plot_obj[feature] = {}
    plot_obj[feature]['nan_per_patient'] = get_nan_count_per_patient(df_lab,feature)
    vals = df_lab[feature].values
    vals = vals[~np.isnan(vals)]
    plot_obj[feature]['hist'] = vals

# %%
for feature in features:
    fig, ax = plt.subplots(2)
    ax[0].hist(plot_obj[feature]['hist'],bins=20)
    ax[0].set_title('Distribution of '+feature)
    ax[1].hist(plot_obj[feature]['nan_per_patient'], bins=20)
    ax[1].set_title('Number of NaNs per patient for '+feature)
    plt.subplots_adjust(hspace=0.5)  # You can change the value of hspace to control the vertical spacing
    fig.savefig(os.path.join(REPORTS_ROOT,feature+'.png'))


# %%
