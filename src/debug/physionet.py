# %%
from src.physionet_dataset import PhysioNetDataset, CollateFn
import os
from src.util.general import parse_config
import matplotlib.pyplot as plt
import numpy as np
# %%


name, dataset_params, model_params, optimizer_params, trainer_params = parse_config('pretrain_physionet')

device_name = os.getenv('DEVICE', 'cpu')

train_data_params = dataset_params['train']
val_data_params = dataset_params['val']

train_dataset = PhysioNetDataset(**train_data_params)
val_dataset = PhysioNetDataset(**val_data_params)

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

# %%
df_lab, df_pid = train_dataset.to_pandas()


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
# Create a single figure and 6x6 grid of axes
fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(15, 15))  # Adjust figsize as needed

for i, feature in enumerate(features):
    # Distribution of the feature
    axes[int(i/6)][i % 6].hist(plot_obj[feature]['nan_per_patient'], bins=20)
    axes[int(i/6)][i % 6].set_title('NaNs per patient for ' + feature)

# Adjust space between plots
plt.subplots_adjust(hspace=0.5, wspace=0.5) 

# Save the figure
fig.savefig(os.path.join(REPORTS_ROOT, 'nan_per_pat_plot.png'))

# %%
