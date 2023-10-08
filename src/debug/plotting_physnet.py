# %%
from src.model.pl_pretrain import PLPretrain
import os
import matplotlib.pyplot as plt
freq = '1H'
REPORTS_ROOT = os.path.join('reports', 'physionet'+freq)

# %%
config = os.getenv('NAME', "pretrain_physnet_lin_warmup1H")
model = PLPretrain(config)


# %%
dataset_train = model.train_dataset
dataset_train = dataset_train.to_pandas()[0]
mask_train = dataset_train[[col for col in dataset_train.columns if col.endswith('_mask')]]
rename_dict = {col: col.replace('_mask', '') for col in mask_train}
mask_train.rename(columns=rename_dict, inplace=True)

dataset_val = model.val_dataset
dataset_val = dataset_val.to_pandas()[0]
mask_val = dataset_val[[col for col in dataset_val.columns if col.endswith('_mask')]]
rename_dict = {col: col.replace('_mask', '') for col in mask_val}
mask_val.rename(columns=rename_dict, inplace=True)

# %%
summed_mask_val = mask_val.groupby(mask_val.index).sum()
summed_mask_train = mask_train.groupby(mask_train.index).sum()

rel_cnt_mask_val = summed_mask_val.div(mask_val.groupby(mask_val.index).size(), axis=0)
rel_cnt_mask_train = summed_mask_train.div(mask_train.groupby(mask_train.index).size(), axis=0)
# %%
features = summed_mask_val.columns

# %%

fig, axes = plt.subplots(6, 6, figsize=(18, 18))

# Iterate through the features and their respective axes
for feature, ax in zip(features, axes.ravel()):
    # Plot train histogram with only border color
    ax.hist(rel_cnt_mask_train[feature], bins=30, histtype='step', edgecolor='blue', label='Train')
    ax.hist(rel_cnt_mask_val[feature], bins=30, histtype='step', edgecolor='red', label='Test')

    # Plot test histogram with only border color    
    ax.set_title(feature)
    ax.legend()
    plt.subplots_adjust(hspace=0.5)  # You can change the value of hspace to control the vertical spacing
    fig.savefig(os.path.join(REPORTS_ROOT,'missingness_test_dev'))


# %%
