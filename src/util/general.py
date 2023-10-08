from torch import nn
import json
import os
import torch
from src.util.lr_scheduler import CyclicLRWithRestarts
from src.dataset.physionet import PhysioNetDataset
from src.dataset.eopti import EoptiDataset





def update_activation(d):
    for key, value in d.items():
        if isinstance(value, dict):  # if the value is another dictionary, recursively call the function
            update_activation(value)
        elif key.endswith("activation"):  # if the key ends with "activation", update its value
            d[key] = parse_activation(value)

def prepare_model_config(model_config):
    update_activation(model_config)
    return model_config



def parse_dataset(dataset_name):
    if dataset_name == "PhysioNet":
        return PhysioNetDataset
    elif dataset_name == "Eopti":
        return EoptiDataset
    else:
        pass



def parse_optimizer(optimizer_config):
    optimizer_name = optimizer_config['name']
    if optimizer_name == "AdamW":
        optimizer_params = {
            'lr': optimizer_config['lr'],
            'weight_decay': optimizer_config['weight_decay']
        }
        return torch.optim.AdamW, optimizer_params
    elif optimizer_name == "Adam":
          # throw not implemented error
        raise NotImplementedError
    elif optimizer_name == "SGD":
        raise NotImplementedError


def parse_activation(activation):
    if activation == "nn.ReLU":
        return nn.ReLU
    elif activation == "nn.Tanh":
        return nn.Tanh
    elif activation == "nn.Identity":
        return None

def parse_dense_config(dense_config):
    if 'last_layer_activation' in dense_config:
        dense_config['last_layer_activation'] = parse_activation(dense_config['last_layer_activation'])
    if 'activation' in dense_config:
        dense_config['activation'] = parse_activation(dense_config['activation'])

    return dense_config



def get_opt_lr(optimizers_config):
    batch_size = optimizers_config['batch_size']
    epoch_size = optimizers_config['epoch_size'] if 'epoch_size' in optimizers_config else None
    optimizer_config = optimizers_config['optimizer']
    scheduler_config = optimizers_config['lr_schedule']

    opt_type, optimizer_params = parse_optimizer(optimizer_config)

    if scheduler_config['name'] == 'CyclicLRWithRestarts':

        def cofig_optimizer_fn(params):
            optimizer = opt_type(params, **optimizer_params)

            scheduler = {
                'scheduler': CyclicLRWithRestarts(optimizer, batch_size, epoch_size),
                'interval': 'step',
                'frequency': 1,
            }

            return [optimizer], [scheduler]

    return  cofig_optimizer_fn



def parse_config(filename):
    # Parse the JSON string into a Python dictionary
    conf_path = os.path.join('config', "{}.json".format(filename))
    with open(conf_path, 'r') as f:
        params = json.load(f)

    model_params = params['model']
    model_params['static_feature_encoder']['last_layer_activation'] = parse_activation(model_params['static_feature_encoder']['last_layer_activation'])

    dataset_params = params['dataset']
    cofig_optimizer_fn =  get_opt_lr(params['optimizers']) 
    trainer_params = params['trainer']
    name = params['name']
    
    batch_size = params['optimizers']['batch_size']

    return name, dataset_params, batch_size, model_params, cofig_optimizer_fn, trainer_params