from torch import nn
import json
import os

def parse_activation(activation):
    if activation == "nn.ReLU":
        return nn.ReLU()
    elif activation == "nn.Tanh":
        return nn.Tanh()
    elif activation == "nn.Identity":
        return nn.Identity()

def parse_config(filename):
    # Parse the JSON string into a Python dictionary
    conf_path = os.path.join('config', "{}.json".format(filename))
    with open(conf_path, 'r') as f:
        params = json.load(f)

    model_params = params['model']
    model_params['static_feature_encoder']['last_layer_activation'] = parse_activation(model_params['static_feature_encoder']['last_layer_activation'])

    dataset_params = params['dataset']
    optimizer_params = params['optimizer']
    trainer_params = params['trainer']

    return dataset_params, model_params, optimizer_params, trainer_params