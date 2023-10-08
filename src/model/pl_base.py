from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch 
from torch import nn
from torch.nn import BCELoss, BCEWithLogitsLoss
from src.model.encoder import Encoder
from pytorch_lightning.utilities import grad_norm
from src.util.general import parse_dataset, parse_optimizer, prepare_model_config
import json
import os
from src.util.lr_scheduler import CyclicLRWithRestarts, LinearWarmup
from collections import OrderedDict



class PLBase(pl.LightningModule):
    def __init__(self,config_name):

        super(PLBase, self).__init__()

        self.config_name = config_name
        self.conf_path = os.path.join('config', "{}.json".format(config_name))

        self._model_params = None
        self.dataset_train = None
        self.dataset_val = None

        # first try read from config json
        with open(self.conf_path, 'r') as f:
            params = json.load(f)

        model_params = prepare_model_config(params['model'])
        
        self.load_pretrained = False
        if 'load_pretrained' in model_params:
            self.load_pretrained = True
            modules = {} #model_params['load_pretrained']['modules']

            self.pretrained_path = model_params['load_pretrained']['checkpoint']
            self.pretrained_encoder = True if 'encoder' in modules else False
            self.pretrained_ssl_head = True if 'ssl_head' in modules else False
            self.pretrained_classifier = True if 'classifier' in modules else False

            del model_params['load_pretrained']
        else:
            self.pretrained_path = None


        # save params
        self.name = params['name']
        self._model_params = model_params
        self._dataset_params = params['dataset']

        optimizer_params = params['optimizers']

        if 'alpha' in params['optimizers'] :
            self.alpha = params['optimizers']['alpha']
            del optimizer_params['alpha']

        self._optimizer_params = optimizer_params

        self.batch_size = self._optimizer_params['batch_size']

        # init dataset
        train_dataset_name =  params['dataset']['train']['name']
        val_dataset_name =  params['dataset']['val']['name']

        self.device_name = os.getenv('DEVICE', 'cuda') 

        device = torch.device(self.device_name)

        train_data_params = self._dataset_params['train']
        val_data_params = self._dataset_params['val']
        train_data_params['device'] = device
        val_data_params['device'] = device

        train_dataset = parse_dataset(train_dataset_name)
        del train_data_params['name']
        val_dataset = parse_dataset(val_dataset_name)
        del val_data_params['name']

        self.train_dataset = train_dataset(**train_data_params)
        self.val_dataset = val_dataset(**val_data_params)

        encoder_params = self._model_params['encoder']
        encoder_params['device'] = device

        self.encoder = Encoder(
            **encoder_params
        )

        if self.load_pretrained:
            self.load_from_checkpoint()


    def load_from_checkpoint(self):
        state_dict = torch.load(self.pretrained_path)
        state_dict = state_dict['state_dict']
        state_dict = OrderedDict(
            (key.replace('encoder.', ''), value)
            for key, value in state_dict.items()
            if key.startswith('encoder.')
        )
        self.encoder.load_state_dict(state_dict)
        

    def training_step(self, batch, batch_idx):
        # no training here
        return 0
    

    def validation_step(self, batch, batch_idx):
       # no validation here
       return 0


    def configure_optimizers(self):
        batch_size = self._optimizer_params['batch_size']
        epoch_size = self._optimizer_params['epoch_size'] \
            if 'epoch_size' in self._optimizer_params else None
        optimizer_config = self._optimizer_params['optimizer']
        scheduler_config = self._optimizer_params['lr_schedule']

        opt, optimizer_params = parse_optimizer(optimizer_config)

        optimizer = opt(self.parameters(), **optimizer_params)

        if scheduler_config['name'] == 'CyclicLRWithRestarts':
            

            scheduler = {
                'scheduler': CyclicLRWithRestarts(optimizer, batch_size, epoch_size),
                'interval': 'step',
                'frequency': 1,
            }

            return [optimizer], [scheduler]       
        elif scheduler_config['name'] == 'LinearWarmup':
            start_lr = scheduler_config['start_lr']
            ramp_up_epochs = scheduler_config['ramp_up_epochs']
            max_lr = scheduler_config['max_lr']
            print(start_lr, ramp_up_epochs, max_lr)
            scheduler = {
                'scheduler': LinearWarmup(optimizer, start_lr, max_lr, ramp_up_epochs),
                'interval': 'step',
                'frequency': 1,
            }

            return [optimizer], [scheduler]    
