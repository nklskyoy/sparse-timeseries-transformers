**Transformer Encoder for Sparse time series**
-

Referring to https://openreview.net/forum?id=HUCgU5EQluN



*Requirements*
- pytorch
- pytorch lightning


**Getting started**


Currently supported datasets:
- Physionet2012 (Has to first downloaded by hand)




The configs are maintained in `config` directory.


For training, run  

    DEVICE=cuda python -m src.script.<training_step> <config_name>

where `<training_step>` is either `pretrain` or `finetune`.


Examle:

    DEVICE=cuda python -m src.script.pretrain pretrain_physionet_lin_warmup1H