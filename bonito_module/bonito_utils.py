import os
import torch
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import models
from importlib import import_module



def column_to_set(filename, idx=0, skip_header=False):
    """
    Pull a column from a file and return a set of the values.
    """
    if filename and os.path.isfile(filename):
        with open(filename, 'r') as tsv:
            if skip_header:
                next(tsv)
            return {line.strip().split()[idx] for line in tsv.readlines()}

def load_symbol(config, symbol):
    """
    Dynamic load a symbol from module specified in model config.
    """
    if config == 'bonito':
        symbol= 'BonitoModel'
        imported = import_module('models.bonitoModel')
    elif config == 'CSnano-c4-c16-c324-TaaRes-gate-l3':
        symbol= 'CSnetBonitoLSTM5Model'
        imported = import_module('models.CSmodel')
        # from models.CSmodel.model7 import CSnetBonitoLSTM5Model as Model      

    #print(config)
    
    return getattr(imported, symbol)

def load_model(dirname, device, weights=None, half=None, chunksize=None, batchsize=None, overlap=None, quantize=False, use_koi=False):
    """
    Load a model config and weights off disk from `dirname`.
    """
    
    
    config = weights
    weights = dirname
    return _load_model(weights, config, device, half, use_koi,chunksize,batchsize,quantize)

def _load_model(model_file, config, device, half=None, use_koi=False,chunksize=2000,batchsize=64,quantize=None):
    device = torch.device(device)
    Model = load_symbol(config, "Model")
    use_amp = False
    scaler = None

    model = Model(
        load_default = True,
        device = device,
        dataloader_train = None, 
        dataloader_validation = None, 
        scaler = scaler,
        use_amp = use_amp,
    )

    

    batchsize=batchsize
    chunksize=chunksize,
    quantize=quantize,
    #print(device)

    model.to(device)
    model.load(model_file, initialize_lazy = True)
    model.eval()
    
    return model