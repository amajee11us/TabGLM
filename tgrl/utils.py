import os
import pandas as pd
import openml
import data_preprocess as dp
import torch
import random 
import yaml
from numpy.random import RandomState, SeedSequence, MT19937
import numpy as np
from typing import Optional
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import pearsonr

from transformers.utils import logging
logging.set_verbosity_error()

'''
Set the seed values for consistent performance metrics
'''
def set_seed(seed, disable_cudnn=False):
    torch.manual_seed(seed)                       # Seed the RNG for all devices (both CPU and CUDA).
    torch.cuda.manual_seed_all(seed)              # If you are using multi-GPU. In case of one GPU, you can use # torch.cuda.manual_seed(SEED).
    np.random.seed(seed)             
    random.seed(seed)                             # Set python seed for custom operators.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    rs = RandomState(MT19937(SeedSequence(seed))) # If any of the libraries or code rely on NumPy seed the global NumPy RNG.
  
    if not disable_cudnn:
        torch.backends.cudnn.benchmark = False    # Causes cuDNN to deterministically select an algorithm, 
                                                  # possibly at the cost of reduced performance 
                                                  # (the algorithm itself may be nondeterministic).
        torch.backends.cudnn.deterministic = True # Causes cuDNN to use a deterministic convolution algorithm, 
                                                  # but may slow down performance.
                                                  # It will not guarantee that your training process is deterministic 
                                                  # if you are using other libraries that may use nondeterministic algorithms 
    else:
        torch.backends.cudnn.enabled = False # Controls whether cuDNN is enabled or not. 
                                             # If you want to enable cuDNN, set it to True.

def normalize_adj_matrix(adj):
    D = torch.sum(adj, 0)
    D_hat = torch.diag(((D) ** (-0.5)))
    adj_normalized = torch.mm(torch.mm(D_hat, adj), D_hat)
    return adj_normalized


def get_config(config_path):     
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    data_config = config['data_config']    
    fit_config = config['fit_config']    
    return data_config, fit_config

def to_dense_array(series):
    # Check if the series is stored as a sparse array
    if isinstance(series.array, pd.arrays.SparseArray):
        # Convert SparseArray to a dense NumPy array
        return series.sparse.to_dense().to_numpy()
    else:
        # If it's not a sparse array, just convert to NumPy array directly
        return series.to_numpy()