import torch
import torch.nn as nn
import numpy as np
import random

def set_seed(seed, return_generator=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if return_generator:
        return torch.Generator().manual_seed(seed)

def print_config(config):
    print("Configuration:")
    for key, value in vars(config).items():
        print(f"\t{key}: {value}")

def split_dataset(dataset, val_size, rng):
    """
    Splits a dataset into a training and validation set for early stopping. Use torch.utils.data.random_split for a random split.
    """
    n_val = int(val_size * len(dataset))
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val], rng)
    return train_dataset, val_dataset

def get_number_of_parameters(model: nn.Module) -> int:
    """
    Returns the number of (real) parameters in the model. 
    """
    num_params = 0
    for param in model.parameters():
        if param.is_complex():
            num_params += param.numel() * 2
        else:
            num_params += param.numel()
    return num_params
