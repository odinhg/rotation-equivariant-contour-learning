import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from types import SimpleNamespace
import cv2
from copy import deepcopy

from utils import set_seed, split_dataset, get_number_of_parameters, print_config
from rotatouille import NodeRegressionModel 
from baseline import Baseline1dCNNRegressor 
from dataset import ContourDatasetRegression

global_config = {
        "datafile_train": "datasets/generated_data/train_curvature_contours.parquet",
        "datafile_test": "datasets/generated_data/test_curvature_contours.parquet",
        "seed": 0,
        "batch_size": 32,
        "num_workers": 0,#4,
        "num_epochs": 100,
        "num_runs": 10,
        "lr": 1e-3,
        "device": "cuda:7" if torch.cuda.is_available() else "cpu",
        "dataset_constructor": ContourDatasetRegression,
        "dataset_kwargs": {
            "length": 100,#64,
            "channels": 1,
        },
        "scheduler_step_size": 10,
        "scheduler_gamma": 0.5,
    }

config_contour = {
        "description": "Contour curvature regression using equivariant layers.",
        "model_constructor": NodeRegressionModel,
        "model_kwargs": {},
        }

config_baseline = {
        "description": "Baseline 1D CNN curvature regression.",
        "model_constructor": Baseline1dCNNRegressor,
        "model_kwargs": {},
        }

configs = {
        "contour": config_contour,
        "baseline": config_baseline,
        }

# Model type from command line argument
available_models = list(configs.keys())
parser = argparse.ArgumentParser(description="Train a model for curvature regression.")
parser.add_argument("model", type=str, choices=available_models, help="Model to train")
args = parser.parse_args()
config_model = configs[args.model]
config = SimpleNamespace(**global_config, **config_model)

rng = set_seed(config.seed, return_generator=True)

train_dataset = config.dataset_constructor(config.datafile_train, **config.dataset_kwargs)
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, generator=rng)

test_dataset = config.dataset_constructor(config.datafile_test, **config.dataset_kwargs)
test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

test_losses = []
for run in range(config.num_runs):
    model = config.model_constructor(**config.model_kwargs).to(config.device)
    print(f"Model has {get_number_of_parameters(model)} parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    criterion = nn.L1Loss()
    for epoch in range(config.num_epochs):
        train_losses = []
        for example in (pbar := tqdm(train_dataloader, desc="Training")):
            example = {key: value.to(config.device) for key, value in example.items()}
            outputs = model(example)
            loss = criterion(outputs, example["curvature"])
            loss.backward()
            optimizer.step()
            model.zero_grad()
            train_losses.append(loss.item())
            mean_train_loss = np.mean(train_losses)
            pbar.set_description(f"Run {run+1} - Epoch {epoch+1} | Training Loss: {mean_train_loss:.4f}")

        scheduler.step()

    losses = []
    for example in tqdm(test_dataloader, desc="Testing"):
        example = {key: value.to(config.device) for key, value in example.items()}
        outputs = model(example)
        loss = criterion(outputs, example["curvature"])
        losses.append(loss.cpu().detach().numpy())
    test_loss = np.mean(losses)
    test_losses.append(test_loss)
    print(f"Test Loss: {test_loss:.4f}")

print(f"Test Losses: {test_losses}")
print(f"Mean Test Loss: {np.mean(test_losses):.4f} ± {np.std(test_losses):.4f}")
