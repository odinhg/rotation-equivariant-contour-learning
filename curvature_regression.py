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
from sklearn.metrics import r2_score

from utils import set_seed, split_dataset, get_number_of_parameters, print_config
from rotatouille import NodeRegressionModel 
from baseline import Baseline1dCNNRegressor, CycleGCNRegressor
from dataset import ContourDatasetRegression
from transforms import rotate_batch

global_config = {
        "datafile_train": "datasets/generated_data/train_curvature_contours.parquet",
        "datafile_test": "datasets/generated_data/test_curvature_contours.parquet",
        "seed": 0,
        "batch_size": 32,
        "num_workers": 4,
        "num_epochs": 100,
        "num_runs": 10,
        "lr": 1e-3,
        "device": "cuda:1" if torch.cuda.is_available() else "cpu",
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
        "rotate_train": False,
        }

config_baseline_cnn = {
        "description": "Baseline 1D CNN curvature regression.",
        "model_constructor": Baseline1dCNNRegressor,
        "model_kwargs": {},
        "rotate_train": True,
        }

config_baseline_gcn = {
        "description": "Baseline GCN curvature regression.",
        "model_constructor": CycleGCNRegressor,
        "model_kwargs": {
            "n": 100,
            "in_dim": 2,
            "hidden_dim": 128,
            "n_layers": 3,
        },
        "rotate_train": True,
        }

configs = {
        "contour": config_contour,
        "baseline_cnn": config_baseline_cnn, 
        "baseline_gcn": config_baseline_gcn,
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
test_r2_scores = []
for run in range(config.num_runs):
    model = config.model_constructor(**config.model_kwargs).to(config.device)
    print(f"Model has {get_number_of_parameters(model)} parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    criterion = nn.L1Loss()
    for epoch in range(config.num_epochs):
        train_losses = []
        for example in (pbar := tqdm(train_dataloader, desc="Training")):
            if config.rotate_train:
                example["data"] = rotate_batch(example["data"], seed=epoch)
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
    y_target = []
    y_pred = []
    for example in tqdm(test_dataloader, desc="Testing"):
        example = {key: value.to(config.device) for key, value in example.items()}
        outputs = model(example)
        loss = criterion(outputs, example["curvature"])
        losses.append(loss.cpu().detach().numpy())
        y_target.append(example["curvature"].cpu().detach().numpy())
        y_pred.append(outputs.cpu().detach().numpy())
    test_loss = np.mean(losses)
    test_losses.append(test_loss)
    y_target = np.concatenate(y_target, axis=0).flatten()
    y_pred = np.concatenate(y_pred, axis=0).flatten()
    test_r2 = r2_score(y_target, y_pred)
    test_r2_scores.append(test_r2)
    print(f"Test Loss: {test_loss:.4f} | R2 Score: {test_r2:.4f}")

print(f"Test Losses: {test_losses}")
print(f"Mean Test Loss: {np.mean(test_losses):.4f} ± {np.std(test_losses):.4f}")
print(f"Test R2 Scores: {test_r2_scores}")
print(f"Mean Test R2 Score: {np.mean(test_r2_scores):.4f} ± {np.std(test_r2_scores):.4f}")
