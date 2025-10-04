import argparse
from types import SimpleNamespace
import numpy as np
from tqdm import tqdm
from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import global_config, configs
from utils import set_seed, print_config, split_dataset, get_number_of_parameters
from transforms import rotate_batch

def compute_accuracy(model, data_loader, criterion, device, rotate: bool=False, repetitions: int=1):
    correct = 0
    total = 0
    val_losses = []
    with torch.no_grad():
        for seed in range(repetitions):
            for example in data_loader:
                if rotate:
                    example["data"] = rotate_batch(example["data"], seed)
                example = {k: v.to(device) for k, v in example.items()}
                output = model(example)
                target = example["label"]
                loss = criterion(output, target)
                val_losses.append(loss.item())
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

    accuracy = correct / total
    val_loss = np.mean(val_losses)

    return val_loss, accuracy

def train_model(config, rng):
    if config.val_data_file is not None:
        dataset_train = config.dataset_constructor(config.train_data_file, **config.dataset_kwargs)
        dataset_val = config.dataset_constructor(config.val_data_file, **config.dataset_kwargs)
    else:
        dataset = config.dataset_constructor(config.train_data_file, **config.dataset_kwargs)
        dataset_train, dataset_val = split_dataset(dataset, config.val_size, rng)

    train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, generator=rng, drop_last=True, num_workers=config.num_workers)
    val_loader = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = config.model_constructor(**config.model_kwargs).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    
    print(f"Training dataset size: {len(dataset_train)}")
    print(f"Validation dataset size: {len(dataset_val)}")

    print_config(config)

    num_params = get_number_of_parameters(model)
    print(f"Number of parameters in the model: {num_params}")

    best_val_accuracy = -1.0
    best_epoch = 0
    best_state_dict = None

    for epoch in range(config.n_epochs):
        model.train()
        train_losses = []
        for example in (pbar := tqdm(train_loader, position=0, leave=False)):
            optimizer.zero_grad()
            if config.rotate_train:
                example["data"] = rotate_batch(example["data"], seed=epoch) 

            example = {k: v.to(config.device) for k, v in example.items()}
            output = model(example)
            target = example["label"]
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            mean_train_loss = np.mean(train_losses)
            pbar.set_description(f"[{epoch + 1:03}|{config.n_epochs:03}] TrainLoss: {mean_train_loss:.3f} – LR: {scheduler.get_last_lr()[0]:.2E} – BestValAcc: {best_val_accuracy:.3f} (epoch {best_epoch + 1})")
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            rotate = config.repetitions > 1
            _, val_accuracy = compute_accuracy(model, val_loader, criterion, config.device, rotate=rotate, repetitions=1)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
    print(f"Best model at epoch {best_epoch + 1} with validation accuracy {best_val_accuracy:.3f}")
    return best_state_dict

def evaluate_model(config, state_dict):
    model = config.model_constructor(**config.model_kwargs).to(config.device)
    model.load_state_dict(state_dict)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    dataset_test = config.dataset_constructor(config.test_data_file, **config.dataset_kwargs)
    test_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    rotate = config.repetitions > 1
    _, test_accuracy = compute_accuracy(model, test_loader, criterion, config.device, rotate=rotate, repetitions=config.repetitions)

    return test_accuracy


if __name__ == "__main__":
    available_models = list(configs.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, choices=available_models, help="Model to train")
    args = parser.parse_args()
    config_model = configs[args.model]
    config = SimpleNamespace(**global_config, **config_model)

    test_accuracies = []
    for run in range(config.n_runs):
        rng = set_seed(config.seed + run, return_generator=True) # Use a different seed for each run
        print(f"Run {run + 1}/{config.n_runs}")
        state_dict = train_model(config, rng)
        test_accuracy = evaluate_model(config, state_dict)
        test_accuracies.append(test_accuracy)
        print(f"Test accuracy for run {run + 1}: {test_accuracy:.4f}")

    print(f"Accuracies for all runs: {test_accuracies}")
    print(f"Average overall accuracy: {np.mean(test_accuracies):.3f} ± {np.std(test_accuracies):.3f}")
