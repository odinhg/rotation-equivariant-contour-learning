import argparse
from types import SimpleNamespace
import numpy as np
from tqdm import tqdm
from pathlib import Path
import copy
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import global_config, configs
from utils import set_seed, print_config, split_dataset, get_number_of_parameters
from transforms import rotate_batch
from train_classifier import compute_accuracy, train_model, evaluate_model
from dataset import ContourDataset
from rotatouille import ContourClassifier

def run_coarsening_method_experiment(global_config):
    method_values = ["learnable", "average", "max"]
    strided_pool_values = [False, True]

    results = {}

    for method, strided_pool in product(method_values, strided_pool_values):
        print(f"Running experiment with method={method}, strided_pool={strided_pool}")

        experiment_config = {
                "model_kwargs": {
                "n_classes": 10,
                "feature_extractor_layers": [
                    {"in_channels": 1, "out_channels": 8, "kernel_size": 9, "p": 1, "method": method, "strided_pool": strided_pool},
                    {"in_channels": 8, "out_channels": 8, "kernel_size": 9, "p": 2, "method": method, "strided_pool": strided_pool},
                    {"in_channels": 8, "out_channels": 16, "kernel_size": 9, "p": 1, "method": method, "strided_pool": strided_pool},
                    {"in_channels": 16, "out_channels": 16, "kernel_size": 9, "p": 2, "method": method, "strided_pool": strided_pool},
                    {"in_channels": 16, "out_channels": 35, "kernel_size": 9, "p": 1, "method": method, "strided_pool": strided_pool},
                    {"in_channels": 35, "out_channels": 35, "kernel_size": 9, "p": 1, "method": method, "strided_pool": strided_pool},
                    {"in_channels": 35, "out_channels": 10, "kernel_size": 9, "p": 1, "method": method, "strided_pool": strided_pool},
                    ],
                "extra_features_dim": 0,
                "fcnn_hidden_dim": 128,
                },
                }

        config = SimpleNamespace(**global_config, **experiment_config)

        test_accuracies = []
        for run in range(config.n_runs):
            rng = set_seed(config.seed + run, return_generator=True)
            print(f"Run {run + 1}/{config.n_runs}")
            state_dict = train_model(config, rng)
            test_accuracy = evaluate_model(config, state_dict)
            test_accuracies.append(test_accuracy)
            print(f"Test accuracy for run {run + 1}: {test_accuracy:.4f}")

        print(f"\nResults for experiment with {method=}, {strided_pool=}:")
        print(f"Accuracies for all runs: {test_accuracies}")
        print(f"Average overall accuracy: {np.mean(test_accuracies):.3f} ± {np.std(test_accuracies):.3f}")
        results[(method, strided_pool)] = (np.mean(test_accuracies), np.std(test_accuracies))
        
    print("\nFinal results for all experiments:")
    for (method, strided_pool), (mean_acc, std_acc) in results.items():
        print(f"{method=}, {strided_pool=}: {mean_acc:.3f} ± {std_acc:.3f}")

def run_no_coarsening_experiment(global_config):
    print("Running experiment with no coarsening layers")

    experiment_config = {
            "model_kwargs": {
            "n_classes": 10,
            "feature_extractor_layers": [
                
                {"in_channels": 1, "out_channels": 8, "kernel_size": 9, "p": 1, "method": "mean"}, # p=1 for no coarsening
                {"in_channels": 8, "out_channels": 8, "kernel_size": 9, "p": 1, "method": "mean"},
                {"in_channels": 8, "out_channels": 16, "kernel_size": 9, "p": 1, "method": "mean"},
                {"in_channels": 16, "out_channels": 16, "kernel_size": 9, "p": 1, "method": "mean"},
                {"in_channels": 16, "out_channels": 35, "kernel_size": 9, "p": 1, "method": "mean"},
                {"in_channels": 35, "out_channels": 35, "kernel_size": 9, "p": 1, "method": "mean"},
                {"in_channels": 35, "out_channels": 10, "kernel_size": 9, "p": 1, "method": "mean"},
                ],
            "extra_features_dim": 0,
            "fcnn_hidden_dim": 128,
            },
            }

    config = SimpleNamespace(**global_config, **experiment_config)

    test_accuracies = []
    for run in range(config.n_runs):
        rng = set_seed(config.seed + run, return_generator=True)
        print(f"Run {run + 1}/{config.n_runs}")
        state_dict = train_model(config, rng)
        test_accuracy = evaluate_model(config, state_dict)
        test_accuracies.append(test_accuracy)
        print(f"Test accuracy for run {run + 1}: {test_accuracy:.4f}")

    print(f"\nResults for experiment with no coarsening layers:")
    print(f"Accuracies for all runs: {test_accuracies}")
    print(f"Average overall accuracy: {np.mean(test_accuracies):.3f} ± {np.std(test_accuracies):.3f}")

def run_activation_functions_experiment(global_config):
    af_values = ["modrelu", "modtanh", "siglog"]

    results = {}

    for af in af_values:
        print(f"Running experiment with activation_function={af}")

        experiment_config = {
                "model_kwargs": {
                "n_classes": 10,
                "feature_extractor_layers": [
                    {"in_channels": 1, "out_channels": 8, "kernel_size": 9, "p": 1, "method": "learnable", "af": af},
                    {"in_channels": 8, "out_channels": 8, "kernel_size": 9, "p": 2, "method": "learnable", "af": af},
                    {"in_channels": 8, "out_channels": 16, "kernel_size": 9, "p": 1, "method": "learnable", "af": af},
                    {"in_channels": 16, "out_channels": 16, "kernel_size": 9, "p": 2, "method": "learnable", "af": af},
                    {"in_channels": 16, "out_channels": 35, "kernel_size": 9, "p": 1, "method": "learnable", "af": af},
                    {"in_channels": 35, "out_channels": 35, "kernel_size": 9, "p": 1, "method": "learnable", "af": af},
                    {"in_channels": 35, "out_channels": 10, "kernel_size": 9, "p": 1, "method": "learnable", "af": af},
                    ],
                "extra_features_dim": 0,
                "fcnn_hidden_dim": 128,
                },
                }

        config = SimpleNamespace(**global_config, **experiment_config)

        test_accuracies = []
        for run in range(config.n_runs):
            rng = set_seed(config.seed + run, return_generator=True)
            print(f"Run {run + 1}/{config.n_runs}")
            state_dict = train_model(config, rng)
            test_accuracy = evaluate_model(config, state_dict)
            test_accuracies.append(test_accuracy)
            print(f"Test accuracy for run {run + 1}: {test_accuracy:.4f}")
        print(f"\nResults for experiment with {af=}:")
        print(f"Accuracies for all runs: {test_accuracies}")
        print(f"Average overall accuracy: {np.mean(test_accuracies):.3f} ± {np.std(test_accuracies):.3f}")
        results[af] = (np.mean(test_accuracies), np.std(test_accuracies))
    print("\nFinal results for all experiments:")
    for af, (mean_acc, std_acc) in results.items():
        print(f"{af=}: {mean_acc:.3f} ± {std_acc:.3f}")

def run_global_pooling_method_experiment(global_config):
    global_pool_method_values = ["average", "max", "learnable"]
    results = {}
    for method in global_pool_method_values:
        print(f"Running experiment with global_pooling_method={method}")

        experiment_config = {
                "model_kwargs": {
                "n_classes": 10,
                "feature_extractor_layers": [
                    {"in_channels": 1, "out_channels": 8, "kernel_size": 9, "p": 1, "method": "learnable", "global_pool_method": method},
                    {"in_channels": 8, "out_channels": 8, "kernel_size": 9, "p": 2, "method": "learnable", "global_pool_method": method},
                    {"in_channels": 8, "out_channels": 16, "kernel_size": 9, "p": 1, "method": "learnable", "global_pool_method": method},
                    {"in_channels": 16, "out_channels": 16, "kernel_size": 9, "p": 2, "method": "learnable", "global_pool_method": method},
                    {"in_channels": 16, "out_channels": 35, "kernel_size": 9, "p": 1, "method": "learnable", "global_pool_method": method},
                    {"in_channels": 35, "out_channels": 35, "kernel_size": 9, "p": 1, "method": "learnable", "global_pool_method": method},
                    {"in_channels": 35, "out_channels": 10, "kernel_size": 9, "p": 1, "method": "learnable", "global_pool_method": method},
                    ],
                "extra_features_dim": 0,
                "fcnn_hidden_dim": 128,
                },
                }

        config = SimpleNamespace(**global_config, **experiment_config)

        test_accuracies = []
        for run in range(config.n_runs):
            rng = set_seed(config.seed + run, return_generator=True)
            print(f"Run {run + 1}/{config.n_runs}")
            state_dict = train_model(config, rng)
            test_accuracy = evaluate_model(config, state_dict)
            test_accuracies.append(test_accuracy)
            print(f"Test accuracy for run {run + 1}: {test_accuracy:.4f}")
        print(f"\nResults for experiment with {method=}:")
        print(f"Accuracies for all runs: {test_accuracies}")
        print(f"Average overall accuracy: {np.mean(test_accuracies):.3f} ± {np.std(test_accuracies):.3f}")
        results[method] = (np.mean(test_accuracies), np.std(test_accuracies))
    print("\nFinal results for all experiments:")
    for method, (mean_acc, std_acc) in results.items():
        print(f"{method=}: {mean_acc:.3f} ± {std_acc:.3f}")

def run_contour_lengths_experiment(global_config):
    contour_length_values = [32, 64, 128, 256]
    contour_length_datasets = {
            32 : {
                "train" : "datasets/generated_data/fashion_mnist_32_train.parquet",
                "test" : "datasets/generated_data/fashion_mnist_32_test.parquet",
                },
            64 : {
                "train" : "datasets/generated_data/fashion_mnist_64_train.parquet",
                "test" : "datasets/generated_data/fashion_mnist_64_test.parquet",
                },
            128 : {
                "train" : "datasets/generated_data/fashion_mnist_train.parquet",
                "test" : "datasets/generated_data/fashion_mnist_test.parquet",
                },
            256 : {
                "train" : "datasets/generated_data/fashion_mnist_256_train.parquet",
                "test" : "datasets/generated_data/fashion_mnist_256_test.parquet",
                },
            }
    experiment_config = {
            "model_kwargs": {
            "n_classes": 10,
            "feature_extractor_layers": [
                {"in_channels": 1, "out_channels": 8, "kernel_size": 9, "p": 1, "method": "learnable", "af": "modrelu"}, 
                {"in_channels": 8, "out_channels": 8, "kernel_size": 9, "p": 2, "method": "learnable", "af": "modrelu"},
                {"in_channels": 8, "out_channels": 16, "kernel_size": 9, "p": 1, "method": "learnable", "af": "modrelu"},
                {"in_channels": 16, "out_channels": 16, "kernel_size": 9, "p": 2, "method": "learnable", "af": "modrelu"},
                {"in_channels": 16, "out_channels": 35, "kernel_size": 9, "p": 1, "method": "learnable", "af": "modrelu"},
                {"in_channels": 35, "out_channels": 35, "kernel_size": 9, "p": 1, "method": "learnable", "af": "modrelu"},
                {"in_channels": 35, "out_channels": 10, "kernel_size": 9, "p": 1, "method": "learnable", "af": "modrelu"},
                ],
            "extra_features_dim": 0,
            "fcnn_hidden_dim": 128,
            },
            }
    config = SimpleNamespace(**global_config, **experiment_config)

    results = {}

    for length in contour_length_values:
        print(f"Running experiment with contour_length={length}")

        config.dataset_kwargs["length"] = length
        config.train_data_file = contour_length_datasets[length]["train"]
        config.test_data_file = contour_length_datasets[length]["test"]

        test_accuracies = []
        for run in range(config.n_runs):
            rng = set_seed(global_config["seed"] + run, return_generator=True)
            print(f"Run {run + 1}/{config.n_runs}")
            state_dict = train_model(config, rng)
            test_accuracy = evaluate_model(config, state_dict)
            test_accuracies.append(test_accuracy)
            print(f"Test accuracy for run {run + 1}: {test_accuracy:.4f}")
        print(f"\nResults for experiment with {length=}:")
        print(f"Accuracies for all runs: {test_accuracies}")
        print(f"Average overall accuracy: {np.mean(test_accuracies):.3f} ± {np.std(test_accuracies):.3f}")
        results[length] = (np.mean(test_accuracies), np.std(test_accuracies))

    print("\nFinal results for all experiments:")

    for length, (mean_acc, std_acc) in results.items():
        print(f"{length=}: {mean_acc:.3f} ± {std_acc:.3f}")

def run_kernel_sizes_experiment(global_config):
    kernel_size_values = [3, 5, 7, 9, 11, 13]

    results = {}
    for kernel_size in kernel_size_values:
        print(f"Running experiment with kernel_size={kernel_size}")

        experiment_config = {
                "model_kwargs": {
                "n_classes": 10,
                "feature_extractor_layers": [
                    {"in_channels": 1, "out_channels": 8, "kernel_size": kernel_size, "p": 1, "method": "learnable", "af": "modrelu"}, 
                    {"in_channels": 8, "out_channels": 8, "kernel_size": kernel_size, "p": 2, "method": "learnable", "af": "modrelu"},
                    {"in_channels": 8, "out_channels": 16, "kernel_size": kernel_size, "p": 1, "method": "learnable", "af": "modrelu"},
                    {"in_channels": 16, "out_channels": 16, "kernel_size": kernel_size, "p": 2, "method": "learnable", "af": "modrelu"},
                    {"in_channels": 16, "out_channels": 35, "kernel_size": kernel_size, "p": 1, "method": "learnable", "af": "modrelu"},
                    {"in_channels": 35, "out_channels": 35, "kernel_size": kernel_size, "p": 1, "method": "learnable", "af": "modrelu"},
                    {"in_channels": 35, "out_channels": 10, "kernel_size": kernel_size, "p": 1, "method": "learnable", "af": "modrelu"},
                    ],
                "extra_features_dim": 0,
                "fcnn_hidden_dim": 128,
                },
                }

        config = SimpleNamespace(**global_config, **experiment_config)

        test_accuracies = []
        for run in range(config.n_runs):
            rng = set_seed(config.seed + run, return_generator=True)
            print(f"Run {run + 1}/{config.n_runs}")
            state_dict = train_model(config, rng)
            test_accuracy = evaluate_model(config, state_dict)
            test_accuracies.append(test_accuracy)
            print(f"Test accuracy for run {run + 1}: {test_accuracy:.4f}")
    
        print(f"\nResults for experiment with {kernel_size=}:")
        print(f"Accuracies for all runs: {test_accuracies}")
        print(f"Average overall accuracy: {np.mean(test_accuracies):.3f} ± {np.std(test_accuracies):.3f}")
        results[kernel_size] = (np.mean(test_accuracies), np.std(test_accuracies))

    print("\nFinal results for all experiments:")
    for kernel_size, (mean_acc, std_acc) in results.items():
        print(f"{kernel_size=}: {mean_acc:.3f} ± {std_acc:.3f}")



if __name__ == "__main__":
    global_config = {
            "seed": 0,                  # Random seed
            "n_runs": 10,               # Number of runs to average results over
            "device": "cuda:1" if torch.cuda.is_available() else "cpu",  # Device to use for training 
            "num_workers": 4,           # Number of workers for the DataLoader
            "description": "Ablation study experiments on Fashion-MNIST dataset",
            "val_size": 0.1,
            "n_epochs": 100,
            "batch_size": 128,
            "repetitions": 10,
            "dataset_constructor": ContourDataset,
            "train_data_file": "datasets/generated_data/fashion_mnist_train.parquet",
            "val_data_file": None,
            "test_data_file": "datasets/generated_data/fashion_mnist_test.parquet",
            "dataset_kwargs": {
                "length": 128,
                "channels": 1,
                "use_extra_features": False,
            },
            "model_constructor": ContourClassifier,
        "lr": 5e-4,
        "scheduler_step_size": 50,
        "scheduler_gamma": 0.1,
        "rotate_train": False,
        }

    experiments = ["coarsening_method", "activation_functions", "global_pooling_method", "no_coarsening", "contour_lengths", "kernel_sizes"]
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Name of the experiment to run", choices=experiments)
    args = parser.parse_args()

    if args.experiment == "coarsening_method":
        run_coarsening_method_experiment(global_config)
    elif args.experiment == "no_coarsening":
        run_no_coarsening_experiment(global_config)
    elif args.experiment == "activation_functions":
        run_activation_functions_experiment(global_config)
    elif args.experiment == "global_pooling_method":
        run_global_pooling_method_experiment(global_config)
    elif args.experiment == "contour_lengths":
        run_contour_lengths_experiment(global_config)
    elif args.experiment == "kernel_sizes":
        run_kernel_sizes_experiment(global_config)
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")
