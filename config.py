import torch
from rotatouille import ContourClassifier
from baseline.models import BaselineClassifier2d, CycleGCNClassifier
from dataset import ContourDataset, ImageDataset

global_config = {
        "seed": 0,                  # Random seed
        "n_runs": 10,               # Number of runs to average results over
        "device": "cuda:7" if torch.cuda.is_available() else "cpu",  # Device to use for training 
        "num_workers": 4,           # Number of workers for the DataLoader
        }

ours_fashion_mnist = {
        "description": "Our classifier on the Fashion MNIST contour dataset",
        "val_size": 0.1,
        "n_epochs": 200,
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
        "model_kwargs": {
            "n_classes": 10,
            "feature_extractor_layers": [
                {"in_channels": 1, "out_channels": 8, "kernel_size": 9, "p": 1, "method": "learnable"},
                {"in_channels": 8, "out_channels": 8, "kernel_size": 9, "p": 2, "method": "learnable"},
                {"in_channels": 8, "out_channels": 16, "kernel_size": 9, "p": 1, "method": "learnable"},
                {"in_channels": 16, "out_channels": 16, "kernel_size": 9, "p": 2, "method": "learnable"},
                {"in_channels": 16, "out_channels": 35, "kernel_size": 9, "p": 1, "method": "learnable"},
                {"in_channels": 35, "out_channels": 35, "kernel_size": 9, "p": 1, "method": "learnable"},
                {"in_channels": 35, "out_channels": 10, "kernel_size": 9, "p": 1, "method": "learnable"},
                ],
            "extra_features_dim": 0,
            "fcnn_hidden_dim": 128,
            },
        "lr": 5e-4,
        "scheduler_step_size": 50,
        "scheduler_gamma": 0.1,
        "rotate_train": False,
        }

baseline_gnn_fashion_mnist = {
        "description": "Baseline GNN (graph convolution) on Fashion MNIST contours.",
        "val_size": 0.1,
        "n_epochs": 200,
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
        "model_constructor": CycleGCNClassifier, 
        "model_kwargs": {
            "n": 128,
            "in_dim": 2,
            "hidden_dim": 128,
            "n_layers": 5,
            "n_classes": 10,
            },
        "lr": 1e-3,
        "scheduler_step_size": 50,
        "scheduler_gamma": 0.5,
        "rotate_train": True,
        }

baseline_contour_fashion_mnist = {
        "description": "Baseline 2D CNN on Fashion MNIST contour images dataset",
        "val_size": 0.1,
        "n_epochs": 200,
        "batch_size": 128,
        "repetitions": 10,
        "dataset_constructor": ImageDataset,
        "train_data_file": "datasets/generated_data/fashion_mnist_train.parquet",
        "val_data_file": None,
        "test_data_file": "datasets/generated_data/fashion_mnist_test.parquet",
        "dataset_kwargs": {
            "image_type": "contour_image",
            "height": 28,
            "width": 28,
            "mean": 0.1037,
            "std": 0.3049,
            },
        "model_constructor": BaselineClassifier2d,
        "model_kwargs": {
            "n_classes": 10,
            "channels": 1,
            },
        "lr": 1e-2,
        "scheduler_step_size": 25,
        "scheduler_gamma": 0.5,
        "rotate_train": True,
        }

baseline_filled_fashion_mnist = {
        "description": "Baseline 2D CNN on Fashion MNIST filled images dataset",
        "val_size": 0.1,
        "n_epochs": 200,
        "batch_size": 128,
        "repetitions": 10,
        "dataset_constructor": ImageDataset,
        "train_data_file": "datasets/generated_data/fashion_mnist_train.parquet",
        "val_data_file": None,
        "test_data_file": "datasets/generated_data/fashion_mnist_test.parquet",
        "dataset_kwargs": {
            "image_type": "filled_image",
            "height": 28,
            "width": 28,
            "mean": 0.4775,
            "std": 0.4995,
            },
        "model_constructor": BaselineClassifier2d,
        "model_kwargs": {
            "n_classes": 10,
            "channels": 1,
            },
        "lr": 1e-2,
        "scheduler_step_size": 25,
        "scheduler_gamma": 0.5,
        "rotate_train": True,
        }

ours_rot_mnist = {
        "description": "Our classifier on the Rotated MNIST contour dataset (contours only)",
        "val_size": 0.05,
        "n_epochs": 200,
        "batch_size": 128,
        "repetitions": 1, 
        "dataset_constructor": ContourDataset,
        "train_data_file": "datasets/generated_data/rotated_mnist_train.parquet",
        "val_data_file": None,
        "test_data_file": "datasets/generated_data/rotated_mnist_test.parquet",
        "dataset_kwargs": {
            "length": 128,
            "channels": 1,
            "use_extra_features": False,
            },
        "model_constructor": ContourClassifier,
        "model_kwargs": {
            "n_classes": 10,
            "feature_extractor_layers": [
                {"in_channels": 1, "out_channels": 8, "kernel_size": 9, "p": 1, "method": "learnable"},
                {"in_channels": 8, "out_channels": 8, "kernel_size": 9, "p": 2, "method": "learnable"},
                {"in_channels": 8, "out_channels": 16, "kernel_size": 9, "p": 1, "method": "learnable"},
                {"in_channels": 16, "out_channels": 16, "kernel_size": 9, "p": 2, "method": "learnable"},
                {"in_channels": 16, "out_channels": 35, "kernel_size": 9, "p": 1, "method": "learnable"},
                {"in_channels": 35, "out_channels": 35, "kernel_size": 9, "p": 1, "method": "learnable"},
                {"in_channels": 35, "out_channels": 10, "kernel_size": 9, "p": 1, "method": "learnable"},
                ],
            "extra_features_dim": 0,
            "fcnn_hidden_dim": 128,
            },
        "lr": 5e-4,
        "scheduler_step_size": 50,
        "scheduler_gamma": 0.1,
        "rotate_train": False,
        }

ours_rh_rot_mnist = {
        "description": "Our classifier on the Rotated MNIST contour dataset (contours and radial histogram)",
        "val_size": 0.05,
        "n_epochs": 200,
        "batch_size": 128,
        "repetitions": 1,
        "dataset_constructor": ContourDataset,
        "train_data_file": "datasets/generated_data/rotated_mnist_train.parquet",
        "val_data_file": None,
        "test_data_file": "datasets/generated_data/rotated_mnist_test.parquet",
        "dataset_kwargs": {
            "length": 128,
            "channels": 1,
            "use_extra_features": True,
            },
        "model_constructor": ContourClassifier,
        "model_kwargs": {
            "n_classes": 10,
            "feature_extractor_layers": [
                {"in_channels": 1, "out_channels": 8, "kernel_size": 9, "p": 1, "method": "learnable"},
                {"in_channels": 8, "out_channels": 8, "kernel_size": 9, "p": 2, "method": "learnable"},
                {"in_channels": 8, "out_channels": 16, "kernel_size": 9, "p": 1, "method": "learnable"},
                {"in_channels": 16, "out_channels": 16, "kernel_size": 9, "p": 2, "method": "learnable"},
                {"in_channels": 16, "out_channels": 35, "kernel_size": 9, "p": 1, "method": "learnable"},
                {"in_channels": 35, "out_channels": 35, "kernel_size": 9, "p": 1, "method": "learnable"},
                {"in_channels": 35, "out_channels": 10, "kernel_size": 9, "p": 1, "method": "learnable"},
                ],
            "extra_features_dim": 14,
            "fcnn_hidden_dim": 128,
            },
        "lr": 5e-4,
        "scheduler_step_size": 50,
        "scheduler_gamma": 0.1,
        "rotate_train": False,
        }

ours_modelnet = {
        "description": "Our classifier on the Modelnet contour dataset",
        "val_size": 0.1,
        "n_epochs": 200,
        "batch_size": 16,
        "repetitions": 10,
        "dataset_constructor": ContourDataset,
        "train_data_file": "datasets/generated_data/modelnet_train.parquet",
        "val_data_file": None,
        "test_data_file": "datasets/generated_data/modelnet_test.parquet",
        "dataset_kwargs": {
            "length": 128,
            "channels": 4,
            "use_extra_features": False,
            },
        "model_constructor": ContourClassifier,
        "model_kwargs": {
            "n_classes": 4,
            "feature_extractor_layers": [
                {"in_channels": 4, "out_channels": 8, "kernel_size": 9, "p": 1, "method": "learnable"},
                {"in_channels": 8, "out_channels": 8, "kernel_size": 9, "p": 2, "method": "learnable"},
                {"in_channels": 8, "out_channels": 16, "kernel_size": 9, "p": 1, "method": "learnable"},
                {"in_channels": 16, "out_channels": 16, "kernel_size": 9, "p": 2, "method": "learnable"},
                {"in_channels": 16, "out_channels": 35, "kernel_size": 9, "p": 1, "method": "learnable"},
                {"in_channels": 35, "out_channels": 35, "kernel_size": 9, "p": 1, "method": "learnable"},
                {"in_channels": 35, "out_channels": 10, "kernel_size": 9, "p": 1, "method": "learnable"},
                ],
            "extra_features_dim": 0,
            "fcnn_hidden_dim": 128,
            },
        "lr": 5e-4,
        "scheduler_step_size": 50,
        "scheduler_gamma": 0.1,
        "rotate_train": False,
        }

baseline_filled_modelnet = {
        "description": "Baseline 2D CNN on ModelNet filled images dataset",
        "val_size": 0.1,
        "n_epochs": 200,
        "batch_size": 16,
        "repetitions": 10,
        "dataset_constructor": ImageDataset,
        "train_data_file": "datasets/generated_data/modelnet_train.parquet",
        "val_data_file": None,
        "test_data_file": "datasets/generated_data/modelnet_test.parquet",
        "dataset_kwargs": {
            "image_type": "filled_image",
            "channels": 4,
            "height": 64,
            "width": 64,
            "mean": 0.4157,
            "std": 0.4928,
            },
        "model_constructor": BaselineClassifier2d,
        "model_kwargs": {
            "n_classes": 4,
            "channels": 4,
            "input_size": 64,
            },
        "lr": 1e-2,
        "scheduler_step_size": 25,
        "scheduler_gamma": 0.5,
        "rotate_train": True,
        }

baseline_contour_modelnet = {
        "description": "Baseline 2D CNN on ModelNet contour images dataset",
        "val_size": 0.1,
        "n_epochs": 200,
        "batch_size": 16,
        "repetitions": 10,
        "dataset_constructor": ImageDataset,
        "train_data_file": "datasets/generated_data/modelnet_train.parquet",
        "val_data_file": None,
        "test_data_file": "datasets/generated_data/modelnet_test.parquet",
        "dataset_kwargs": {
            "image_type": "contour_image",
            "channels": 4,
            "height": 64,
            "width": 64,
            "mean": 0.0400,
            "std": 0.1959,
            },
        "model_constructor": BaselineClassifier2d,
        "model_kwargs": {
            "n_classes": 4,
            "channels": 4,
            "input_size": 64,
            },
        "lr": 1e-2,
        "scheduler_step_size": 25,
        "scheduler_gamma": 0.5,
        "rotate_train": True,
        }

baseline_gnn_modelnet = {
        "description": "Baseline GNN (graph convolution) on the ModelNet contour dataset.", 
        "val_size": 0.1,
        "n_epochs": 200,
        "batch_size": 16,
        "repetitions": 10,
        "dataset_constructor": ContourDataset,
        "train_data_file": "datasets/generated_data/modelnet_train.parquet",
        "val_data_file": None,
        "test_data_file": "datasets/generated_data/modelnet_test.parquet",
        "dataset_kwargs": {
            "length": 128,
            "channels": 4,
            "use_extra_features": False,
            },
        "model_constructor": CycleGCNClassifier, 
        "model_kwargs": {
            "n": 128,
            "in_dim": 8,
            "hidden_dim": 128,
            "n_layers": 5,
            "n_classes": 10,
            },
        "lr": 1e-3,
        "scheduler_step_size": 50,
        "scheduler_gamma": 0.5,
        "rotate_train": True,
        }


# Dictionary of configurations
configs = {
    "ours_fashion_mnist": ours_fashion_mnist,
    "baseline_gnn_fashion_mnist": baseline_gnn_fashion_mnist,
    "baseline_contour_fashion_mnist": baseline_contour_fashion_mnist,
    "baseline_filled_fashion_mnist": baseline_filled_fashion_mnist,
    
    "ours_rot_mnist": ours_rot_mnist,
    "ours_rh_rot_mnist": ours_rh_rot_mnist,

    "baseline_filled_modelnet": baseline_filled_modelnet,
    "baseline_contour_modelnet": baseline_contour_modelnet,
    "baseline_gnn_modelnet": baseline_gnn_modelnet,
    "ours_modelnet": ours_modelnet,
}

