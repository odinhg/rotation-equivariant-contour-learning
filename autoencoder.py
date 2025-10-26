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
import random

from utils import set_seed, split_dataset, get_number_of_parameters, print_config
from rotatouille import ShapeAutoEncoder
from baseline import BaselineAutoencoder2d
from dataset import ContourDatasetAutoencoder, ImageDatasetAutoencoder
from transforms import rotate_batch

global_config = {
    "data_file": "datasets/generated_data/cell_segmentations.parquet",
    "val_size": 0.1,
    "seed": 0,
    "batch_size": 32,
    "num_workers": 4,
    "epochs": 200,
    "device": "cuda:1" if torch.cuda.is_available() else "cpu",
    "lr": 0.001,
}

config_contour = {
    "description": "Autoencoder for contour data on PCST cell dataset.",
    "model_constructor": ShapeAutoEncoder,
    "model_kwargs": {
        "layers": [
            {"in_channels": 1, "out_channels": 4, "kernel_size": 11, "p": 2},
            {"in_channels": 4, "out_channels": 4, "kernel_size": 9, "p": 2},
            {"in_channels": 4, "out_channels": 4, "kernel_size": 7, "p": 2},
            {"in_channels": 4, "out_channels": 4, "kernel_size": 5, "p": 2},
            {"in_channels": 4, "out_channels": 4, "kernel_size": 3, "p": 2},
        ]
    },
    "dataset_constructor": ContourDatasetAutoencoder,
    "dataset_kwargs": {
        "length": 128,
        "channels": 1,
    },
    "criterion": lambda output, target: F.mse_loss(torch.view_as_real(output), torch.view_as_real(target)),
    "checkpoint_path": Path("checkpoints/best_autoencoder_contour.pth"),
    "rotate_train": False,
}

config_image = {
    "description": "Autoencoder for image data on PCST cell dataset.",
    "model_constructor": BaselineAutoencoder2d,
    "model_kwargs": {
        "channels": 1,
    },
    "dataset_constructor": ImageDatasetAutoencoder,
    "dataset_kwargs": {
        "height": 128,
        "width": 128,
        "mean": 0.1748,
        "std": 0.3798,
        "channels": 1,
    },
    "criterion": F.mse_loss,
    "checkpoint_path": Path("checkpoints/best_autoencoder_image.pth"),
    "rotate_train": True,
}

configs = {
    "contour": config_contour,
    "image": config_image,
}

# Model type from command line argument
available_models = list(configs.keys())
parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, choices=available_models, help="Model to train")
args = parser.parse_args()
config_model = configs[args.model]
config = SimpleNamespace(**global_config, **config_model)

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

dataset = config.dataset_constructor(config.data_file, **config.dataset_kwargs)
rng = set_seed(config.seed, return_generator=True)
dataset_train, dataset_val = split_dataset(dataset, config.val_size, rng)
train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, generator=rng, drop_last=True, num_workers=config.num_workers, persistent_workers=True)
val_loader = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False, num_workers=0)

model = config.model_constructor(**config.model_kwargs).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

print_config(config)
num_params = get_number_of_parameters(model)
print(f"Number of parameters in the autoencoder model: {num_params}")

def train_autoencoder(model, train_loader, val_loader, optimizer, config):
    best_val_loss = float("inf")
    model.train()
    for epoch in range(config.epochs):
        total_loss = 0.0
        for batch in tqdm(train_loader):
            if config.rotate_train:
                batch["data"] = rotate_batch(batch["data"], seed=epoch)
            batch = {key: value.to(config.device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            optimizer.zero_grad()
            output = model(batch)
            target = batch["data"]
            loss = config.criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{config.epochs}], Loss: {avg_loss:.4f}")

        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in val_loader:
                batch = {key: value.to(config.device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
                output = model(batch)
                target = batch["data"]
                loss = config.criterion(output, target) 
                val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.checkpoint_path) 
        model.train()

# If best_autoencoder.pth exists, load it
if not Path(config.checkpoint_path).is_file():
    print("No pre-trained autoencoder model found. Training from scratch.")
    train_autoencoder(model, train_loader, val_loader, optimizer, config)

# Visualize reconstructions for both image and contour models 
print("Generating reconstructions for visualization...")
def compute_reconstructions(model, loader, config, unnormalize=False):
    model.eval()
    originals = []
    reconstructions = []
    with torch.no_grad():
        for batch in tqdm(loader):
            data = batch["data"].to(config.device)
            output = model({"data": data})
            if unnormalize:
                output = (output * config.dataset_kwargs["std"]) + config.dataset_kwargs["mean"]
                data = (data * config.dataset_kwargs["std"]) + config.dataset_kwargs["mean"]
            originals.append(data.cpu())
            reconstructions.append(output.cpu())
    return torch.cat(originals, dim=0), torch.cat(reconstructions, dim=0)

config_image = SimpleNamespace(**global_config, **config_image)
config_contour = SimpleNamespace(**global_config, **config_contour)

model_image = config_image.model_constructor(**config_image.model_kwargs).to(config.device) 
model_contour = config_contour.model_constructor(**config_contour.model_kwargs).to(config.device) 

if not config_image.checkpoint_path.is_file() or not config_contour.checkpoint_path.is_file():
    raise FileNotFoundError(f"Checkpoint files not found: {config_image.checkpoint_path} or {config_contour.checkpoint_path}. Please train both models first.")

model_image.load_state_dict(torch.load(config_image.checkpoint_path, weights_only=True, map_location=config.device))
model_contour.load_state_dict(torch.load(config_contour.checkpoint_path, weights_only=True, map_location=config.device))

dataset_image = config_image.dataset_constructor(config.data_file, **config_image.dataset_kwargs)
dataset_contour = config_contour.dataset_constructor(config.data_file, **config_contour.dataset_kwargs)

rng = set_seed(config.seed, return_generator=True)
_, dataset_image = split_dataset(dataset_image, config.val_size, rng)
rng = set_seed(config.seed, return_generator=True)
_, dataset_contour = split_dataset(dataset_contour, config.val_size, rng)

loader_image = DataLoader(dataset_image, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
loader_contour = DataLoader(dataset_contour, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)


data = {"original_image": [], "reconstructed_image": [], "original_contour": [], "reconstructed_contour": []}
data["original_image"], data["reconstructed_image"] = compute_reconstructions(model_image, loader_image, config_image, unnormalize=True)
data["original_contour"], data["reconstructed_contour"] = compute_reconstructions(model_contour, loader_contour, config_contour)

def contour_to_binary_image(contour, image_size=128):
    contour = np.asarray(contour, dtype=np.float32)
    min_xy = contour.min(axis=0)
    max_xy = contour.max(axis=0)
    size = max_xy - min_xy
    max_extent = size.max()

    # Scale to fit inside padded image region
    scale = image_size / max_extent
    scaled_contour = (contour - min_xy) * scale

    # Compute centroid and offset to center the shape
    center_offset = (image_size / 2) - (scaled_contour.mean(axis=0))
    centered_contour = scaled_contour + center_offset

    # Round and convert to int for OpenCV
    int_contour = np.round(centered_contour).astype(np.int32)

    # Create blank image and fill polygon
    image = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.fillPoly(image, [int_contour], 255)

    return (image / 255).astype(np.float32)

def plot_reconstruction(original_image, reconstructed_image, original_contour, reconstructed_contour, filename, dpi=50):
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))

    axs[0].imshow(original_image, cmap="gray")
    axs[0].axis("off")

    reconstructed_image = (reconstructed_image > 0.5).astype(np.float32)
    axs[1].imshow(reconstructed_image, cmap="gray")
    axs[1].axis("off")

    axs[2].plot(original_contour[:, 0], original_contour[:, 1], color="black")
    axs[2].axis("equal")
    axs[2].axis("off")

    # Close contour by appending the first point to the end
    reconstructed_contour = np.vstack([reconstructed_contour, reconstructed_contour[0]])
    axs[3].plot(reconstructed_contour[:, 0], reconstructed_contour[:, 1], color="black")
    axs[3].axis("equal")
    axs[3].axis("off")

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=dpi)
    plt.close(fig)

reconstructions_dir = Path("reconstructions")
reconstructions_dir.mkdir(exist_ok=True)

print("Saving reconstruction visualizations to disk...")
for _ in range(50): 
    idx = random.randint(0, len(data["original_image"]) - 1)
    original_image = data["original_image"][idx].cpu().squeeze().numpy()
    reconstructed_image = data["reconstructed_image"][idx].cpu().squeeze().numpy()

    original_contour = deepcopy(data["original_contour"][idx])
    original_contour = torch.view_as_real(original_contour).cpu().squeeze().numpy()
    original_contour[:, 1] = -original_contour[:, 1] # Conjugate flip for visualization

    reconstructed_contour = deepcopy(data["reconstructed_contour"][idx])
    reconstructed_contour = torch.view_as_real(reconstructed_contour).cpu().squeeze().numpy()
    reconstructed_contour[:, 1] = -reconstructed_contour[:, 1]

    filename = reconstructions_dir / f"{idx:08d}.png"
    plot_reconstruction(original_image, reconstructed_image, original_contour, reconstructed_contour, filename, dpi=300)

def compute_iou(img1, img2, threshold=0.5):
    mask1 = (img1 > threshold)
    mask2 = (img2 > threshold)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union

print("Computing IoU scores...")
# Compute IoUs for images
ious_image = []
for i in range(len(data["original_image"])):
    original = data["original_image"][i].cpu().squeeze().numpy()
    reconstructed = data["reconstructed_image"][i].cpu().squeeze().numpy()

    iou = compute_iou(original, reconstructed)
    ious_image.append(iou)

mean_iou_image = np.mean(ious_image)
print(f"Mean IoU for image reconstructions: {mean_iou_image:.3f}")

# Compute IoUs for contours
ious_contour = []
for i in range(len(data["original_contour"])):
    original = data["original_contour"][i]
    original = torch.view_as_real(original).cpu().squeeze().numpy()
    original = contour_to_binary_image(original)

    reconstructed = data["reconstructed_contour"][i]
    reconstructed = torch.view_as_real(reconstructed).cpu().squeeze().numpy()
    reconstructed = contour_to_binary_image(reconstructed)
    
    iou = compute_iou(original, reconstructed)
    ious_contour.append(iou)

mean_iou_contour = np.mean(ious_contour)
print(f"Mean IoU for contour reconstructions: {mean_iou_contour:.3f}")
