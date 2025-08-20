from rotatouille import ShapeAutoEncoder
from baseline import BaselineAutoencoder2d
from dataset import ContourDatasetAutoencoder, ImageDatasetAutoencoder

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

def create_gif(input_filenames, output_filename):
    """
    Create a GIF from a list of image filenames.
    """
    images = [Image.open(filename) for filename in input_filenames]
    images[0].save(output_filename, save_all=True, append_images=images[1:], duration=100, loop=0)
    print(f"GIF saved to {output_filename}")

# Global settings
data_file = "datasets/generated_data/cell_segmentations.parquet"
np.random.seed(1234)
idx = np.random.randint(0, 9000)
n_images = 200 
dpi = 100
angles = np.linspace(0, 2 * np.pi, n_images)

# RotaTouille Shape Autoencoder
print(f"Generating visualization for RotaTouille Shape Autoencoder on contours...")
checkpoint_path = Path("checkpoints/best_autoencoder_contour.pth")
model = ShapeAutoEncoder(layers = [
    {"in_channels": 1, "out_channels": 4, "kernel_size": 11, "p": 2},
    {"in_channels": 4, "out_channels": 4, "kernel_size": 9, "p": 2},
    {"in_channels": 4, "out_channels": 4, "kernel_size": 7, "p": 2},
    {"in_channels": 4, "out_channels": 4, "kernel_size": 5, "p": 2},
    {"in_channels": 4, "out_channels": 4, "kernel_size": 3, "p": 2}],)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
dataset = ContourDatasetAutoencoder(data_file, length=128, channels=1)
model.eval()
filenames = []
for j, angle in tqdm(enumerate(angles), total=n_images):
    w = np.exp(1j * angle)
    winv = np.exp(1j * -angle)
    example = dataset[idx]
    example["data"] = example["data"] * w
    with torch.no_grad():
        example["data"] = example["data"].unsqueeze(0)
        output = model(example).squeeze(0)

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    original = torch.view_as_real(example["data"].squeeze()).numpy().T
    reconstruction = torch.view_as_real(output.squeeze()).numpy().T
    stabilized = torch.view_as_real(output.squeeze() * winv).numpy().T

    axs[0].plot(*original, c="black")
    axs[0].set_title("Original")
    axs[1].plot(*reconstruction, c="black")
    axs[1].set_title("Reconstruction")
    axs[2].plot(*(stabilized), c="black")
    axs[2].set_title(f"Stabilized Reconstruction")
    min_val = -6
    max_val = 6
    for ax in axs:
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect("equal")
        ax.axis("off")
    plt.tight_layout()
    plt.suptitle(f"RotaTouille (Angle: {angle:.2f})", fontsize=16)
    filename = f"resources/animation/contour/{j:04d}.png"
    filenames.append(filename)
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.1, dpi=dpi)
    plt.close(fig)
create_gif(filenames, "resources/animation/contour/rotatouille_contour.gif")

# Baseline Autoencoder (Image 2D CNN)
print(f"Generating visualization for Baseline Autoencoder on images...")
checkpoint_path = Path("checkpoints/best_autoencoder_image.pth")
model = BaselineAutoencoder2d(channels=1)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
dataset = ImageDatasetAutoencoder(data_file, height=128, width=128, mean=0.1748, std=0.3798, channels=1)
model.eval()
filenames = []
from scipy.ndimage import rotate
for j, angle in tqdm(enumerate(angles), total=n_images):
    # rotate image using scipy
    example = dataset[idx]
    example["data"] = rotate(example["data"].squeeze().numpy(), angle * 180 / np.pi, reshape=False, mode="nearest", order=1)
    example["data"] = torch.tensor(example["data"]).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    with torch.no_grad():
        output = model(example).squeeze(0)
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    original = example["data"].squeeze().numpy()
    reconstruction = output.squeeze().numpy()
    stabilized = rotate(reconstruction, -angle * 180 / np.pi, reshape=False, mode="nearest", order=1)
    axs[0].imshow(original, cmap="gray")
    axs[0].set_title("Original")
    axs[1].imshow(reconstruction, cmap="gray")
    axs[1].set_title("Reconstruction")
    axs[2].imshow(stabilized, cmap="gray")
    axs[2].set_title(f"Stabilized Reconstruction")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.suptitle(f"Baseline Autoencoder (Angle: {angle:.2f})", fontsize=16)
    filename = f"resources/animation/image/{j:04d}.png"
    filenames.append(filename)
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.1, dpi=dpi)
    plt.close(fig)
create_gif(filenames, "resources/animation/image/baseline_image.gif")
