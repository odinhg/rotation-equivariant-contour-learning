import trimesh
import numpy as np
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import cv2
import json
from skimage.morphology import dilation, disk
from tqdm import tqdm

from utils import find_contour, resample_contour

def get_slices(points, K):
    # Divide the points into K slices based on their z-coordinates
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    values = np.linspace(y_min, y_max, K + 1)
    slices = []
    for j in range(K):
        mask = (points[:, 1] >= values[j]) & (points[:, 1] < values[j + 1]) # Slice in y
        slice = points[mask]
        slices.append(slice)
    return slices

def create_binary_image(points_2d, img_size, x_min, x_max, z_min, z_max):
    # Project 3D points to 2D and create a binary image
    if len(points_2d) == 0:
        return np.zeros((img_size, img_size), dtype=np.uint8)  # Empty image

    # Add padding
    dx = x_max - x_min
    dz = z_max - z_min
    pad_x = (dx * 10) / img_size
    pad_z = (dz * 10) / img_size
    x_min -= pad_x
    x_max += pad_x
    z_min -= pad_z
    z_max += pad_z

    x_norm = (points_2d[:, 0] - x_min) / (x_max - x_min)
    z_norm = (points_2d[:, 2] - z_min) / (z_max - z_min)

    # Convert to pixel indices
    x_pix = (x_norm * (img_size - 1)).astype(int)
    z_pix = (z_norm * (img_size - 1)).astype(int)


    # Create binary image
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    img[z_pix, x_pix] = 255  # Mark pixels corresponding to points
    dilation_kernel = disk(2)
    img = dilation(img, dilation_kernel)

    return img

def save_parquet(examples, path):
    rows = []
    for example in tqdm(examples):
        row = {
            "label": example["label"],
            "contour_image": example["contour_image"].reshape(-1) / 255.0,
            "filled_image": example["filled_image"].reshape(-1) / 255.0,
            "contour": example["contour"].reshape(-1),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)
    print(f"Saved dataset to {path}")

if __name__ == "__main__":
    classes = ["bottle", "bowl", "cone", "cup"] # Classes to include
    img_size = 64 
    K = 4 # Number of "slices" to create for each example
    contour_length = 128
    data_dir = Path("original_data/modelnet40")
    output_path = Path("generated_data")
    output_file_train = output_path / "modelnet_train.parquet" 
    output_file_test = output_path / "modelnet_test.parquet" 
    output_file_metadata = output_path / "modelnet_metadata.json"

    examples = {"train": [], "test": []}
    class_to_id = {label: i for i, label in enumerate(sorted(classes))}
    N = 200_000 # Number of points to sample from mesh

    for category in data_dir.iterdir():
        if not category.is_dir() or category.name not in classes:
            continue
        for split in examples.keys():
            split_dir = category / split
            if not split_dir.is_dir():
                continue
            dir_files = sorted(list(split_dir.glob("*.off")))
            print(len(dir_files), split_dir)
            for filename in tqdm(dir_files):
                mesh = trimesh.load(filename)
                points, face_indices = trimesh.sample.sample_surface(mesh, N)
                slices = get_slices(points, K)
                x_min, x_max = points[:, 0].min(), points[:, 0].max()
                z_min, z_max = points[:, 2].min(), points[:, 2].max()
                try:
                    contours = []
                    contour_images = []
                    filled_images = []
                    for j in range(K):
                        slice = slices[j]
                        img = create_binary_image(slice, img_size, x_min, x_max, z_min, z_max)
                        contour = find_contour(img)

                        contour_image = np.zeros((img_size, img_size), dtype=np.uint8)
                        cv2.drawContours(contour_image, [contour], -1, 255, 1)

                        filled_image = np.zeros((img_size, img_size), dtype=np.uint8)
                        cv2.drawContours(filled_image, [contour], -1, 255, -1)

                        contour = resample_contour(contour, length=contour_length)
                        contour -= np.mean(contour, axis=0)  # Center the contour

                        contours.append(contour)
                        contour_images.append(contour_image)
                        filled_images.append(filled_image)

                    label = class_to_id[category.name]
                    examples[split].append({"contour": np.array(contours), "label": label, "filename": filename, "contour_image": np.array(contour_images), "filled_image": np.array(filled_images)})
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    train_examples = examples["train"]
    test_examples = examples["test"]

    output_path.mkdir(exist_ok=True)
    save_parquet(train_examples, output_file_train)
    save_parquet(test_examples, output_file_test)

    # Load train dataset and compute mean and std of images
    train_dataset = pq.read_table(output_file_train).to_pandas()
    train_contour_images = np.stack(train_dataset["contour_image"].values)
    mean_contour_image = train_contour_images.mean()
    std_contour_image = train_contour_images.std()

    train_filled_images = np.stack(train_dataset["filled_image"].values)
    mean_filled_image = train_filled_images.mean()
    std_filled_image = train_filled_images.std()

    metadata = {
        "label": "ModelNet Contours",
        "description": "ModelNet contour dataset with contour images",
        "image_width": img_size,
        "image_height": img_size, 
        "contour_length": contour_length,
        "mean_contour_image": mean_contour_image,
        "std_contour_image": std_contour_image,
        "mean_filled_image": mean_filled_image,
        "std_filled_image": std_filled_image,
    }

    # Save metadata to JSON file
    output_file_metadata.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file_metadata, "w") as f:
        json.dump(metadata, f)

    print(f"Saved metadata to {output_file_metadata}")

