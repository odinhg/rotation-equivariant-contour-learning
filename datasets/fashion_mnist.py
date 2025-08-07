from torchvision.datasets import FashionMNIST
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json

from utils import pad_image, find_contour, resample_contour, radial_histogram

def binarize_image(img: np.ndarray) -> np.ndarray:
    _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    return img

def save_parquet(dataset, path, contour_length):
    rows = []
    for image, label in tqdm(dataset):
        image = np.array(image)
        image_binarized = binarize_image(image)
        contour = find_contour(image_binarized)
        contour_image = np.zeros_like(image_binarized)
        cv2.drawContours(contour_image, [contour], -1, 255, 1)
        filled_image = np.zeros_like(image_binarized)
        cv2.drawContours(filled_image, [contour], -1, 255, -1)
        contour = resample_contour(contour, contour_length)

        row = {
            "label": label,
            "contour_image": contour_image.reshape(-1) / 255.0,
            "filled_image": filled_image.reshape(-1) / 255.0,
            "contour": contour.reshape(-1),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)
    print(f"Saved dataset to {path}")

if __name__ == "__main__":
    # Load Fashion MNIST dataset
    train_dataset = FashionMNIST(root="data", train=True, download=True)
    test_dataset = FashionMNIST(root="data", train=False, download=True)

    # Parameters
    contour_length = 128
    output_path = Path("generated_data")
    output_file_train = output_path / "fashion_mnist_train.parquet" 
    output_file_test = output_path / "fashion_mnist_test.parquet" 
    output_file_metadata = output_path / "fashion_mnist_metadata.json"

    output_path.mkdir(exist_ok=True)
    save_parquet(train_dataset, output_file_train, contour_length=contour_length)
    save_parquet(test_dataset, output_file_test, contour_length=contour_length)

    # Load train dataset and compute mean and std of images
    train_dataset = pq.read_table(output_file_train).to_pandas()
    train_contour_images = np.stack(train_dataset["contour_image"].values)
    mean_contour_image = train_contour_images.mean()
    std_contour_image = train_contour_images.std()

    train_filled_images = np.stack(train_dataset["filled_image"].values)
    mean_filled_image = train_filled_images.mean()
    std_filled_image = train_filled_images.std()

    metadata = {
        "label": "Fashion MNIST",
        "description": "Fashion MNIST dataset with contour images",
        "image_width": 28,
        "image_height": 28, 
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


