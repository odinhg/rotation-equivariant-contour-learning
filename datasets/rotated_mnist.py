import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from utils import pad_image, find_contour, resample_contour, radial_histogram

def binarize_image(img):
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    binary_image = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return binary_image

def save_parquet(dataset, path, contour_length):
    rows = []
    for image, label in tqdm(dataset):
        image = cv2.convertScaleAbs(image, alpha=255)
        image_binarized = binarize_image(image)
        contour = find_contour(image_binarized)
        contour = resample_contour(contour, contour_length)

        row = {
            "label": label,
            "contour": contour.reshape(-1),
            "extra_features": radial_histogram(image, num_bins=14),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)
    print(f"Saved dataset to {path}")

if __name__ == "__main__":
    # Load Rotated MNIST dataset
    data_train = np.loadtxt("original_data/mnist_all_rotation_normalized_float_train_valid.amat")
    data_test = np.loadtxt("original_data/mnist_all_rotation_normalized_float_test.amat")
    X_train = data_train[:, :-1].reshape(-1, 28, 28)
    y_train = data_train[:, -1].astype(int)
    X_test = data_test[:, :-1].reshape(-1, 28, 28)
    y_test = data_test[:, -1].astype(int)

    train_dataset = list(zip(X_train, y_train))
    test_dataset = list(zip(X_test, y_test))

    # Parameters
    contour_length = 128
    output_path = Path("generated_data")
    output_file_train = output_path / "rotated_mnist_train.parquet"
    output_file_test = output_path / "rotated_mnist_test.parquet" 

    output_path.mkdir(exist_ok=True)
    save_parquet(train_dataset, output_file_train, contour_length=contour_length)
    save_parquet(test_dataset, output_file_test, contour_length=contour_length)

