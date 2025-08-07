import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

def contour_to_tensor(contour: torch.Tensor, rescale: bool=True) -> torch.Tensor:
    """
    Real-valued contour to complex tensor. Input shape (C, n, 2), output shape (C, n).
    """
    contour = torch.tensor(contour, dtype=torch.float32)
    contour = torch.view_as_complex(contour)

    if rescale:
        contour = contour / torch.std(contour.abs())

    return contour

def image_to_tensor(image: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Image to tensor with standardization. 
    """
    image = torch.tensor(image, dtype=torch.float32)
    image = (image - mean) / std
    if len(image.shape) == 2: # Add channel dimension if missing
        image = image.unsqueeze(0)

    return image

class ContourDataset(Dataset):
    def __init__(self, data_file: str | Path, length: int=128, channels: int=1, use_extra_features: bool=False) -> None:
        self.data_file = Path(data_file)
        if not self.data_file.is_file():
            raise FileNotFoundError(f"File not found: {self.data_file}")

        self.dataset = pq.read_table(self.data_file) 

        self.contours = self.dataset["contour"]
        self.labels = self.dataset["label"]
        self.use_extra_features = use_extra_features
        if use_extra_features:
            self.extra_features = self.dataset["extra_features"]
        self.length = length
        self.channels = channels

        self.unique_labels, self.unique_counts = np.unique(self.labels, return_counts=True)
        self.n_classes = len(self.unique_labels)

        print(f"Loaded {len(self.dataset)} examples of {self.n_classes} classes")
        for label, count in zip(self.unique_labels, self.unique_counts):
            print(f"\tLabel {label}: {count} examples")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        contour = np.array(self.contours[idx].as_py()).reshape(self.channels, self.length, 2)
        contour = contour_to_tensor(contour).view(self.channels, self.length)
        label = self.labels[idx].as_py()

        if self.use_extra_features:
            extra_features = self.extra_features[idx].as_py()
            extra_features = torch.tensor(extra_features, dtype=torch.float32)
            return {"data": contour, "label": label, "extra_features": extra_features}

        return {"data": contour, "label": label}

class ImageDataset(Dataset):
    def __init__(self, data_file: str | Path, image_type:str="contour_image", height: int=28, width: int=28, mean: float=0.0, std: float=1.0, channels: int=1) -> None:
        self.data_file = Path(data_file)
        if not self.data_file.is_file():
            raise FileNotFoundError(f"File not found: {self.data_file}")

        assert image_type in ["contour_image", "filled_image"], f"Invalid image type: {image_type}"

        self.dataset = pq.read_table(self.data_file) 

        self.images = self.dataset[image_type]
        self.labels = self.dataset["label"]
        self.height = height
        self.width = width
        self.mean = mean
        self.std = std
        self.channels = channels

        self.unique_labels, self.unique_counts = np.unique(self.labels, return_counts=True)
        self.n_classes = len(self.unique_labels)

        print(f"Loaded {len(self.dataset)} examples of {self.n_classes} classes")
        for label, count in zip(self.unique_labels, self.unique_counts):
            print(f"\tLabel {label}: {count} examples")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = np.array(self.images[idx].as_py()).reshape(self.channels, self.height, self.width)
        image = image_to_tensor(image, mean=self.mean, std=self.std)
        label = self.labels[idx].as_py()

        return {"data": image, "label": label}


class ContourDatasetAutoencoder(Dataset):
    def __init__(self, data_file: str | Path, length: int=128, channels: int=1) -> None:
        self.data_file = Path(data_file)
        if not self.data_file.is_file():
            raise FileNotFoundError(f"File not found: {self.data_file}")

        self.dataset = pq.read_table(self.data_file) 
        self.contours = self.dataset["contour"]
        self.ecc_values = self.dataset["ecc"]
        self.rand_values = self.dataset["rand"]
        self.length = length
        self.channels = channels

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        contour = np.array(self.contours[idx].as_py()).reshape(self.channels, self.length, 2)
        contour = contour_to_tensor(contour).view(self.channels, self.length)
        ecc = self.ecc_values[idx].as_py()
        rand = self.rand_values[idx].as_py()
        return {"data": contour, "ecc": ecc, "rand": rand}

class ImageDatasetAutoencoder(Dataset):
    def __init__(self, data_file: str | Path, height: int=64, width: int=64, mean: float=0.0, std: float=1.0, channels: int=1) -> None:
        self.data_file = Path(data_file)
        if not self.data_file.is_file():
            raise FileNotFoundError(f"File not found: {self.data_file}")

        self.dataset = pq.read_table(self.data_file)
        self.images = self.dataset["filled_image"]
        self.ecc_values = self.dataset["ecc"]
        self.rand_values = self.dataset["rand"]
        self.height = height
        self.width = width
        self.mean = mean
        self.std = std
        self.channels = channels

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = np.array(self.images[idx].as_py()).reshape(self.channels, self.height, self.width) 
        image = image_to_tensor(image, mean=self.mean, std=self.std)
        ecc = self.ecc_values[idx].as_py()
        rand = self.rand_values[idx].as_py()
        return {"data": image, "ecc": ecc, "rand": rand}

class ContourDatasetRegression(Dataset):
    def __init__(self, data_file: str | Path, length: int=128, channels: int=1) -> None:
        self.data_file = Path(data_file)
        if not self.data_file.is_file():
            raise FileNotFoundError(f"File not found: {self.data_file}")

        self.dataset = pq.read_table(self.data_file) 
        self.contours = self.dataset["contour"]
        self.curvatures = self.dataset["curvature"]
        self.length = length
        self.channels = channels

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        contour = np.array(self.contours[idx].as_py()).reshape(self.channels, self.length, 2)
        contour = contour_to_tensor(contour).view(self.channels, self.length)

        curvature = np.array(self.curvatures[idx].as_py()).reshape(1, self.length)
        curvature = torch.tensor(curvature, dtype=torch.float32)

        return {"data": contour, "curvature": curvature}

