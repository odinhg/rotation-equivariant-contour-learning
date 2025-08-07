# Datasets

This directory contains all code needed to generate the contour datasets. Generated data is saved in the `generated_data` directory. When external datasets are needed, these should be put in the `original_data` directory. For the contour only (including contour images and filled images), metadata is saved to `json` files in the `generated_data` directory and includes image size, contour length and image statistics used for standardization of the images.

## Classification Datasets

### Fashion MNIST

The Fashion contour MNIST dataset is derived from the original Fashion MNIST dataset. The dataset is downloaded using `torchvision.datasets.FashionMNIST` and then processed to create contours. 

Run `fashion_mnist.py` to generate the dataset. 

### Rotated MNIST

The Rotated MNIST contour dataset is derived from the Rotated MNIST dataset. The dataset is available [here](https://www.kaggle.com/datasets/saiteja0101/rotated-mnist).

### ModelNet Contours

The ModelNet Contours dataset is derived from the ModelNet dataset. The original dataset is available [here](https://modelnet.cs.princeton.edu/).

Run `modelnet.py` to generate the dataset.

## Autoencoder Datasets

### PCST Cell Contours

The PCST Cell Contours dataset is derived from the PCST dataset. The original dataset is available [here](https://zenodo.org/records/7388245).

The full dataset is around 22 GB and also includes different textures (perlin noise) for the same cell shapes. To extract the PNG files needed, run `pcst_extract.sh`. Then put the extracted files in the `original_data/cell_segmentations` and run `pcst_cells.py` to generate contours for a subset of the dataset.

### Curvature Regression

To generate the contours for the node-level curvature regression task, run `curvature.py`.

