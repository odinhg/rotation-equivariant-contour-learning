import numpy as np
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd


# visualize contours
def visualize_contours(dataset, num_samples=10):
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        image = dataset.iloc[i]['contour'].reshape(128, 2)
        label = dataset.iloc[i]['label']
        plt.subplot(2, num_samples // 2, i + 1)
        plt.scatter(image[:, 0], image[:, 1], s=10, c='blue', alpha=0.5) 
        plt.title(f'Label: {label}')
        plt.axis('equal')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_filled_images(dataset, num_samples=10):
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        image = dataset.iloc[i]['filled_image'].reshape(28, 28)
        label = dataset.iloc[i]['label']
        plt.subplot(2, num_samples // 2, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'Label: {label}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_contour_images(dataset, num_samples=10):
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        image = dataset.iloc[i]['contour_image'].reshape(28, 28)
        label = dataset.iloc[i]['label']
        plt.subplot(2, num_samples // 2, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'Label: {label}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Rotated MNIST dataset
train_dataset = pq.read_table("generated_data/rotated_mnist_train.parquet").to_pandas()
print("Visualizing Rotated MNIST contours...")
visualize_contours(train_dataset, num_samples=20)

# Fashion MNIST dataset
train_dataset = pq.read_table("generated_data/fashion_mnist_train.parquet").to_pandas()
print("Visualizing Fashion MNIST contours...")
visualize_contours(train_dataset, num_samples=20)
print("Visualizing Fashion MNIST contour images...")
visualize_contour_images(train_dataset, num_samples=20)
print("Visualizing Fashion MNIST filled images...")
visualize_filled_images(train_dataset, num_samples=20)

