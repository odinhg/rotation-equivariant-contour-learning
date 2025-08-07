import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import rotate

#def rotate_image_batch(batch: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
#    B = batch.shape[0]
#    degrees = thetas * 180 / np.pi
#    rotated_batch = torch.empty_like(batch)
#    for i in range(B):
#        rotated_batch[i] = torch.tensor(rotate(batch[i].squeeze().numpy(), degrees[i], reshape=False, mode="nearest", order=1))
#    return rotated_batch

def rotate_image_batch(batch: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
    B, C, H, W = batch.shape
    degrees = thetas * 180 / np.pi
    rotated_batch = torch.empty_like(batch)
    
    for i in range(B):
        for c in range(C):
            rotated_batch[i, c] = torch.tensor(
                rotate(batch[i, c].numpy(), degrees[i], reshape=False, mode="nearest", order=1)
            )
    
    return rotated_batch

def rotate_complex_batch(batch: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
    B = batch.shape[0]
    means = batch.mean(dim=-1, keepdim=True)
    batch = batch - means
    ws = torch.polar(torch.ones_like(thetas), thetas).view(-1, 1, 1)
    rotated_batch = batch * ws + means
    return rotated_batch

def rotate_batch(batch: torch.Tensor, seed) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    B = batch.shape[0]
    thetas = torch.rand(B, generator=rng) * 2 * np.pi 

    # 2D data (grayscale/binary image)
    if len(batch.shape) == 4:
        return rotate_image_batch(batch, thetas)

    # 1D data (complex contours)
    if len(batch.shape) == 3 and torch.is_complex(batch):
        return rotate_complex_batch(batch, thetas)

    raise ValueError(f"Unsupported batch shape: {batch.shape}. Expected 3D (complex) or 4D (image) batch tensor.")


