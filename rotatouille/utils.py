import torch
import torch.nn as nn
import torch.nn.functional as F

def pad_contour(contours: torch.Tensor, size: int = 1) -> torch.Tensor:
    """
    Apply circular padding to complex-valued contour. Input tensor of shape (B, C, n). Output tensor of shape (B, C, n + 2 * size).
    """
    return F.pad(contours, pad=(size,) * 2, mode="circular")

def center_contour(contours: torch.Tensor) -> torch.Tensor:
    """
    Translate centroid of complex-valued contour to zero. Input tensor of shape (B, C, n). Output tensor of shape (B, C, n).
    """
    centroid = contours.mean(dim=-1, keepdim=True)
    return contours - centroid

