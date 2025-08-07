import cv2
import numpy as np

def pad_image(img: np.ndarray, pad_size: int) -> np.ndarray:
    pad_value = int(np.min(img))
    img = cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=pad_value)
    return img

def find_contour(img: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contour_areas = [cv2.contourArea(contour) for contour in contours]
    contour = contours[np.argsort(contour_areas)[-1]]
    contour = contour.squeeze(1)
    return contour

def resample_contour(contour: np.ndarray, length: int) -> np.ndarray:
    d = np.cumsum(np.r_[0, np.sqrt((np.diff(contour, axis=0) ** 2).sum(axis=1))])
    d_sampled = np.linspace(0, d.max(), length)
    contour_resampled = np.c_[
        np.interp(d_sampled, d, contour[:, 0]),
        np.interp(d_sampled, d, contour[:, 1]),
    ]
    return contour_resampled

def radial_histogram(image: np.ndarray, num_bins: int = 14) -> np.ndarray:
    h, w = image.shape
    cx, cy = (w - 1) / 2, (h - 1) / 2
    # Compute distances from center
    y, x = np.indices(image.shape)
    distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    # Compute distance from the center for each pixel
    y, x = np.indices(image.shape)
    distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    # Bin the distances
    bins = np.linspace(0, np.max(distances), num_bins + 1)
    #bins = np.geomspace(1, np.max(distances), num_bins + 1) - 1
    hist = np.zeros(num_bins, dtype=np.float32)

    for i in range(num_bins):
        mask = (distances >= bins[i]) & (distances < bins[i + 1])
        hist[i] = np.mean(image[mask]) if np.any(mask) else 0

    return hist / 255.0
