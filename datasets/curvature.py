import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.interpolate import interp1d
from tqdm import tqdm
from sklearn.metrics import r2_score 

def generate_random_closed_curve(n_points=500, n_modes=5, seed=None):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=True)

    def random_coeffs():
        return rng.uniform(-1, 1, n_modes)
    
    a_x, b_x = random_coeffs(), random_coeffs()
    a_y, b_y = random_coeffs(), random_coeffs()

    x = np.zeros_like(t)
    y = np.zeros_like(t)

    for k in range(1, n_modes + 1):
        x += a_x[k - 1] * np.cos(k * t) + b_x[k - 1] * np.sin(k * t)
        y += a_y[k - 1] * np.cos(k * t) + b_y[k - 1] * np.sin(k * t)

    return np.stack([x, y], axis=1), t, a_x, b_x, a_y, b_y

def compute_analytic_curvature(t, a_x, b_x, a_y, b_y):
    """
    Compute the curvature analytically from Fourier coefficients at given t values.
    """
    n_modes = len(a_x)
    dx = np.zeros_like(t)
    dy = np.zeros_like(t)
    ddx = np.zeros_like(t)
    ddy = np.zeros_like(t)

    for k in range(1, n_modes + 1):
        dx  += -k * a_x[k - 1] * np.sin(k * t) + k * b_x[k - 1] * np.cos(k * t)
        dy  += -k * a_y[k - 1] * np.sin(k * t) + k * b_y[k - 1] * np.cos(k * t)
        ddx += -k**2 * a_x[k - 1] * np.cos(k * t) - k**2 * b_x[k - 1] * np.sin(k * t)
        ddy += -k**2 * a_y[k - 1] * np.cos(k * t) - k**2 * b_y[k - 1] * np.sin(k * t)

    numerator = dx * ddy - dy * ddx
    denominator = (dx**2 + dy**2) ** 1.5
    curvature = np.abs(numerator / (denominator + 1e-8))
    return curvature

def plot_continuous_curvature(points, curvature, cmap="plasma", ax=None):
    """
    Plot the curve with color gradient based on continuous curvature.
    """
    x, y = points[:, 0], points[:, 1]

    if ax is None:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()

    scatter = ax.scatter(x, y, c=curvature, cmap=cmap, s=20, alpha=1, edgecolor="none", vmin=0, vmax=curvature.max())
    cbar = plt.colorbar(scatter, ax=ax, orientation="vertical")
    cbar.set_label("Normalized Curvature", rotation=270, labelpad=15)
    ax.axis("equal")
    ax.set_title("Smooth Closed Curve Colored by Continuous Curvature")


def resample_curve_and_curvature(points, curvature, n_points=500):
    """
    Resample the curve to be equidistant in arc length and interpolate curvature.
    Parameters:
        points: (N, 2) array of curve points (x, y)
        curvature: (N,) array of curvature values
        n_points: number of resampled points
    Returns:
        resampled_points: (n_points, 2)
        resampled_curvature: (n_points,)
    """
    # Step 1: Compute arc length
    deltas = np.diff(points, axis=0, append=points[:1])
    segment_lengths = np.linalg.norm(deltas, axis=1)
    arc_length = np.cumsum(segment_lengths)
    arc_length = np.insert(arc_length, 0, 0.0)[:-1]  # start at 0

    total_length = arc_length[-1] + segment_lengths[-1]  # total closed loop

    # Step 2: Uniform sampling along arc length
    new_arc = np.linspace(0, total_length, n_points, endpoint=False)

    # Step 3: Interpolate x, y, and curvature over arc length
    interp_x = interp1d(arc_length, points[:, 0], kind="linear", fill_value="extrapolate", assume_sorted=True)
    interp_y = interp1d(arc_length, points[:, 1], kind="linear", fill_value="extrapolate", assume_sorted=True)
    interp_curv = interp1d(arc_length, curvature, kind="linear", fill_value="extrapolate", assume_sorted=True)

    resampled_x = interp_x(new_arc)
    resampled_y = interp_y(new_arc)
    resampled_curvature = interp_curv(new_arc)

    resampled_points = np.stack([resampled_x, resampled_y], axis=1)
    return resampled_points, resampled_curvature

# Configuration
n_contours = 3000  # Total number of contours to generate
test_size = 1000  # Number of contours for testing
contour_length = 100 # Number of points in each contour
curvature_cutoff = 1000.0  # Maximum curvature threshold for filtering
min_modes = 2
max_modes = 5
dataset = []
seed = 123
np.random.seed(seed)

while len(dataset) < n_contours:
    n_modes = np.random.randint(min_modes, max_modes + 1)
    curve, t, a_x, b_x, a_y, b_y = generate_random_closed_curve(n_points=contour_length, n_modes=n_modes, seed=seed)
    curvature = compute_analytic_curvature(t, a_x, b_x, a_y, b_y)
    
    max_curvature = curvature.max()
    if max_curvature > curvature_cutoff:
        continue

    curve, curvature = resample_curve_and_curvature(curve, curvature, n_points=contour_length)
    curve -= curve.mean(axis=0)

    dataset.append({
        "curve": curve,
        "curvature": curvature,
        "seed": seed,
        "n_modes": n_modes,
    })

    seed += 1

# Split into training and test sets
train_size = n_contours - test_size
train_dataset = dataset[:train_size]
test_dataset = dataset[train_size:]

print(f"Generated {len(dataset)} contours with {train_size} for training and {test_size} for testing.")

# Approximate curvature using finite differences

def approximate_curvature_closed(points):
    """
    Approximate the curvature of a closed planar curve using periodic finite differences (with np.roll).
    
    Parameters:
        points: (N, 2) array of curve points (x, y), assumed to form a closed loop.
    
    Returns:
        curvature: (N,) array of curvature values
    """
    # Shifted points using np.roll for periodicity
    p_prev = np.roll(points, shift=1, axis=0)
    p_next = np.roll(points, shift=-1, axis=0)

    # First derivatives (central differences)
    dx = (p_next[:, 0] - p_prev[:, 0]) / 2
    dy = (p_next[:, 1] - p_prev[:, 1]) / 2

    # Second derivatives (central second difference)
    ddx = p_next[:, 0] - 2 * points[:, 0] + p_prev[:, 0]
    ddy = p_next[:, 1] - 2 * points[:, 1] + p_prev[:, 1]

    # Curvature formula
    numerator = dx * ddy - dy * ddx
    denominator = (dx**2 + dy**2) ** 1.5 + 1e-8
    curvature = np.abs(numerator / denominator)

    return curvature

def circumcircle_curvature(points):
    """
    Approximate curvature via the circumcircle of three consecutive points.
    """
    p_prev = np.roll(points, 1, axis=0)
    p_curr = points
    p_next = np.roll(points, -1, axis=0)
    
    a = np.linalg.norm(p_curr - p_prev, axis=1)
    b = np.linalg.norm(p_next - p_curr, axis=1)
    c = np.linalg.norm(p_next - p_prev, axis=1)

    # Semi-perimeter
    s = (a + b + c) / 2

    # Area using Heron's formula
    area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0))

    # Circumradius R = (abc) / (4 * area)
    with np.errstate(divide='ignore', invalid='ignore'):
        R = (a * b * c) / (4.0 * area + 1e-8)
        curvature = 1.0 / (R + 1e-8)

    # Remove invalid values (collinear points -> infinite radius -> zero curvature)
    curvature[np.isnan(curvature) | np.isinf(curvature)] = 0.0
    return curvature


target_curvatures = np.array([data["curvature"] for data in test_dataset])

# Compute mean absolute error between approximated curvature and analytic curvature over the dataset
finite_difference_curvatures = np.array([approximate_curvature_closed(data["curve"]) for data in test_dataset])
finite_difference_mae = np.mean(np.abs(target_curvatures - finite_difference_curvatures))
finite_difference_r2 = r2_score(target_curvatures.flatten(), finite_difference_curvatures.flatten())

print(f"Finite Difference Curvature MAE: {finite_difference_mae:.4f}, R²: {finite_difference_r2:.4f}")

# Compute mean absolute error between circumcircle curvature and analytic curvature
circumcircle_curvatures = np.array([circumcircle_curvature(data["curve"]) for data in test_dataset])
circumcircle_mae = np.mean(np.abs(target_curvatures - circumcircle_curvatures))
circumcircle_r2 = r2_score(target_curvatures.flatten(), circumcircle_curvatures.flatten())
print(f"Circumcircle Curvature MAE: {circumcircle_mae:.4f}, R²: {circumcircle_r2:.4f}")

# Save datasets to file
def save_parquet(examples, path):
    rows = []
    for example in tqdm(examples):
        row = {
            "contour": example["curve"].reshape(-1),
            "curvature": example["curvature"],
            "seed": example["seed"],
            "n_modes": example["n_modes"],
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)
    print(f"Saved {len(rows)} examples to {path}")

# Save the datasets to Parquet files
save_parquet(train_dataset, "generated_data/train_curvature_contours.parquet")
save_parquet(test_dataset, "generated_data/test_curvature_contours.parquet")
