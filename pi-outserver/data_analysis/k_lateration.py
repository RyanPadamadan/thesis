import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from scipy.optimize import least_squares
import math
from ext import load_experiment_data, map_rssi_coords, path_loss_dist, get_transmission_rssi, get_device_coordinates

# Settings
tx_power_curr = get_transmission_rssi("rssi_1m.csv")
target_mac = "04:99:bb:d8:6e:2e"
exp_dir_name = "exp_1"

# Core k-lateration fitting function
def compute_k_lateration_estimate(mapped, tx_power, k):
    if len(mapped) < k:
        return None
    subset = mapped[:k]
    positions = []
    distances = []
    for x, y, z, rssi in subset:
        if rssi is None:
            continue
        positions.append((x, y, z))
        distances.append(path_loss_dist(rssi, tx_power))
    if len(positions) < 4:
        return None
    def residuals(guess, positions, distances):
        x, y, z = guess
        return [
            math.sqrt((x - xi)**2 + (y - yi)**2 + (z - zi)**2) - di
            for (xi, yi, zi), di in zip(positions, distances)
        ]
    initial_guess = np.mean(positions, axis=0)
    result = least_squares(residuals, initial_guess, args=(positions, distances))
    return tuple(result.x)

# Wrapper to collect all 3 estimates for a given k
def localize_all_methods(mapped, tx_power, k):
    points = []
    for _ in range(5):
        np.random.shuffle(mapped)
        estimate = compute_k_lateration_estimate(mapped, tx_power, k)
        if estimate:
            points.append(estimate)
    if not points:
        return None, None, None
    points_array = np.array(points)
    est_median = tuple(np.median(points_array, axis=0))
    kmeans = KMeans(n_clusters=1, n_init=10, random_state=0).fit(points_array)
    est_kmeans = tuple(kmeans.cluster_centers_[0])
    db = DBSCAN(eps=1.0, min_samples=2).fit(points_array)
    labels = db.labels_
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(unique_labels) == 0:
        est_dbscan = None
    else:
        largest_cluster = unique_labels[np.argmax(counts)]
        cluster_points = points_array[labels == largest_cluster]
        est_dbscan = tuple(np.median(cluster_points, axis=0))
    return est_median, est_kmeans, est_dbscan

def calculate_position_error(estimated, real):
    if estimated is None:
        return None
    x_hat, y_hat, z_hat = estimated
    x, y, z = real
    return math.sqrt((x_hat - x)**2 + (y_hat - y)**2 + (z_hat - z)**2)

# Load real experiment data
coords_df, dev, rssi_df = load_experiment_data(exp_dir_name)
mapped = map_rssi_coords(coords_df, rssi_df)
actual = get_device_coordinates(dev, target_mac)

# Run sweep
k_values = list(range(4, min(16, len(mapped)+1)))
errors_median = []
errors_kmeans = []
errors_dbscan = []

for k in k_values:
    est_median, est_kmeans, est_dbscan = localize_all_methods(mapped, tx_power_curr, k)
    errors_median.append(calculate_position_error(est_median, actual))
    errors_kmeans.append(calculate_position_error(est_kmeans, actual))
    errors_dbscan.append(calculate_position_error(est_dbscan, actual))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(k_values, errors_median, label="Median", marker="s")
plt.plot(k_values, errors_kmeans, label="KMeans", marker="o")
plt.plot(k_values, errors_dbscan, label="DBSCAN", marker="x")
plt.xlabel("Number of Reference Points (k)")
plt.ylabel("Localization Error (meters)")
plt.title("exp_1: Localization Error vs. Number of Reference Points (k)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("k_lateration_exp1.png")
plt.show()
