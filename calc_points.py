import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from statistics import median
from itertools import combinations
from helper import closest_pair, get_inters, path_loss_dist

# Step 1: Load RSSI data from file
rssi_data = {}
with open("output", "r") as file:
    for line in file:
        if ':' in line:
            coord_part, values_part = line.split(":")
            coord = tuple(map(float, coord_part.strip("() ").split(",")))
            values = list(map(int, values_part.strip().split()))
            rssi_data[coord] = values

# Step 2: Get reference Tx value from KDE at (0.5, 1.0)
tx_coord = (0.5, 1.0)
tx_values = rssi_data[tx_coord]
tx_kde = gaussian_kde(tx_values)
xs_tx = np.linspace(min(tx_values), max(tx_values), 200)
tx_value = xs_tx[np.argmax(tx_kde(xs_tx))]

# Step 3: Estimate distance from all coordinates
distances = {}
for coord, values in rssi_data.items():
    kde = gaussian_kde(values)
    xs = np.linspace(min(values), max(values), 200)
    rssi_kde = xs[np.argmax(kde(xs))]
    distances[coord] = path_loss_dist(rssi_kde, tx_value)

# Step 4: Find estimated positions from triplets
intersection_points = []

for triplet in combinations(rssi_data.keys(), 3):
    all_inters = []
    for a, b in combinations(triplet, 2):
        r0 = distances[a]
        r1 = distances[b]
        pts = get_inters(a, b, r0, r1)
        if pts:
            all_inters.extend(pts)
    if len(all_inters) >= 2:
        cp = closest_pair(all_inters)
        if cp:
            mid_x = (cp[0][0] + cp[1][0]) / 2
            mid_y = (cp[0][1] + cp[1][1]) / 2
            intersection_points.append((mid_x, mid_y))

# Step 5: KDE, Median, and Kalman-style smoothing
xs = [pt[0] for pt in intersection_points]
ys = [pt[1] for pt in intersection_points]

# KDE Mode (most dense point)
xy = np.vstack([xs, ys])
kde = gaussian_kde(xy)
density = kde(xy)
kde_idx = np.argmax(density)
kde_point = (xs[kde_idx], ys[kde_idx])

# Median point
median_point = (median(xs), median(ys))

# Exponential moving average (Kalman-style smoothing)
alpha = 0.2
smoothed_x = xs[0]
smoothed_y = ys[0]
for i in range(1, len(xs)):
    smoothed_x = alpha * xs[i] + (1 - alpha) * smoothed_x
    smoothed_y = alpha * ys[i] + (1 - alpha) * smoothed_y
kalman_point = (smoothed_x, smoothed_y)

# Step 6: Print estimates
print(f"📍 KDE Estimated Point      : {kde_point}")
print(f"📍 Median Estimated Point   : {median_point}")
print(f"📍 Kalman-style Estimated Pt: {kalman_point}")

# Step 7: Plot heatmaps
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# KDE Heatmap
ax1.hist2d(xs, ys, bins=30, cmap='viridis')
ax1.scatter(*kde_point, color='red', label='KDE Peak', zorder=5)
ax1.set_title("KDE Heatmap")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.legend()

# Median Point
ax2.hist2d(xs, ys, bins=30, cmap='plasma')
ax2.scatter(*median_point, color='blue', label='Median', zorder=5)
ax2.set_title("Median Overlay")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.legend()

# Kalman-like EMA
kf_x, kf_y = zip(*intersection_points)
ax3.plot(kf_x, kf_y, alpha=0.3, label='Raw Intersections')
ax3.scatter(*kalman_point, color='green', label='Kalman-style', zorder=5)
ax3.set_title("Kalman-style Smoothing")
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.legend()

plt.tight_layout()
plt.show()
