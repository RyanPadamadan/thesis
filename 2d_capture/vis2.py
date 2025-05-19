import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from statistics import median
from collections import defaultdict

# Read RSSI data from the 'output' file
rssi_data = {}
with open("output", "r") as file:
    for line in file:
        if ':' in line:
            coord_part, values_part = line.split(":")
            coord = tuple(map(float, coord_part.strip("() ").split(",")))
            values = list(map(int, values_part.strip().split()))
            rssi_data[coord] = values

# Prepare data
X, Y, Z_kernel, Z_median = [], [], [], []
heatmap_kernel = defaultdict(dict)
heatmap_median = defaultdict(dict)

for (x, y), rssi_values in rssi_data.items():
    kde = gaussian_kde(rssi_values)
    xs = np.linspace(min(rssi_values), max(rssi_values), 200)
    peak = xs[np.argmax(kde(xs))]
    med = median(rssi_values)

    X.append(x)
    Y.append(y)
    Z_kernel.append(peak)
    Z_median.append(med)

    heatmap_kernel[y][x] = peak
    heatmap_median[y][x] = med

# Unique sorted coordinates for heatmap grid
x_unique = sorted(set(X))
y_unique = sorted(set(Y))

def create_grid(heatmap_dict):
    grid = np.zeros((len(y_unique), len(x_unique)))
    for i, y in enumerate(y_unique):
        for j, x in enumerate(x_unique):
            grid[i, j] = heatmap_dict[y].get(x, np.nan)
    return grid

kernel_grid = create_grid(heatmap_kernel)
median_grid = create_grid(heatmap_median)

# Plot KDE Peak and Median Heatmaps
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# KDE Heatmap
im1 = ax1.imshow(kernel_grid, cmap='viridis', origin='lower', extent=[min(x_unique), max(x_unique), min(y_unique), max(y_unique)])
ax1.set_title("Heatmap: Most Likely RSSI (KDE Peak)")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.plot(0, 0.56, marker='o', color='red', markersize=8, label='Device Location')
ax1.text(0, 0.56 + 0.02, "Device Location", color='blue', ha='center')
plt.colorbar(im1, ax=ax1, shrink=0.8)

# Median Heatmap
im2 = ax2.imshow(median_grid, cmap='plasma', origin='lower', extent=[min(x_unique), max(x_unique), min(y_unique), max(y_unique)])
ax2.set_title("Heatmap: Median RSSI")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.plot(0, 0.56, marker='o', color='blue', markersize=8, label='Device Location')
ax2.text(0.3, 0.56 + 0.06, "Device Location", color='blue', ha='center')
plt.colorbar(im2, ax=ax2, shrink=0.8)

plt.tight_layout()
plt.show()
