import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from statistics import median
from mpl_toolkits.mplot3d import Axes3D

# Read data from the output.txt file
rssi_data = {}
with open("output", "r") as file:
    for line in file:
        if ':' in line:
            coord_part, values_part = line.split(":")
            coord = tuple(map(float, coord_part.strip("() ").split(",")))
            values = list(map(int, values_part.strip().split()))
            rssi_data[coord] = values

# Prepare 3D data lists
X, Y, Z_kernel, Z_median = [], [], [], []

# Compute KDE peak and median for each coordinate
for (x, y), rssi_values in rssi_data.items():
    kde = gaussian_kde(rssi_values)
    xs = np.linspace(min(rssi_values), max(rssi_values), 200)
    peak = xs[np.argmax(kde(xs))]
    med = median(rssi_values)

    X.append(x)
    Y.append(y)
    Z_kernel.append(peak)
    Z_median.append(med)

# Create 3D plots
fig = plt.figure(figsize=(14, 6))

# KDE Plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X, Y, Z_kernel, c=Z_kernel, cmap='viridis')
ax1.set_title("Most Likely RSSI (KDE Peak)")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("RSSI")

# Median Plot
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X, Y, Z_median, c=Z_median, cmap='plasma')
ax2.set_title("Median RSSI")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("RSSI")

plt.tight_layout()
plt.show()
