from scipy.optimize import least_squares
import numpy as np
from scipy.stats import gaussian_kde
from helper import path_loss_dist

# Step 1: Read RSSI data from file
rssi_data = {}
with open("output", "r") as file:
    for line in file:
        if ':' in line:
            coord_part, values_part = line.split(":")
            coord = tuple(map(float, coord_part.strip("() ").split(",")))
            values = list(map(int, values_part.strip().split()))
            rssi_data[coord] = values

# Step 2: Get reference Tx value using KDE at (0.5, 1.0)
tx_coord = (0.5, 1.0)
tx_values = rssi_data[tx_coord]
tx_kde = gaussian_kde(tx_values)
xs_tx = np.linspace(min(tx_values), max(tx_values), 200)
tx_value = xs_tx[np.argmax(tx_kde(xs_tx))]

# Step 3: Calculate most likely RSSI (KDE) and corresponding distance
most_likely_rssi = {}
distances = {}

for coord, values in rssi_data.items():
    kde = gaussian_kde(values)
    xs = np.linspace(min(values), max(values), 200)
    peak_rssi = xs[np.argmax(kde(xs))]
    most_likely_rssi[coord] = peak_rssi
    distances[coord] = path_loss_dist(peak_rssi, tx_value)

# Step 4: Use least squares to compute best estimate from all anchors
def residuals(pos, anchors, dists):
    return [np.linalg.norm(np.array(pos) - np.array(anchor)) - d for anchor, d in zip(anchors, dists)]

anchors = list(distances.keys())
dists = [distances[coord] for coord in anchors]
initial_guess = np.mean(anchors, axis=0)

result = least_squares(residuals, initial_guess, args=(anchors, dists))
estimated_position = result.x

print(estimated_position)
