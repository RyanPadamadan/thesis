import pandas as pd
import itertools
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from ext import load_experiment_data, map_rssi_coords, path_loss_dist, trilateration, get_transmission_rssi, get_device_coordinates
import math

tx_power_curr = get_transmission_rssi("rssi_1m.csv")
target_mac = "04:99:bb:d8:6e:2e"
ls1 = []
ls2 = []
ls3 = []

def compute_intersections(mapped, tx_power):
    triples = list(itertools.combinations(mapped, 3))
    intersection_points = []

    for (x1, y1, z1, rssi1), (x2, y2, z2, rssi2), (x3, y3, z3, rssi3) in triples:
        if None in [rssi1, rssi2, rssi3]:
            continue
        r1 = path_loss_dist(rssi1, tx_power)
        r2 = path_loss_dist(rssi2, tx_power)
        r3 = path_loss_dist(rssi3, tx_power)
        point = trilateration((x1, y1, z1), r1, (x2, y2, z2), r2, (x3, y3, z3), r3)
        if point:
            intersection_points.append(point)

    return intersection_points

def localize_device_median(exp_dir_name, tx_power):
    coords_df, dev, rssi_df = load_experiment_data(exp_dir_name)
    mapped = map_rssi_coords(coords_df, rssi_df)
    intersection_points = compute_intersections(mapped, tx_power)
    if not intersection_points:
        return None

    points_array = np.array(intersection_points)
    estimated_position = np.median(points_array, axis=0)
    return tuple(estimated_position), get_device_coordinates(dev, target_mac)

def localize_device_kmeans(exp_dir_name, tx_power, n_clusters=1):
    coords_df, dev, rssi_df = load_experiment_data(exp_dir_name)
    mapped = map_rssi_coords(coords_df, rssi_df)
    intersection_points = compute_intersections(mapped, tx_power)
    if not intersection_points:
        return None

    points_array = np.array(intersection_points)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(points_array)
    estimated_position = kmeans.cluster_centers_[0]
    return tuple(estimated_position), get_device_coordinates(dev, target_mac)

def localize_device_dbscan(exp_dir_name, tx_power, eps=1.0, min_samples=3):
    coords_df, dev, rssi_df = load_experiment_data(exp_dir_name)
    mapped = map_rssi_coords(coords_df, rssi_df)
    intersection_points = compute_intersections(mapped, tx_power)
    if not intersection_points:
        return None

    points_array = np.array(intersection_points)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points_array)
    labels = db.labels_

    # Find the largest cluster
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(unique_labels) == 0:
        return None

    largest_cluster_label = unique_labels[np.argmax(counts)]
    largest_cluster_points = points_array[labels == largest_cluster_label]

    estimated_position = np.median(largest_cluster_points, axis=0)
    return tuple(estimated_position), get_device_coordinates(dev, target_mac)

def calculate_position_error(estimated, real):
    x_hat, y_hat, z_hat = estimated
    x, y, z = real
    error = math.sqrt((x_hat - x)**2 + (y_hat - y)**2 + (z_hat - z)**2)
    return error

def run_experiment(exp_name, localization_function, *args):
    print(f"Running {exp_name}...")
    estimated_position, actual = localization_function(*args)
    if estimated_position is None:
        print(f"{exp_name}: Localization failed.")
        return

    error = calculate_position_error(estimated_position, actual)
    print(f"Estimated Position: {estimated_position}")
    print(f"Error: {error:.4f} meters")

if __name__ == "__main__":
    for i in range(1, 6):
        experiment = f"exp_{i}"
        print(f"---Running Experiment {i}---")
        run_experiment("Median", localize_device_median, experiment, tx_power_curr,)
        run_experiment("KMeans", localize_device_kmeans, experiment, tx_power_curr,)
        run_experiment("DBSCAN", localize_device_dbscan, experiment, tx_power_curr,)
