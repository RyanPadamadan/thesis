import pandas as pd
import itertools
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from ext import load_experiment_data, map_rssi_coords, path_loss_dist, trilateration, get_transmission_rssi, get_device_coordinates
import matplotlib.pyplot as plt
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


def plot_localization_errors(errors_dict):
    plt.figure(figsize=(10, 6))
    x = list(range(1, len(next(iter(errors_dict.values()))) + 1))

    for label, errors in errors_dict.items():
        plt.plot(x, errors, label=label, marker='o')

    plt.xlabel("Experiment Number")
    plt.ylabel("Localization Error (meters)")
    plt.title("Localization Error Comparison Across Experiments")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("m1")
    plt.show()


if __name__ == "__main__":
    median_errors = []
    kmeans_errors = []
    dbscan_errors = []

    for i in range(1, 8):
        experiment = f"exp_{i}"
        print(f"\n--- Running Experiment {i} ---")

        for method_name, func, store in [
            ("Median", localize_device_median, median_errors),
            ("KMeans", localize_device_kmeans, kmeans_errors),
            ("DBSCAN", localize_device_dbscan, dbscan_errors),
        ]:
            print(f"Running {method_name}...")
            result = func(experiment, tx_power_curr)
            if result is None:
                print(f"{method_name}: Localization failed.")
                store.append(None)
            else:
                estimated, actual = result
                error = calculate_position_error(estimated, actual)
                print(f"Estimated Position: {estimated}")
                print(f"Error: {error:.4f} meters")
                store.append(error)

    # Filter out experiments where any method failed
    all_errors = {
        "Median": [e for e in median_errors if e is not None],
        "KMeans": [e for e in kmeans_errors if e is not None],
        "DBSCAN": [e for e in dbscan_errors if e is not None],
    }

    plot_localization_errors(all_errors)
