import pandas as pd
import itertools
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from ext import load_experiment_data, map_rssi_coords, path_loss_dist, trilateration, get_transmission_rssi, get_device_coordinates
import math

tx_power_curr = get_transmission_rssi("rssi_1m.csv")
target_mac = "04:99:bb:d8:6e:2e"

# Persistent references for each method
ref_median = None
ref_kmeans = None
ref_dbscan = None

accepted_median = []
accepted_kmeans = []
accepted_dbscan = []

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

def refine_with_method(method, points, ref_point, accepted_points, n_clusters=1, eps=1.0, min_samples=3, threshold=2.5):
    if ref_point is None:
        # First-time initialization
        if method == "median":
            ref_point = np.median(np.array(points), axis=0)
        elif method == "kmeans":
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(points)
            ref_point = kmeans.cluster_centers_[0]
        elif method == "dbscan":
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
            labels = db.labels_
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            if len(unique_labels) > 0:
                label = unique_labels[np.argmax(counts)]
                cluster = points[labels == label]
                ref_point = np.median(cluster, axis=0)
            else:
                ref_point = np.median(points, axis=0)
        accepted_points.extend(points)
    else:
        for p in points:
            dist = np.linalg.norm(np.array(p) - np.array(ref_point))
            if dist <= threshold:
                accepted_points.append(p)

        # Weighted refinement
        accepted_array = np.array(accepted_points)
        distances = np.linalg.norm(accepted_array - ref_point, axis=1)
        weights = 1 / (distances + 1e-6)
        ref_point = np.average(accepted_array, axis=0, weights=weights)

    return ref_point, accepted_points

def localize_device_incremental(exp_dir_name, tx_power):
    global ref_median, ref_kmeans, ref_dbscan
    global accepted_median, accepted_kmeans, accepted_dbscan

    coords_df, dev, rssi_df = load_experiment_data(exp_dir_name)
    mapped = map_rssi_coords(coords_df, rssi_df)
    intersection_points = compute_intersections(mapped, tx_power)
    if not intersection_points:
        return None

    points_array = np.array(intersection_points)

    ref_median, accepted_median = refine_with_method("median", points_array, ref_median, accepted_median)
    ref_kmeans, accepted_kmeans = refine_with_method("kmeans", points_array, ref_kmeans, accepted_kmeans)
    ref_dbscan, accepted_dbscan = refine_with_method("dbscan", points_array, ref_dbscan, accepted_dbscan)

    actual = get_device_coordinates(dev, target_mac)

    return {
        "Median": (tuple(ref_median), actual),
        "KMeans": (tuple(ref_kmeans), actual),
        "DBSCAN": (tuple(ref_dbscan), actual)
    }

def calculate_position_error(estimated, real):
    x_hat, y_hat, z_hat = estimated
    x, y, z = real
    error = math.sqrt((x_hat - x)**2 + (y_hat - y)**2 + (z_hat - z)**2)
    return error

def run_experiment(exp_id, localization_function, *args):
    print(f"--- Running Experiment {exp_id} ---")
    results = localization_function(*args)
    if results is None:
        print(f"Experiment {exp_id}: Localization failed.")
        return

    for method, (estimated_position, actual_position) in results.items():
        error = calculate_position_error(estimated_position, actual_position)
        print(f"{method} → Estimated: {estimated_position}, Error: {error:.4f} meters")

if __name__ == "__main__":
    for i in range(1, 5):
        experiment = f"exp_{i}"
        run_experiment(i, localize_device_incremental, experiment, tx_power_curr)
