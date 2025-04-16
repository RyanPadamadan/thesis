import pandas as pd
import itertools
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from extraction import load_experiment_data, map_rssi_coords, path_loss_dist, trilateration, get_transmission_rssi
tx_power_curr =  get_transmission_rssi("rssi_1m.csv")

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
    coords_df, _, rssi_df = load_experiment_data(exp_dir_name)
    mapped = map_rssi_coords(coords_df, rssi_df)
    print(mapped)
    intersection_points = compute_intersections(mapped, tx_power)
    print(intersection_points)
    if not intersection_points:
        return None

    points_array = np.array(intersection_points)
    estimated_position = np.median(points_array, axis=0)
    return tuple(estimated_position)

if __name__ == "__main__":
    print("EXP 1")
    print("Median: ", localize_device_median("exp_1", tx_power_curr))