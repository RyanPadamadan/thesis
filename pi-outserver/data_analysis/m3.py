import itertools
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from m1 import plot_localization_errors
from ext import load_experiment_data_with_mesh, map_rssi_coords, path_loss_dist, trilateration, get_transmission_rssi, get_device_coordinates

VOXEL_SIZE = 0.2
tx_power_curr = get_transmission_rssi("rssi_1m.csv")
target_mac = "04:99:bb:d8:6e:2e"



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

def generate_voxel_set(df, voxel_size=VOXEL_SIZE):
    voxel_set = set()
    for _, row in df.iterrows():
        voxel = (
            int(math.floor(row["x"] / voxel_size)),
            int(math.floor(row["y"] / voxel_size)),
            int(math.floor(row["z"] / voxel_size))
        )
        voxel_set.add(voxel)

    return voxel_set


def distance_3d(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2 +
        (p1[2] - p2[2]) ** 2
    )

def set_weights(voxel_set, points):
    weights = {}
    for pt in points:
        nearest_dist = float("inf")
        for voxel in voxel_set:
            nearest_dist = min(distance_3d(pt, voxel), nearest_dist)
        pt_tp = (pt[0], pt[1], pt[2])
        if nearest_dist > 3 * VOXEL_SIZE:
            weights[pt_tp] = 0
        elif nearest_dist > 2 * VOXEL_SIZE:
            weights[pt_tp] = 1
        elif nearest_dist > 1 * VOXEL_SIZE:
            weights[pt_tp] = 2
        else:
            weights[pt_tp] = 3
    return weights
def plot_intersections_and_voxels(intersection_points, voxel_set, name, voxel_size=0.1):
    intersection_points = np.array(intersection_points)
    voxel_centers = np.array([
        [ix * voxel_size + voxel_size / 2,
         iy * voxel_size + voxel_size / 2,
         iz * voxel_size + voxel_size / 2]
        for (ix, iy, iz) in voxel_set
    ])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot intersections
    ax.scatter(intersection_points[:, 0], intersection_points[:, 1], intersection_points[:, 2],
               c='red', label='Intersection Points', s=20)

    # Plot voxel centers
    ax.scatter(voxel_centers[:, 0], voxel_centers[:, 1], voxel_centers[:, 2],
               c='blue', label='Voxel Centers', s=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Intersection Points and Voxel Grid')
    ax.legend()
    plt.tight_layout()
    plt.savefig(name)
    plt.show()

def filter_intersections(intersection_pts, voxel_set):
    """
    Filters intersection points based on whether they fall within known voxels.

    Parameters:
    - intersection_pts: List of 3D coordinates (tuples or np arrays).
    - voxel_set: A set of voxel coordinates (e.g., {(x, y, z), ...}).
    - VOXEL_SIZE: Assumed to be defined globally.

    Returns:
    - A list of intersection points that are inside the voxel set.
    """
    filtered = []
    for pt in intersection_pts:
        # Convert point to voxel index based on VOXEL_SIZE
        voxel = tuple((np.array(pt) // VOXEL_SIZE).astype(int))
        if voxel in voxel_set:
            filtered.append(pt)
    return filtered

errors = []

if __name__ == "__main__":
    for i in range(1, 8):
        exp_name = f"exp_{i}"
        tx_power = get_transmission_rssi("rssi_1m.csv") 

        data = load_experiment_data_with_mesh(exp_name)
        coordinates_df = data["coordinates"]
        rssi_df = data["rssi"]
        meshpoints_df = data["meshpoints"]
        devices = data["devices"]

        # Map RSSI values to spatial coordinates
        mapped = map_rssi_coords(coordinates_df, rssi_df)

        # Compute trilateration intersections
        intersection_points = compute_intersections(mapped, tx_power)

        print(f"Loaded {len(meshpoints_df)} meshpoints from scene {i}.")
        print(f"Found {len(intersection_points)} intersection points from {exp_name}")

        st = generate_voxel_set(meshpoints_df)
        out = filter_intersections(intersection_points, st)

        if not out:
            print(f"No valid points found for {exp_name}. Skipping.")
            errors.append(None)
            continue

        # Run KMeans
        points_array = np.array(out)
        kmeans = KMeans(n_clusters=1, n_init=10, random_state=0).fit(points_array)
        estimated_position = kmeans.cluster_centers_[0]

        # Get true position
        true_position = get_device_coordinates(devices, target_mac)

        # Compute Euclidean error
        error = np.linalg.norm(np.array(true_position) - estimated_position)
        errors.append(error)

        print(f"Scene {i}: Estimated position = {estimated_position}, True = {true_position}, Error = {error:.2f}m")

    # Plot errors
    valid_errors = [e if e is not None else np.nan for e in errors]
    scene_indices = list(range(1, 8))

    plt.figure(figsize=(10, 5))
    plt.plot(scene_indices, valid_errors, marker='o', linestyle='-', color='dodgerblue', label='Localization Error')

    # Optional: annotate points
    for i, e in enumerate(valid_errors):
        if not np.isnan(e):
            plt.text(scene_indices[i], e + 0.05, f"{e:.2f}", ha='center', va='bottom', fontsize=9)

    plt.xticks(scene_indices, [f"Scene {i}" for i in scene_indices])
    plt.xlabel("Scene")
    plt.ylabel("Localization Error (meters)")
    plt.title("KMeans Localization Error per Scene")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()