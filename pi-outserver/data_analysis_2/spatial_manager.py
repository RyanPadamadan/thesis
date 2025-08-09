from data_cleaner import *
import numpy as np
from itertools import combinations
from scipy.optimize import least_squares
from algorithms import *
from scipy.spatial import KDTree
import random
# algorithms with spatial data
"""
from experiments its clear that beyond a certain threshold theres not much we can do about the error 
"""
tx_power_curr = get_transmission_rssi("rssi_1m.csv")
target_mac = "04:99:bb:d8:6e:2e"
# experiment 2 : post processing estimates
voxel_tree = None
surface_tree = None

def setup_experiment_with_mesh(exp_name):
    coordinates_df, device_df, rssi_df, spatial_df = load_experiment_data_with_mesh(exp_name)
    device_pt = get_device_coordinates(device_df)
    points = map_distance_coords(coordinates_df, rssi_df, tx_power_curr)
    voxel_set = generate_voxel_set(spatial_df)    
    return points, device_pt, voxel_set

def setup_field(exp_name):
    coordinates_df, _, rssi_df, _ = load_experiment_data_with_mesh(exp_name)
    points = map_rssi_coords(coordinates_df, rssi_df, tx_power_curr)
    return points

def get_surface_voxels(voxel_set):
    """
    Identifies and returns surface voxels from a given set of voxels.
    
    A surface voxel is one that has at least one face exposed, i.e., one or more of its
    6 axis-aligned neighbors (±VOXELSIZE along x, y, or z) are not present in the set.

    Parameters:
    voxel_set (set of tuple): A set containing (x, y, z) coordinates of voxels.

    Returns:
    list: A list of surface voxels as (x, y, z) tuples.
    """
    surface_voxels = []
    directions = [
        (VOXELSIZE, 0, 0), (-VOXELSIZE, 0, 0),
        (0, VOXELSIZE, 0), (0, -VOXELSIZE, 0),
        (0, 0, VOXELSIZE), (0, 0, -VOXELSIZE)
    ]

    for voxel in voxel_set:
        for dx, dy, dz in directions:
            neighbor = (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz)
            if neighbor not in voxel_set:
                surface_voxels.append(voxel)
                break  # Only need one missing neighbor to qualify as surface
    return surface_voxels

def get_best_voxel(experiment_dir, k=4, clustering_algorithm=k_means_ewma, it=False, prev=None):
    final, device = find_best_experimental(experiment_dir, clustering_algorithm, k, it, prev)
    _, _, voxel_set = setup_experiment_with_mesh(experiment_dir)
    surface = get_surface_voxels(voxel_set)
    closest = None
    for voxel in surface:
        if closest is None:
            closest = voxel
            continue
        closest = voxel if distance(voxel, final) < distance(closest, final) else closest
    print(f"Error: {distance(device, closest)}")
    return closest

def assign_weights(experiment_dir, k, P=2.0, max_dist=3):
    # Load data and get estimates
    points, device, voxel_set = setup_experiment_with_mesh(experiment_dir)
    surface_voxels = get_surface_voxels(voxel_set)  # List of (x, y, z) tuples
    estimates = get_estimates(points, k)  # Estimate points to be weighted

    # Build KDTree from surface voxels
    surface_coords = np.array(surface_voxels)
    tree = KDTree(surface_coords)

    # For each estimate point, find distance to closest surface voxel
    distances, _ = tree.query(estimates)
    weights = np.exp(-(P/max_dist) * distances)  # Exponential decay, setting the 

    # Map each estimate (tuple) to its weight
    estimate_weights = {tuple(point): weight for point, weight in zip(estimates, weights)}
    return estimate_weights, device

def assign_weights_rssi(experiment_dir, k=4, Q=3.0, P=3.0, alpha=0.9, rssi_max=-70):
    # --- Step 1: Load data ---
    rssi_points = setup_field(experiment_dir)  
    points, device, voxel_set = setup_experiment_with_mesh(experiment_dir)
    surface_voxels = get_surface_voxels(voxel_set)
    voxel_rssi_map = generate_voxel_rssi_field(rssi_points)
    estimates = get_estimates(points, k)

    # --- Step 2: Build KD-Trees ---
    surface_tree = KDTree(np.array(surface_voxels))
    voxel_coords = [np.array(voxel) * VOXELSIZER for voxel in voxel_rssi_map.keys()]
    voxel_tree = KDTree(voxel_coords)
    voxel_keys = list(voxel_rssi_map.keys())

    # --- Step 3: Compute weights ---
    combined_weights = {}
    for est in estimates:
        d_i, _ = surface_tree.query(est)

        δ_i, voxel_idx = voxel_tree.query(est)
        voxel = voxel_keys[voxel_idx]
        rssi = voxel_rssi_map[voxel]

        # Compute confidence penalty
        c_i = (rssi - rssi_max) / rssi_max
        # Apply exponential decay filtering
        x_i = np.exp(-Q * c_i) * np.exp(-P * δ_i)
        w_i = x_i * (1 + alpha * np.exp(-Q * d_i))

        combined_weights[tuple(est)] = w_i
    return combined_weights, device

def assign_weights_rssi_v2(experiment_dir, estimates, k=10, Q=3.0, P=3.0, alpha=0.9, rssi_max=-70):
    # --- Step 1: Load data ---
    rssi_points = setup_field(experiment_dir)  
    points, device, voxel_set = setup_experiment_with_mesh(experiment_dir)
    surface_voxels = get_surface_voxels(voxel_set)
    voxel_rssi_map = generate_voxel_rssi_field(rssi_points)

    # --- Step 2: Build KD-Trees ---
    surface_tree = KDTree(np.array(surface_voxels))
    voxel_coords = [np.array(voxel) * VOXELSIZER for voxel in voxel_rssi_map.keys()]
    voxel_tree = KDTree(voxel_coords)
    voxel_keys = list(voxel_rssi_map.keys())

    # --- Step 3: Compute weights ---
    combined_weights = {}
    for est in estimates:
        d_i, _ = surface_tree.query(est)

        δ_i, voxel_idx = voxel_tree.query(est)
        voxel = voxel_keys[voxel_idx]
        rssi = voxel_rssi_map[voxel]

        # Compute confidence penalty
        c_i = (rssi - rssi_max) / rssi_max
        # Apply exponential decay filtering
        x_i = np.exp(-Q * c_i) * np.exp(-Q * δ_i)
        w_i = x_i * (1 + alpha * np.exp(-P * d_i))

        combined_weights[tuple(est)] = w_i
    return combined_weights, device

def generate_voxel_rssi_field(points):
    voxel_rssi_map = {}

    for x, y, z, rssi in points:
        i = int(np.floor(x / VOXELSIZER))
        j = int(np.floor(y / VOXELSIZER))
        k = int(np.floor(z / VOXELSIZER))

        key = (i, j, k)  # Use indices, not real coordinates

        if key not in voxel_rssi_map:
            voxel_rssi_map[key] = []
        voxel_rssi_map[key].append(rssi)

    return {k: np.median(v) for k, v in voxel_rssi_map.items()}


def get_best_surface(surface, final):
    closest = None
    for voxel in surface:
        if closest is None:
            closest = voxel
            continue
        closest = voxel if distance(voxel, final) < distance(closest, final) else closest
    return closest

def get_exponential_decay(exp_dir, k=4):
    """
    Performs exponential weighting and returns:
    - nearest surface voxel to the weighted estimate (final predicted location)
    - actual device location
    - original weighted estimate (optional debug)

    Args:
        exp_dir (str): Experiment name (e.g., "exp_3")
        k (int): Number of points per trilateration subset

    Returns:
        tuple: (predicted_surface_voxel, device, raw_weighted_estimate)
    """
    # global surface_tree

    # Weighted estimate from RSSI + spatial weights
    estimate_weights, device = assign_weights_rssi(exp_dir, k)
    raw_pred = weighted_mean(estimate_weights)

    # Align prediction to closest surface voxel
    _, _, voxel_set = setup_experiment_with_mesh(exp_dir)
    surface_voxels = get_surface_voxels(voxel_set)
    surface_tree = KDTree(np.array(surface_voxels))
    _, idx = surface_tree.query(raw_pred)
    nearest_surface_voxel = surface_voxels[idx]

    return nearest_surface_voxel, device


if __name__ == "__main__":
    answers = []
    prev = None
    it = True
    for i in range(1, 36):
        try:
            experiment = f"exp_{i}"
            print(experiment)
            estimate_weights, device = assign_weights_rssi(experiment, 4)
            pred = weighted_mean(estimate_weights)
            # pred, device = get_exponential_decay(experiment)
            e1 = distance(device, pred)
            print(f"Error: {e1}")

            # Get the surface voxel closest to prediction
            # _, _, voxel_set = setup_experiment_with_mesh(experiment)
            # surface_voxels = get_surface_voxels(voxel_set)
            # surface_tree = KDTree(np.array(surface_voxels))
            # _, idx = surface_tree.query(pred)
            # nearest_surface_voxel = surface_voxels[idx]
            # surface_error = distance(pred, nearest_surface_voxel)
            # print(f"Closest surface voxel: {nearest_surface_voxel}, Error to surface: {surface_error}")
        
        except Exception as e:
            print(e)
            break
