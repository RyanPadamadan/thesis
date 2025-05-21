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

def setup_experiment_with_mesh(exp_name):
    coordinates_df, device_df, rssi_df, spatial_df = load_experiment_data_with_mesh(exp_name)
    device_pt = get_device_coordinates(device_df)
    points = map_distance_coords(coordinates_df, rssi_df, tx_power_curr)
    voxel_set = generate_voxel_set(spatial_df)    
    return points, device_pt, voxel_set

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

def assign_weights(experiment_dir, P=2.0):
    # Load data and get estimates
    points, device, voxel_set = setup_experiment_with_mesh(experiment_dir)
    surface_voxels = get_surface_voxels(voxel_set)  # List of (x, y, z) tuples
    estimates = get_estimates(points, 4)  # Estimate points to be weighted

    # Build KDTree from surface voxels
    surface_coords = np.array(surface_voxels)
    tree = KDTree(surface_coords)

    # For each estimate point, find distance to closest surface voxel
    distances, _ = tree.query(estimates)
    weights = np.exp(-P * distances)  # Exponential decay

    # Map each estimate (tuple) to its weight
    estimate_weights = {tuple(point): weight for point, weight in zip(estimates, weights)}
    return estimate_weights, device

def get_best_surface(surface, final):
    closest = None
    for voxel in surface:
        if closest is None:
            closest = voxel
            continue
        closest = voxel if distance(voxel, final) < distance(closest, final) else closest
    return closest

if __name__ == "__main__":
    answers = []
    prev = None
    it = True
    for i in range(1, 8):
        try:
            experiment = f"exp_{i}"
            print(experiment)
            estimate_weights, device = assign_weights(experiment)
            pred = weighted_mean(estimate_weights)
            e1 = distance(device, pred)

            print(e1)
        except Exception as e:
            print(e)