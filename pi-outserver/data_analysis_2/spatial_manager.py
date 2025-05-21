from data_cleaner import *
import numpy as np
from itertools import combinations
from scipy.optimize import least_squares
from algorithms import *
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

if __name__ == "__main__":
    answers = []
    prev = None
    it = True
    for i in range(1, 3):
        try:
            experiment = f"exp_{i}"
            print(experiment)
            prev = get_best_voxel(experiment_dir=experiment, clustering_algorithm=k_means)
        except Exception as e:
            print(e)