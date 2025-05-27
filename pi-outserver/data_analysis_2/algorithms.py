"""
This is source file contains the some algorithms I've written.
The algorithms created are:
k lateration
k means
k means with ewma
k medoid
iterative distance improvement
"""
from data_cleaner import *
import numpy as np
from itertools import combinations
from scipy.optimize import least_squares
import random

# Some globals that we will use
tx_power_curr = get_transmission_rssi("rssi_1m.csv")
target_mac = "04:99:bb:d8:6e:2e"

def setup_experiment_no_mesh(exp_name):
    coordinates_df, device_df, rssi_df, _ = load_experiment_data_with_mesh(exp_name)
    device_pt = get_device_coordinates(device_df)
    points = map_distance_coords(coordinates_df, rssi_df, tx_power_curr)
    return points, device_pt

def  n_lateration(points):
    """
    Best estimate of an IoT device given n devices 
    Args:
        points (list of tuples): Each tuple is (x, y, z, distance), with at least 4 points.
    Returns:
        np.ndarray: Estimated position (x, y, z) of the unknown device.
    """
    positions = []
    distances = []

    for x, y, z, d in points:
        if d is None or not np.isfinite(d):
            continue
        positions.append((x, y, z))
        distances.append(d)

    if len(positions) < 4:
        raise ValueError("Not enough valid points for lateration.")

    def residuals(guess, positions, distances):
        x, y, z = guess
        return [
            math.sqrt((x - xi)**2 + (y - yi)**2 + (z - zi)**2) - di
            for (xi, yi, zi), di in zip(positions, distances)
        ]

    initial_guess = np.mean(positions, axis=0)
    result = least_squares(residuals, initial_guess, args=(positions, distances))
    return result.x

def get_estimates(points, k):
    """
    Run n-lateration on up to 2000 random k-point combinations.
    
    Args:
        points: list of (x, y, z, distance)
        k: number of points per subset (k >= 4)
        
    Returns:
        np.ndarray of estimated positions
    """
    if k < 4:
        raise ValueError("k must be at least 4 for 3D trilateration.")
    if len(points) < k:
        raise ValueError("Not enough points to select k from.")

    all_combinations = list(combinations(points, k))
    print(len(all_combinations))
    random.shuffle(all_combinations)
    sampled_combinations = all_combinations[:3000] 

    estimates = []
    for subset in sampled_combinations:
        try:
            estimate = n_lateration(list(subset))
            estimates.append(estimate)
        except Exception:
            continue

    return np.array(estimates)

def k_means_ewma(points, prev=None, alpha=0.7):
    """
    Compute centroid of points, optionally blended with previous estimate using EWMA.

    Args:
        points (np.ndarray): Array of shape (N, 3)
        prev (np.ndarray or None): Previous (x, y, z) estimate
        alpha (float): EWMA factor (0 < alpha ≤ 1), higher = more weight to current

    Returns:
        np.ndarray: Blended centroid
    """
    current = np.mean(points, axis=0)
    
    if prev is None:
        return current
    
    prev = np.asarray(prev, dtype=float)
    return (1 - alpha) * current + alpha * prev

def k_means(points):
    """
    Compute centroid of points, optionally blended with previous estimate using EWMA.

    Args:
        points (np.ndarray): Array of shape (N, 3)
        prev (np.ndarray or None): Previous (x, y, z) estimate
        alpha (float): EWMA factor (0 < alpha ≤ 1), higher = more weight to current

    Returns:
        np.ndarray: Blended centroid
    """
    return np.mean(points, axis=0)

def median(points):
    return np.median(points, axis=0)

def weighted_mean(weight_map):
    """
    Compute the weighted mean (centroid) from a dictionary of {(x, y, z): weight}.
    
    Args:
        weight_map (dict): Mapping from (x, y, z) tuple to weight (float).
        
    Returns:
        np.ndarray: Weighted mean as a (3,) numpy array.
    """
    if not weight_map:
        return np.array([0.0, 0.0, 0.0])

    points = np.array(list(weight_map.keys()))
    weights = np.array(list(weight_map.values()))

    weighted_sum = np.sum(points * weights[:, np.newaxis], axis=0)
    total_weight = np.sum(weights)

    return weighted_sum / total_weight


def k_medoid(points):
    """
    Compute the medoid (most central actual point) among a set of 3D points.
    This is the point that has the minimal total Euclidean distance to all others.

    Args:
        points (np.ndarray): Array of shape (N, 3)

    Returns:
        np.ndarray: The medoid point (x, y, z)
    """
    if len(points) == 0:
        raise ValueError("No points provided.")

    if len(points) == 1:
        return points[0]  
    distances = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    total_distances = np.sum(distances, axis=1)
    medoid_index = np.argmin(total_distances)
    return points[medoid_index]

def find_best_experimental(exp_dir, clustering_algorithm, k, it=False, prev=None):
    """
    Run an experiment to find the best estimated device location using a specified clustering algorithm.

    Parameters:
    exp_dir (str): The name or path of the experiment directory containing the data.
    clustering_algorithm (Callable): A clustering function that takes a list of estimates and returns a final estimate.
    k (int): The number of estimates to generate from the experiment data.

    Returns:
    tuple: A tuple containing:
        - final: The final estimated location of the device from the clustering algorithm.
        - device: The actual device location used as ground truth.
    """
    points, device = setup_experiment_no_mesh(exp_dir)
    estimates = get_estimates(points, k)
    if it:
        final = clustering_algorithm(estimates, prev)
    else:
        final = clustering_algorithm(estimates)
    return final, device


if __name__ == "__main__":
    answers = []
    for i in range(1, 8):
        try:
            experiment = f"exp_{i}"
            print(experiment)
            final, device = find_best_experimental(experiment, k_means, 3)
            error = distance(device, final)
            print(error)
            answers.append(final)
        except Exception as e:
            print(e)
    print(error)

