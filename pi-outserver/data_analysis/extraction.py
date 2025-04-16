import pandas as pd
import numpy as np
import os
import math

def load_experiment_data(exp_dir_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, "..", exp_dir_name))

    # Define file paths
    coordinates_file = os.path.join(base_dir, "coordinates.csv")
    device_file = os.path.join(base_dir, "devices.csv")
    rssi_log_file = os.path.join(base_dir, "rssi_log.csv")

    # Load the data into DataFrames
    coordinates_df = pd.read_csv(coordinates_file)
    device_df = pd.read_csv(device_file)
    rssi_df = pd.read_csv(rssi_log_file)

    return coordinates_df, device_df, rssi_df

def map_rssi_coords(coordinates_df, rssi_df):
    rssi_vals = rssi_df.copy()
    coords = coordinates_df.copy()
    rssi_vals['timestamp'] = pd.to_numeric(rssi_df['timestamp'], errors='coerce')
    coords['timestamp'] = pd.to_numeric(coordinates_df['timestamp'], errors='coerce')

    rssi_mapped = []
    for coord_time, x, y, z in zip(coordinates_df['timestamp'], coordinates_df['x'], coordinates_df['y'], coordinates_df['z']):
        window_start = coord_time - 1
        window_end = coord_time + 1
        in_window = rssi_df[(rssi_df['timestamp'] >= window_start) & (rssi_df['timestamp'] <= window_end)]
        if not in_window.empty:
            median_rssi = in_window['rssi'].median()
            rssi_mapped.append((x, y, z, median_rssi))

    return rssi_mapped

def get_transmission_rssi(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.abspath(os.path.join(script_dir, "..", file_name))
    df = pd.read_csv(file_path)
    return df['rssi'].median()
def path_loss_dist(rssi, tx):
    return 10 ** ((tx - rssi)/10 * 2) # assuming n = 2, because mostly open space

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])

def trilateration(p1, r1, p2, r2, p3, r3):
    P1, P2, P3 = np.array(p1), np.array(p2), np.array(p3)
    ex = (P2 - P1)
    ex = ex / np.linalg.norm(ex)
    i = np.dot(ex, P3 - P1)
    temp = P3 - P1 - i * ex
    ey = temp / np.linalg.norm(temp)
    ez = np.cross(ex, ey)
    d = np.linalg.norm(P2 - P1)
    j = np.dot(ey, P3 - P1)

    x = (r1**2 - r2**2 + d**2) / (2 * d)
    y = (r1**2 - r3**2 + i**2 + j**2 - 2 * i * x) / (2 * j)
    z_square = r1**2 - x**2 - y**2
    if z_square < 0:
        return None
    z = np.sqrt(z_square)

    result = P1 + x * ex + y * ey + z * ez
    return result.tolist()
