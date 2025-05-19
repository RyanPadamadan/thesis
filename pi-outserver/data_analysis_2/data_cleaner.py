import pandas as pd
import numpy as np
import os
import math

# Define a point to be a tuple with x,y,z; not using classes because I cannot be bothered
def load_experiment_data(exp_dir_name):
    # just setting the correct script directory for 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, "../", exp_dir_name))

    # Define file paths
    coordinates_file = os.path.join(base_dir, "coordinates.csv")
    device_file = os.path.join(base_dir, "devices.csv")
    rssi_log_file = os.path.join(base_dir, "rssi_log.csv")

    # Load the data into DataFrames
    coordinates_df = pd.read_csv(coordinates_file)
    device_df = pd.read_csv(device_file)
    rssi_df = pd.read_csv(rssi_log_file)
    return coordinates_df, device_df, rssi_df

def load_experiment_data_with_mesh(exp_dir_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, "../", exp_dir_name))

    # Define file paths
    coordinates_file = os.path.join(base_dir, "coordinates.csv")
    device_file = os.path.join(base_dir, "devices.csv")
    rssi_log_file = os.path.join(base_dir, "rssi_log.csv")
    meshpoints_file = os.path.join(base_dir, "meshpoints.csv") 

    # Load the data into DataFrames
    coordinates_df = pd.read_csv(coordinates_file)
    device_df = pd.read_csv(device_file)
    rssi_df = pd.read_csv(rssi_log_file)
    meshpoints_df = pd.read_csv(meshpoints_file)  

    return coordinates_df, device_df, rssi_df, meshpoints_df  
    


def get_device_coordinates(device_df, target_mac=None):
    """
    Extracts the (x, y, z) coordinates of the target device from the device dataframe.
    If no target_mac is provided, and only one device is present, it returns that.
    
    Args:
        device_df (DataFrame): DataFrame containing 'mac', 'x', 'y', 'z' columns
        target_mac (str, optional): MAC address to find. Defaults to None.
    
    Returns:
        tuple: (x, y, z) coordinates of the device
    """
    if target_mac:
        device_row = device_df[device_df['mac'] == target_mac]
    else:
        device_row = device_df

    if device_row.empty:
        raise ValueError(f"Device with MAC {target_mac} not found in device dataframe.")

    x = device_row['x'].values[0]
    y = device_row['y'].values[0]
    z = device_row['z'].values[0]

    return (x, y, z)


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

def map_coords_dist(coordinates_df, rssi_df):
    rssi_vals = rssi_df.copy()
    coords = coordinates_df.copy()
    rssi_vals['timestamp'] = pd.to_numeric(rssi_df['timestamp'], errors='coerce')
    coords['timestamp'] = pd.to_numeric(coordinates_df['timestamp'], errors='coerce')

    coords_dist = []
    for coord_time, x, y, z in zip(coordinates_df['timestamp'], coordinates_df['x'], coordinates_df['y'], coordinates_df['z']):
        window_start = coord_time - 1
        window_end = coord_time + 1
        in_window = rssi_df[(rssi_df['timestamp'] >= window_start) & (rssi_df['timestamp'] <= window_end)]
        if not in_window.empty:
            median_rssi = in_window['rssi'].median()
            coords_dist.append((x, y, z, median_rssi))

    return coords_dist

def map_distance_coords(coordinates_df, rssi_df, tx):
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
            distance = path_loss_dist(median_rssi, tx)
            if distance > 2: # temp considering room size probably need to measure this nicely
                continue
            rssi_mapped.append((x, y, z, distance))

    return rssi_mapped

def get_transmission_rssi(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.abspath(os.path.join(script_dir, "..", file_name))
    df = pd.read_csv(file_path)
    return df['rssi'].median()

def path_loss_dist(rssi, tx):
    return 10 ** ((tx - rssi)/(10 * 2.5)) # assuming n = 2, because mostly open space

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])

