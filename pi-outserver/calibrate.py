import requests
import numpy as np
import pandas as pd
import os
import re
import json
from sklearn.cluster import KMeans
from data_analysis.ext import map_rssi_coords, compute_intersections, path_loss_dist, trilateration, get_transmission_rssi

""" Live calibration module """

# API endpoints
RSSI_API = "http://127.0.0.1:8081/rssi"
COORD_API = "http://127.0.0.1:8081/coords"
UPDATE_DEVICES_API = "http://127.0.0.1:8081/update_devices"

# Calibration parameters
TX_POWER_FILE = "rssi_1m.csv"
tx_power_curr = get_transmission_rssi(TX_POWER_FILE)

# Predefined MAC addresses to calibrate
TARGET_MACS = [
    "04:99:bb:d8:6e:2e",
    "04:99:bb:d8:6e:3f",
    "04:99:bb:d8:6e:4a"
]

def fetch_api_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[ERROR] Failed to fetch from {url}: {e}")
        return []

def post_calibrated_devices(devices):
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(UPDATE_DEVICES_API, data=json.dumps(devices), headers=headers)
        response.raise_for_status()
        print("[INFO] Successfully posted calibrated device positions to server.")
    except Exception as e:
        print(f"[ERROR] Failed to post calibrated devices: {e}")

def localize_live_kmeans(coords_df, rssi_df, tx_power, target_mac, n_clusters=1):
    filtered_rssi_df = rssi_df[rssi_df['src'] == target_mac]
    if filtered_rssi_df.empty:
        print(f"[INFO] No RSSI data found for device {target_mac}.")
        return None

    mapped = map_rssi_coords(coords_df, filtered_rssi_df)
    intersection_points = compute_intersections(mapped, tx_power)
    if not intersection_points:
        print(f"[INFO] No valid intersection points found for {target_mac}.")
        return None

    points_array = np.array(intersection_points)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(points_array)
    estimated_position = kmeans.cluster_centers_[0]
    return tuple(estimated_position)

def run_live_calibration(target_macs):
    print("Fetching RSSI data...")
    rssi_data = fetch_api_data(RSSI_API)
    rssi_df = pd.DataFrame(rssi_data)

    print("Fetching coordinate data...")
    coord_data = fetch_api_data(COORD_API)
    coords_df = pd.DataFrame(coord_data)

    if rssi_df.empty or coords_df.empty:
        print("Calibration failed: One or more datasets are empty.")
        return

    calibrated_devices = []

    for target_mac in target_macs:
        print(f"Running live KMeans localization for device {target_mac}...")
        estimated_position = localize_live_kmeans(coords_df, rssi_df, tx_power_curr, target_mac=target_mac)
        if estimated_position is None:
            print(f"[WARNING] Failed to estimate position for {target_mac}. Skipping.")
            continue

        x, y, z = estimated_position

        calibrated_devices.append({
            "mac": target_mac,
            "x": x,
            "y": y,
            "z": z
        })

        print(f"Estimated Position for {target_mac}: ({x:.2f}, {y:.2f}, {z:.2f})")

    if calibrated_devices:
        print("Posting calibrated devices to server...")
        post_calibrated_devices(calibrated_devices)
    else:
        print("No devices calibrated successfully. Nothing to post.")

def calculate_position_error(estimated, real):
    x_hat, y_hat, z_hat = estimated
    x, y, z = real
    return np.sqrt((x_hat - x)**2 + (y_hat - y)**2 + (z_hat - z)**2)

if __name__ == "__main__":
    run_live_calibration(TARGET_MACS)




