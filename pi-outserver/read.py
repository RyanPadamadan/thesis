import requests
import csv
import os
import re

# API endpoints
RSSI_API = "http://127.0.0.1:8081/rssi"
COORD_API = "http://127.0.0.1:8081/coords"
DEVICE_API = "http://127.0.0.1:8081/device"
MESH_API = "http://127.0.0.1:8081/meshpoints"  # ✅ new

# File names
RSSI_FILE = "rssi_log.csv"
COORD_FILE = "coordinates.csv"
DEVICE_FILE = "devices.csv"
MESH_FILE = "meshpoints.csv"  # ✅ new

def fetch_api_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[ERROR] Failed to fetch from {url}: {e}")
        return []

def save_to_csv(data, filename):
    if not data:
        print(f"[INFO] No data to write to {filename}.")
        return

    fieldnames = list(data[0].keys())

    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    print(f"Saved {len(data)} entries to {filename}")

def get_next_exp_folder(base_name="exp_"):
    existing = [d for d in os.listdir() if os.path.isdir(d) and re.match(f"^{base_name}[0-9]+$", d)]
    nums = [int(d.split("_")[1]) for d in existing]
    next_num = max(nums) + 1 if nums else 1
    return f"{base_name}{next_num}"

def save_all_logs():
    print("Determining experiment folder...")
    exp_dir = get_next_exp_folder()
    os.makedirs(exp_dir)
    print(f"Created experiment directory: {exp_dir}")

    # Fetch + save RSSI
    print("[⬇️] Fetching RSSI data...")
    rssi_data = fetch_api_data(RSSI_API)
    save_to_csv(rssi_data, os.path.join(exp_dir, RSSI_FILE))

    # Fetch + save Coordinates
    print("[⬇️] Fetching coordinate data...")
    coord_data = fetch_api_data(COORD_API)
    save_to_csv(coord_data, os.path.join(exp_dir, COORD_FILE))

    # Fetch + save Devices
    print("[⬇️] Fetching device data...")
    device_data = fetch_api_data(DEVICE_API)
    save_to_csv(device_data, os.path.join(exp_dir, DEVICE_FILE))

    # ✅ Fetch + save Mesh Points
    print("[⬇️] Fetching meshpoints data...")
    mesh_data = fetch_api_data(MESH_API)
    save_to_csv(mesh_data, os.path.join(exp_dir, MESH_FILE))

    print(f"[✅] All data saved under {exp_dir}/")
    return exp_dir

if __name__ == "__main__":
    save_all_logs()
