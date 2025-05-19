from data_cleaner import *
import numpy as np
from itertools import combinations
from scipy.optimize import least_squares
import random

tx_power_curr = get_transmission_rssi("rssi_1m.csv")
target_mac = "04:99:bb:d8:6e:2e"

def setup_experiment_no_mesh(exp_name):
    coordinates_df, device_df, rssi_df, spatial_data = load_experiment_data_with_mesh(exp_name)
    device_pt = get_device_coordinates(device_df)
    points = map_distance_coords(coordinates_df, rssi_df, tx_power_curr)
    
    return points, device_pt
