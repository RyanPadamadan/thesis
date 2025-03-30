import matplotlib.pyplot as plt
import os

rssi_data = {}


for d in range(4):
    filename = f"data_{d}"
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            # Convert to integers or floats
            rssi_values = [float(line.strip()) for line in lines if line.strip()]
            rssi_data[d] = rssi_values
    else:
        print(f"File {filename} not found.")


x = []
y = []

for distance, rssi_list in rssi_data.items():
    x.extend([distance] * len(rssi_list))
    y.extend(rssi_list)

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(x, y, alpha=0.7)
plt.title("RSSI vs Distance")
plt.xlabel("Distance (m)")
plt.ylabel("RSSI (dBm)")
plt.grid(True)
plt.xticks(range(min(x), max(x) + 1))
plt.gca().invert_yaxis() 
plt.gca().invert_xaxis() 
plt.show()

