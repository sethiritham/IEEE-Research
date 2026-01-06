import numpy as np
import pandas as pd
import torch

# 1. Load your master edge list to know how many edges we have
try:
    edges = pd.read_csv('data/london_edges_master.csv')
    num_edges = len(edges)
    print(f"Generating data for {num_edges} edges...")
except FileNotFoundError:
    print("Error: 'london_edges_master.csv' not found. Run london_map.py first.")
    exit()

# 2. Simulation Parameters
NUM_SAMPLES = 1000  # Let's simulate 1000 time-windows
LOOKBACK = 12       # 3 hours history
NUM_FEATURES = 1    # Speed only

# 3. Create Random Dummy Arrays
x_dummy = np.random.rand(NUM_SAMPLES, LOOKBACK, NUM_FEATURES).astype(np.float32)

# Targets:
y_speed_dummy = np.random.rand(NUM_SAMPLES, 1).astype(np.float32) # Next speed
y_acc_dummy = np.random.randint(0, 2, (NUM_SAMPLES, 1)).astype(np.float32) # 0 or 1

# 4. Create the Edge Mapping (Crucial)
edge_indices_dummy = np.random.randint(0, num_edges, NUM_SAMPLES)

# 5. Save as .npz (Adarsh's Format)
np.savez_compressed(
    'data/training_data_2019.npz',
    x=x_dummy, 
    y_speed=y_speed_dummy, 
    y_acc=y_acc_dummy,
    edge_indices=edge_indices_dummy 
)

print("SUCCESS: 'training_data_2019.npz' created.")
print("You can now run train_temp.py")