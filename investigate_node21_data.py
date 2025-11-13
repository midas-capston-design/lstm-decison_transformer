#!/usr/bin/env python3
"""
Investigate whether 21→11 route data is actually in the training set
"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import pickle

# Load processed data
data_dir = Path('hyena/processed_data_hyena')
states_train = np.load(data_dir / 'states_train.npy')
positions_train = np.load(data_dir / 'positions_train.npy')

with open(data_dir / 'metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

coords_min = np.array(metadata['normalization']['coords_min'])
coords_max = np.array(metadata['normalization']['coords_max'])

# Denormalize
def denormalize_coords(coords_norm, coords_min, coords_max):
    coords_range = coords_max - coords_min
    return (coords_norm + 1) / 2 * coords_range + coords_min

positions_train_real = denormalize_coords(positions_train, coords_min, coords_max)

# Load node coordinates
nodes_df = pd.read_csv('nodes_final.csv')
node_coords = {}
for _, row in nodes_df.iterrows():
    node_coords[row['id']] = (row['x_m'], row['y_m'])

print("=" * 80)
print("Investigating Node 21 Data")
print("=" * 80)

# Node 21 and 20 positions
node_21_pos = node_coords[21]
node_20_pos = node_coords[20]
print(f"\nNode 21: {node_21_pos}")
print(f"Node 20: {node_20_pos}")

# Grid function
GRID_SIZE = 0.9

def coord_to_grid(x, y):
    return (int(x / GRID_SIZE), int(y / GRID_SIZE))

grid_21 = coord_to_grid(*node_21_pos)
grid_20 = coord_to_grid(*node_20_pos)
print(f"\nGrid 21: {grid_21}")
print(f"Grid 20: {grid_20}")

# Find all samples near node 21 (within 2 meters)
print(f"\n{'='*80}")
print("Samples near Node 21 (within 2m)")
print("=" * 80)

samples_near_21 = []
for i, (x, y) in enumerate(positions_train_real):
    dist_to_21 = np.sqrt((x - node_21_pos[0])**2 + (y - node_21_pos[1])**2)
    if dist_to_21 < 2.0:
        samples_near_21.append((i, x, y, dist_to_21))

print(f"\nFound {len(samples_near_21)} samples within 2m of node 21")

if len(samples_near_21) > 0:
    print("\nFirst 10 samples:")
    print(f"{'Index':<8} {'X':<10} {'Y':<10} {'Distance':<10} {'Grid'}")
    print("-" * 80)
    for idx, x, y, dist in samples_near_21[:10]:
        grid = coord_to_grid(x, y)
        print(f"{idx:<8} {x:<10.2f} {y:<10.2f} {dist:<10.2f} {grid}")
else:
    print("❌ NO SAMPLES FOUND NEAR NODE 21!")
    print("This means the 21→11 route files have NOT been processed yet.")

# Find all samples near node 20 (within 2 meters)
print(f"\n{'='*80}")
print("Samples near Node 20 (within 2m)")
print("=" * 80)

samples_near_20 = []
for i, (x, y) in enumerate(positions_train_real):
    dist_to_20 = np.sqrt((x - node_20_pos[0])**2 + (y - node_20_pos[1])**2)
    if dist_to_20 < 2.0:
        samples_near_20.append((i, x, y, dist_to_20))

print(f"\nFound {len(samples_near_20)} samples within 2m of node 20")

if len(samples_near_20) > 0:
    print("\nFirst 10 samples:")
    print(f"{'Index':<8} {'X':<10} {'Y':<10} {'Distance':<10} {'Grid'}")
    print("-" * 80)
    for idx, x, y, dist in samples_near_20[:10]:
        grid = coord_to_grid(x, y)
        print(f"{idx:<8} {x:<10.2f} {y:<10.2f} {dist:<10.2f} {grid}")

# Check grid coverage
print(f"\n{'='*80}")
print("Grid cells with samples")
print("=" * 80)

grid_samples = defaultdict(int)
for x, y in positions_train_real:
    grid = coord_to_grid(x, y)
    grid_samples[grid] += 1

# Check if grid_21 and grid_20 have samples
print(f"\nGrid {grid_21} (node 21): {grid_samples.get(grid_21, 0)} samples")
print(f"Grid {grid_20} (node 20): {grid_samples.get(grid_20, 0)} samples")

# Show grids with highest sample counts (near 21 and 20 area)
print("\nNearby grids (-95 to -94, 0 to 6):")
for gx in range(-96, -93):
    for gy in range(-1, 7):
        count = grid_samples.get((gx, gy), 0)
        if count > 0:
            print(f"  Grid ({gx}, {gy}): {count} samples")

print("\n" + "=" * 80)
