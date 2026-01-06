import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from model import TrafficSTGNN

# --- CONFIGURATION ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Running on: {DEVICE}")

# --- 1. LOAD STATIC GRAPH (Structure) ---
edges_df = pd.read_csv('data/london_edges_master.csv')
nodes_df = pd.read_csv('data/london_nodes_master.csv')

edge_index = torch.tensor(
    [edges_df['u'].values, edges_df['v'].values], 
    dtype=torch.long
).to(DEVICE)

edge_map_u = torch.tensor(edges_df['u'].values, dtype=torch.long).to(DEVICE)
edge_map_v = torch.tensor(edges_df['v'].values, dtype=torch.long).to(DEVICE)

num_nodes = len(nodes_df)
print(f"Graph Structure Loaded: {num_nodes} Nodes, {len(edges_df)} Edges")

# --- 2. LOAD DYNAMIC DATA (Adarsh's Inputs) ---
data = np.load('data/training_data_2019.npz')
X_all = torch.from_numpy(data['x'])            
Y_speed_all = torch.from_numpy(data['y_speed'])
Y_acc_all = torch.from_numpy(data['y_acc'])   
Edge_idx_all = torch.from_numpy(data['edge_indices']) 

from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(X_all, Y_speed_all, Y_acc_all, Edge_idx_all)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = TrafficSTGNN(num_node_features=1, hidden_channels=32).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion_speed = nn.MSELoss()
criterion_acc = nn.BCEWithLogitsLoss()

# --- 4. TRAINING LOOP ---
print("\n--- Starting Training Loop ---")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    
    for batch_idx, (x_batch, y_s_batch, y_a_batch, edge_ids) in enumerate(train_loader):
        # Move batch to GPU
        x_batch = x_batch.to(DEVICE)   # [Batch, 12, 1]
        y_s_batch = y_s_batch.to(DEVICE)
        y_a_batch = y_a_batch.to(DEVICE)
        edge_ids = edge_ids.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Adarsh gives us Edge Data. GCN needs Node Data.
        # We assume Node Feature = Average of connected Edge Features.
        # Since this is a batch, we'll map the batch features to the FULL graph nodes.
        
        node_features = torch.zeros((num_nodes, 12, 1), device=DEVICE)
        batch_u_nodes = edge_map_u[edge_ids] 
        node_features[batch_u_nodes] = x_batch
        
        pred_speed_all, pred_acc_all = model(
            node_features, 
            edge_index, 
            edge_map_u, 
            edge_map_v
        )
        
        # Select only the edges in this batch
        pred_speed = pred_speed_all[edge_ids]
        pred_acc = pred_acc_all[edge_ids]
        
        # --- LOSS CALCULATION ---
        loss_s = criterion_speed(pred_speed, y_s_batch)
        loss_a = criterion_acc(pred_acc, y_a_batch)
        
        loss = loss_s + loss_a # Multi-task Loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

print("\nDONE! If you see loss decreasing, the pipeline is solid.")