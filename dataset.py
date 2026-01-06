import pandas as pd
import torch
import numpy as np
import os
from torch_geometric.data import Data, Dataset

class LondonTrafficDataset(Dataset):
    def __init__(self, root_dir, nodes_file, edges_file, traffic_file=None, lookback=12):
        super().__init__()
        self.root_dir = root_dir
        self.lookback = lookback
        
        # Paths
        self.nodes_path = os.path.join(root_dir, nodes_file)
        self.edges_path = os.path.join(root_dir, edges_file)
        self.traffic_path = traffic_file
        
        self._process_static_graph()

    def _process_static_graph(self):
        # 1. Load your specific CSV
        df_edges = pd.read_csv(self.edges_path)
        
        # 2. Construct Edge Index
        self.edge_index = torch.tensor(
            [df_edges['u'].values, df_edges['v'].values], 
            dtype=torch.long
        )
        
        # 3. Static Features
        speed = pd.to_numeric(df_edges['speed_kph'], errors='coerce').fillna(30) / 120.0
        length = pd.to_numeric(df_edges['length'], errors='coerce').fillna(50) / 1000.0
        
        # Shape: [Num_Edges, 2]
        self.edge_attr = torch.tensor(
            np.stack([speed.values, length.values], axis=1), 
            dtype=torch.float32
        )
        
        self.num_edges = len(df_edges)
        print(f"Graph Loaded: {self.num_edges} edges.")

    def len(self):
        # Placeholder until traffic data arrives
        return 100 

    def get(self, idx):
        # Placeholder for Model Testing
        dummy_x = torch.zeros((self.num_edges, self.lookback, 1))
        dummy_y_speed = torch.zeros((self.num_edges, 1))
        dummy_y_acc = torch.zeros((self.num_edges, 1))

        return Data(
            x=dummy_x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            y_speed=dummy_y_speed,
            y_acc=dummy_y_acc
        )