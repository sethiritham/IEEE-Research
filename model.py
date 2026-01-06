import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TrafficSTGNN(nn.Module):
    def __init__(self, num_node_features, hidden_channels, out_channels_speed=1, out_channels_acc=1):
        super(TrafficSTGNN, self).__init__()
        
        # 1. Spatial Component (GCN)
        # We learn "Intersection Embeddings"
        self.gcn1 = GCNConv(num_node_features, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)

        # 2. Temporal Component (GRU)
        # This learns the history trend
        self.gru = nn.GRU(hidden_channels, hidden_channels, batch_first=True)

        # 3. Prediction Heads (The "Edge" Predictors)
        # Input size is 'hidden_channels * 2' because we combine Node U + Node V
        self.head_speed = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels_speed)
        )
        
        self.head_acc = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels_acc) 
        )

    # --- THIS IS THE PART THAT CAUSED YOUR ERROR ---
    # We added edge_mapping_u and edge_mapping_v to the arguments
    def forward(self, x, edge_index, edge_mapping_u, edge_mapping_v):
        """
        x: [Num_Nodes, Seq_Len, Features]
        edge_index: [2, Num_Edges] 
        edge_mapping_u: [Num_Edges] (Index of start node for every edge)
        edge_mapping_v: [Num_Edges] (Index of end node for every edge)
        """
        
        batch_size_nodes, seq_len, num_features = x.shape
        
        # --- Step 1: Spatial Learning (GCN) ---
        # Flatten time temporarily for GCN: [Num_Nodes * Seq_Len, Features]
        # (Looping over time steps is safer for beginner implementation)
        spatial_out = []
        for t in range(seq_len):
            x_t = x[:, t, :] # [Num_Nodes, Features]
            
            h = self.gcn1(x_t, edge_index)
            h = F.relu(h)
            h = self.gcn2(h, edge_index) 
            
            spatial_out.append(h.unsqueeze(1))
            
        # Stack back to [Num_Nodes, Seq_Len, Hidden]
        spatial_features = torch.cat(spatial_out, dim=1)

        # --- Step 2: Temporal Learning (GRU) ---
        # GRU returns: output, h_n
        # We only care about the final hidden state (h_n) which summarizes the history
        _, h_n = self.gru(spatial_features) 
        
        # h_n shape: [1, Num_Nodes, Hidden] -> Squeeze to [Num_Nodes, Hidden]
        node_embeddings = h_n.squeeze(0)

        # --- Step 3: Edge Prediction (The "Link" Step) ---
        # To predict for an Edge, we look up the embeddings of its two nodes
        u_embed = node_embeddings[edge_mapping_u] # [Num_Edges, Hidden]
        v_embed = node_embeddings[edge_mapping_v] # [Num_Edges, Hidden]
        
        # Concatenate them to represent the "Road"
        # If Node A is [0.1, 0.5] and Node B is [0.9, 0.2]
        # The Road A->B becomes [0.1, 0.5, 0.9, 0.2]
        edge_embeddings = torch.cat([u_embed, v_embed], dim=1) 
        
        # --- Step 4: Multi-Task Output ---
        pred_speed = self.head_speed(edge_embeddings)
        pred_acc = self.head_acc(edge_embeddings)
        
        return pred_speed, pred_acc