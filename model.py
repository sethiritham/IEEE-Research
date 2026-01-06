import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TrafficSTGNN(nn.Module):
    def __init__(self, num_node_features, hidden_channels, out_channels_speed=1, out_channels_acc=1):
        super(TrafficSTGNN, self).__init__()
        
        # 1. Spatial Encoder (GCN)
        self.gcn1 = GCNConv(num_node_features, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)

        # 2. Temporal Encoder (GRU)
        self.gru = nn.GRU(hidden_channels, hidden_channels, batch_first=True)

        # 3. Heads
        self.head_speed = nn.Linear(hidden_channels, out_channels_speed)
        self.head_acc = nn.Linear(hidden_channels, out_channels_acc)

    def forward(self, x, edge_index):
        """
        x: [Num_Nodes, Seq_Len, Features] - We need to derive this from Edges later
        edge_index: [2, Num_Edges]
        """
        return x, x