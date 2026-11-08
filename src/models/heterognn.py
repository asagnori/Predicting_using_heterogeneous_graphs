import torch
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv
from torch_geometric.nn import SAGEConv


class GNNModel(torch.nn.Module):

    def __init__(self, edge_types, hidden_channels, out_channels):

        super().__init__()
        self.edge_types = tuple(edge_types)

        # =====================================================
        # Primeira Camada
        # =====================================================

        self.conv1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels)
            for edge_type in self.edge_types
        }, aggr='sum')

        # =====================================================
        # Segunda Camada
        # =====================================================

        self.conv2 = HeteroConv({
            edge_type: SAGEConv((-1, -1), out_channels)
            for edge_type in self.edge_types
        }, aggr='sum')


    def forward(self, x_dict, edge_index_dict):
        first_layer = self.conv1(x_dict, edge_index_dict)
        first_layer = {
            key: F.relu(first_layer[key])
            if key in first_layer else x_dict[key]
            for key in x_dict
        }
        return self.conv2(first_layer, edge_index_dict)