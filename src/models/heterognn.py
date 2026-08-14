import torch
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv
from torch_geometric.nn import SAGEConv


class GNNModel(torch.nn.Module):

    def __init__(self, hidden_channels, out_channels):

        super().__init__()

        # =====================================================
        # Primeira Camada
        # =====================================================

        self.conv1 = HeteroConv({

            ('customer', 'to', 'order'):
                SAGEConv((-1, -1), hidden_channels),

            ('product', 'to', 'order'):
                SAGEConv((-1, -1), hidden_channels),

            ('brand', 'to', 'product'):
                SAGEConv((-1, -1), hidden_channels),

            ('category', 'to', 'product'):
                SAGEConv((-1, -1), hidden_channels),

            ('store', 'to', 'order'):
                SAGEConv((-1, -1), hidden_channels),

            ('staff', 'to', 'order'):
                SAGEConv((-1, -1), hidden_channels),

            # Self-Loops
            ('customer', 'to', 'customer'):
                SAGEConv((-1, -1), hidden_channels),

            ('brand', 'to', 'brand'):
                SAGEConv((-1, -1), hidden_channels),

            ('order', 'to', 'order'):
                SAGEConv((-1, -1), hidden_channels),

            ('product', 'to', 'product'):
                SAGEConv((-1, -1), hidden_channels),

        }, aggr='sum')

        # =====================================================
        # Segunda Camada
        # =====================================================

        self.conv2 = HeteroConv({

            ('customer', 'to', 'order'):
                SAGEConv((-1, -1), out_channels),

            ('product', 'to', 'order'):
                SAGEConv((-1, -1), out_channels),

            ('brand', 'to', 'product'):
                SAGEConv((-1, -1), out_channels),

            ('category', 'to', 'product'):
                SAGEConv((-1, -1), out_channels),

            ('store', 'to', 'order'):
                SAGEConv((-1, -1), out_channels),

            ('staff', 'to', 'order'):
                SAGEConv((-1, -1), out_channels),

            # Self-Loops
            ('customer', 'to', 'customer'):
                SAGEConv((-1, -1), out_channels),

            ('brand', 'to', 'brand'):
                SAGEConv((-1, -1), out_channels),

            ('order', 'to', 'order'):
                SAGEConv((-1, -1), out_channels),

            ('product', 'to', 'product'):
                SAGEConv((-1, -1), out_channels),

        }, aggr='sum')


    def forward(self, x_dict, edge_index_dict):

        x_dict = self.conv1(x_dict, edge_index_dict)

        x_dict = {
            key: F.relu(x)
            for key, x in x_dict.items()
        }

        x_dict = self.conv2(x_dict, edge_index_dict)

        return x_dict