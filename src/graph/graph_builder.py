import torch
from torch_geometric.data import HeteroData

from graph.mappings import (
    get_mapping,
    create_edges
)

from data.feature_engineering import create_features


def build_graph(df_dict):

    data = HeteroData()

    # =====================================================
    # Mappings
    # =====================================================

    maps = {
        'customer': get_mapping(df_dict['customers'], 'customer_id'),
        'order': get_mapping(df_dict['orders'], 'order_id'),
        'product': get_mapping(df_dict['products'], 'product_id'),
        'brand': get_mapping(df_dict['brands'], 'brand_id'),
        'category': get_mapping(df_dict['categories'], 'category_id'),
        'store': get_mapping(df_dict['stores'], 'store_id'),
        'staff': get_mapping(df_dict['staffs'], 'staff_id')
    }

    # =====================================================
    # Features
    # =====================================================

    data, df_orders = create_features(df_dict, data)

    # =====================================================
    # Temporal Split
    # =====================================================

    order_dates = pd.to_datetime(df_orders['order_date'])

    data['order'].train_mask = torch.tensor(
        (order_dates < '2018-01-01').values,
        dtype=torch.bool
    )

    data['order'].test_mask = torch.tensor(
        (order_dates >= '2018-01-01').values,
        dtype=torch.bool
    )

    # =====================================================
    # Relations
    # =====================================================

    data['customer', 'to', 'order'].edge_index = create_edges(
        df_dict['orders'],
        'customer_id',
        'order_id',
        maps['customer'],
        maps['order']
    )

    data['product', 'to', 'order'].edge_index = create_edges(
        df_dict['order_items'],
        'product_id',
        'order_id',
        maps['product'],
        maps['order']
    )

    data['brand', 'to', 'product'].edge_index = create_edges(
        df_dict['products'],
        'brand_id',
        'product_id',
        maps['brand'],
        maps['product']
    )

    data['category', 'to', 'product'].edge_index = create_edges(
        df_dict['products'],
        'category_id',
        'product_id',
        maps['category'],
        maps['product']
    )

    data['store', 'to', 'order'].edge_index = create_edges(
        df_dict['orders'],
        'store_id',
        'order_id',
        maps['store'],
        maps['order']
    )

    data['staff', 'to', 'order'].edge_index = create_edges(
        df_dict['orders'],
        'staff_id',
        'order_id',
        maps['staff'],
        maps['order']
    )

    # =====================================================
    # Self Loops
    # =====================================================

    for node_type in data.node_types:
        num_nodes = data[node_type].x.shape[0]
        indices = torch.arange(
            num_nodes,
            dtype=torch.long
        )

        data[node_type, 'to', node_type].edge_index = torch.stack(
            [indices, indices],
            dim=0
        )
    return data, df_orders