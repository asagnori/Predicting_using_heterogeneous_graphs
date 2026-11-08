"""Feature engineering for heterogeneous graph nodes."""

from __future__ import annotations

import pandas as pd
import torch
from torch_geometric.data import HeteroData


def create_features(
    df_dict: dict[str, pd.DataFrame],
    data: HeteroData,
) -> tuple[HeteroData, pd.DataFrame]:
    """Create node attributes and labels following the thesis experiment."""
    df_orders = df_dict["orders"].copy()

    customer_states = pd.get_dummies(
        df_dict["customers"]["state"],
        dtype=float,
    )
    data["customer"].x = torch.tensor(
        customer_states.to_numpy(),
        dtype=torch.float,
    )

    prices = df_dict["products"]["list_price"].astype(float)
    price_range = prices.max() - prices.min()
    price_norm = (
        (prices - prices.min()) / price_range
        if price_range
        else prices * 0
    )
    data["product"].x = torch.tensor(
        price_norm.to_numpy().reshape(-1, 1),
        dtype=torch.float,
    )

    table_by_node_type = {
        "brand": "brands",
        "category": "categories",
        "store": "stores",
        "staff": "staffs",
    }
    for node_type, table_name in table_by_node_type.items():
        data[node_type].x = torch.eye(
            len(df_dict[table_name]),
            dtype=torch.float,
        )

    order_month = pd.to_datetime(df_orders["order_date"]).dt.month
    order_features = pd.get_dummies(order_month, dtype=float)
    data["order"].x = torch.tensor(
        order_features.to_numpy(),
        dtype=torch.float,
    )
    data["order"].y = torch.tensor(
        df_orders["y"].to_numpy(),
        dtype=torch.long,
    )
    return data, df_orders
