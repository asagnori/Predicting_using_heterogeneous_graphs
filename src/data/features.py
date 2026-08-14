import pandas as pd
import torch

def create_features(df_dict, data):

    # =====================================================
    # Customer Features
    # =====================================================

    cust_features = pd.get_dummies(
        df_dict['customers']['state']
    ).astype(float)

    data['customer'].x = torch.tensor(
        cust_features.values,
        dtype=torch.float
    )

    # =====================================================
    # Product Features
    # =====================================================

    price = df_dict['products']['list_price']

    price_norm = (
        (price - price.min()) /
        (price.max() - price.min())
    )

    data['product'].x = torch.tensor(
        price_norm.values.reshape(-1, 1),
        dtype=torch.float
    )

    # =====================================================
    # Brand Features
    # =====================================================

    data['brand'].x = torch.eye(
        len(df_dict['brands']),
        dtype=torch.float
    )

    # =====================================================
    # Order Features
    # =====================================================

    df_orders = df_dict['orders'].copy()

    df_orders['y'] = (
        df_orders['shipped_date'] >
        df_orders['required_date']
    ).astype(int)

    data['order'].y = torch.tensor(
        df_orders['y'].values,
        dtype=torch.long
    )

    order_month = pd.to_datetime(
        df_orders['order_date']
    ).dt.month

    data['order'].x = torch.tensor(
        pd.get_dummies(order_month).values,
        dtype=torch.float
    )

    return data, df_orders