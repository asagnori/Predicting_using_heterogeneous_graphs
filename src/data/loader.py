import pandas as pd
from sqlalchemy.engine import Engine


TABLES = (
    "customers",
    "orders",
    "order_items",
    "products",
    "staffs",
    "stores",
    "brands",
    "categories",
)


def load_data(engine: Engine) -> dict[str, pd.DataFrame]:

    data = {
        table: pd.read_sql_table(table, engine)
        for table in TABLES
    }
    for table, frame in data.items():
        print(f"Tabela {table} carregada: {len(frame)} linhas.")
    return data


def prepare_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Create the delay target and return orders in chronological order."""
    prepared = orders.copy()
    prepared["order_date"] = pd.to_datetime(prepared["order_date"])
    prepared["required_date"] = pd.to_datetime(prepared["required_date"])
    prepared["shipped_date"] = pd.to_datetime(prepared["shipped_date"])
    prepared["y"] = (
        prepared["shipped_date"] > prepared["required_date"]
    ).astype("int64")
    return prepared.sort_values("order_date").reset_index(drop=True)