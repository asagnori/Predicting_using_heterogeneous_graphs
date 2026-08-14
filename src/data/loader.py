import pandas as pd

def load_data(engine):

    tables = [
        'customers',
        'orders',
        'order_items',
        'products',
        'staffs',
        'stores',
        'brands',
        'categories'
    ]

    data = {}

    for t in tables:
        data[t] = pd.read_sql(f"SELECT * FROM {t}", engine)
        print(f"Tabela {t} carregada: {len(data[t])} linhas.")

    return data