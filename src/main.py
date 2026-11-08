from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from src.config.settings import DATABASE_CONFIG
from src.data.loader import load_data
from src.graph.graph_builder import build_graph


def create_database_engine():
    url = URL.create(
        "mysql+pymysql",
        username=DATABASE_CONFIG["user"],
        password=DATABASE_CONFIG["password"],
        host=DATABASE_CONFIG["host"].split(":")[0],
        port=int(DATABASE_CONFIG["port"])
        if DATABASE_CONFIG["port"]
        else (
            int(DATABASE_CONFIG["host"].split(":")[1])
            if ":" in DATABASE_CONFIG["host"]
            else None
        ),
        database=DATABASE_CONFIG["database"],
    )
    return create_engine(url)


def main() -> None:
    engine = create_database_engine()
    df_dict = load_data(engine)
    data, df_orders, maps = build_graph(df_dict)
    print(data)
    print(f"Mapeamentos criados: {', '.join(maps)}")
    print(f"Pedidos preparados: {len(df_orders)}")


if __name__ == "__main__":
    main()