from data.loader import load_data
from graph.graph_builder import build_graph
from sqlalchemy import create_engine
from config.settings import DATABASE_CONFIG

config = DATABASE_CONFIG

engine = create_engine(
    f"mysql+pymysql://"
    f"{config['user']}:"
    f"{config['password']}@"
    f"{config['host']}/"
    f"{config['database']}"
)

# conexão
df_dict = load_data(engine)

# build grafo
data, df_orders = build_graph(df_dict)

print(data)