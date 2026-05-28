#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:31:04 2026

@author: angelosagnori
"""

#%% Instalando os pacotes
 
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

from sqlalchemy import create_engine

import torch 
import torch_geometric
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.nn import GraphConv, to_hetero
import torch.nn.functional as F
from torch_geometric.data import HeteroData


import sklearn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score, average_precision_score
import xgboost as xgb

# Print Versions
print(f"PyTorch: {torch.__version__}")
print(f"PyG: {torch_geometric.__version__}")
print(f"XGBoost: {xgboost.__version__}")
print(f"Scikit-Learn: {sklearn.__version__}")


#%% 1 Configuração 
# --- 1.1 Configuração de Hardware (Essencial para Mac M1/M2/M3) ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Usando GPU Apple (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Usando GPU NVIDIA")
else:
    device = torch.device("cpu")
    print("Usando CPU")
    
# 1.2 Conexão com o Banco de Dados
config = {
    'user': 'root',
    'password': 'Vitoria%402526',
    'host': 'localhost:3306',
    'database': 'bike_store'
}

engine = create_engine(
    "mysql+pymysql://root:Vitoria%402526@localhost:3306/bike_store"
)

#%%
# 2. Extração das Tabelas
def load_data(engine):
    tables = ['customers', 'orders', 'order_items', 'products', 
              'staffs', 'stores', 'brands', 'categories']
    data = {}
    for t in tables:
        data[t] = pd.read_sql(f"SELECT * FROM {t}", engine)
        print(f"Tabela {t} carregada: {len(data[t])} linhas.")
    return data

df_dict = load_data(engine)

# --- 1. Configuração de Hardware (Essencial para Mac M1/M2/M3) ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Usando GPU Apple (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Usando GPU NVIDIA")
else:
    device = torch.device("cpu")
    print("Usando CPU")

# --- 2. Preparação do Objeto HeteroData ---
data = HeteroData()

# --- 3. Mapeamento de IDs ---
def get_mapping(df, id_col):
    return {old_id: new_id for new_id, old_id in enumerate(df[id_col].unique())}

maps = {
    'customer': get_mapping(df_dict['customers'], 'customer_id'),
    'order': get_mapping(df_dict['orders'], 'order_id'),
    'product': get_mapping(df_dict['products'], 'product_id'),
    'brand': get_mapping(df_dict['brands'], 'brand_id'),
}

# Criando o objeto principal
data = HeteroData()

# Feature Clientes: One-hot do estado
cust_features = pd.get_dummies(df_dict['customers']['state']).astype(float)
data['customer'].x = torch.tensor(cust_features.values, dtype=torch.float)

# Feature Produtos: Preço normalizado
price = df_dict['products']['list_price']
price_norm = (price - price.min()) / (price.max() - price.min())
data['product'].x = torch.tensor(price_norm.values.reshape(-1, 1), dtype=torch.float)

# Feature Marcas: Identidade (Matriz identidade como placeholder)
data['brand'].x = torch.eye(len(df_dict['brands']), dtype=torch.float)

# Feature Pedidos: Mês do pedido + Target (y)
df_orders = df_dict['orders'].copy()
df_orders['y'] = (df_orders['shipped_date'] > df_orders['required_date']).astype(int)
data['order'].y = torch.tensor(df_orders['y'].values, dtype=torch.long)

order_month = pd.to_datetime(df_orders['order_date']).dt.month
data['order'].x = torch.tensor(pd.get_dummies(order_month).values, dtype=torch.float)

# Split Temporal
order_dates = pd.to_datetime(df_orders['order_date'])
data['order'].train_mask = torch.tensor((order_dates < '2018-01-01').values, dtype=torch.bool)
data['order'].test_mask = torch.tensor((order_dates >= '2018-01-01').values, dtype=torch.bool)


# 4. Definição das Arestas (Fluxo para o Pedido)

def create_edges(df, src_col, dst_col, src_map, dst_map):
    src_idx = [src_map[i] for i in df[src_col]]
    dst_idx = [dst_map[i] for i in df[dst_col]]
    return torch.tensor([src_idx, dst_idx], dtype=torch.long)

# Criando as relações onde 'order' e 'product' RECEBEM informação
data['customer', 'to', 'order'].edge_index = create_edges(
    df_dict['orders'], 'customer_id', 'order_id', maps['customer'], maps['order'])

data['product', 'to', 'order'].edge_index = create_edges(
    df_dict['order_items'], 'product_id', 'order_id', maps['product'], maps['order'])

data['brand', 'to', 'product'].edge_index = create_edges(
    df_dict['products'], 'brand_id', 'product_id', maps['brand'], maps['product'])

print("--- Grafo Heterogêneo Preparado ---")
print(data)

# Gerando o esquema visual do Grafo  
# Relações ('customer' > 'order'), etc
schema_path = "esquema_grafo_bike_store.png"

try:
    G = nx.DiGraph()
    # Adiciona relações do HeteroData
    for src, rel, dst in data.edge_types:
        G.add_edge(src, dst, label=rel)

    # Layout do grafo
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(10, 7))

    # Nós + arestas
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=3500,
        font_size=10,
        arrows=True
    )

    # Labels das relações
    edge_labels = nx.get_edge_attributes(G, "label")

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels
    )

    plt.title("Schema do Grafo Heterogêneo")
    plt.savefig(schema_path, bbox_inches="tight")
    plt.show()

    print(f"Sucesso! O diagrama do esquema foi salvo em: {schema_path}")

except Exception as e:
    print(f"Erro ao gerar visual: {e}")
    print("Dica: pip install networkx matplotlib")

# 5. Arquitetura do Modelo

class GNNModel(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        
        # HeteroConv manual: Aqui controlamos exatamente quem fala com quem
        self.conv1 = HeteroConv({
            # Mensagens de vizinhos
            ('customer', 'to', 'order'): SAGEConv((-1, -1), hidden_channels),
            ('product', 'to', 'order'): SAGEConv((-1, -1), hidden_channels),
            ('brand', 'to', 'product'): SAGEConv((-1, -1), hidden_channels),
            
            # O SEGREDO: Self-loops manuais para os nós não "sumirem"
            ('customer', 'to', 'customer'): SAGEConv((-1, -1), hidden_channels),
            ('brand', 'to', 'brand'): SAGEConv((-1, -1), hidden_channels),
            ('order', 'to', 'order'): SAGEConv((-1, -1), hidden_channels),
            ('product', 'to', 'product'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('customer', 'to', 'order'): SAGEConv((-1, -1), out_channels),
            ('product', 'to', 'order'): SAGEConv((-1, -1), out_channels),
            ('brand', 'to', 'product'): SAGEConv((-1, -1), out_channels),
            
            # Mantendo a estrutura na segunda camada
            ('customer', 'to', 'customer'): SAGEConv((-1, -1), out_channels),
            ('brand', 'to', 'brand'): SAGEConv((-1, -1), out_channels),
            ('order', 'to', 'order'): SAGEConv((-1, -1), out_channels),
            ('product', 'to', 'product'): SAGEConv((-1, -1), out_channels),
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        # Primeira camada
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        # Segunda camada
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

# Preparação do edge_index_dict com Self-Loops
# Criar arestas que ligam o nó a ele mesmo: [0, 1, 2] > [0, 1, 2]
for node_type in data.node_types:
    num_nodes = data[node_type].x.shape[0]
    indices = torch.arange(num_nodes, dtype=torch.long)
    data[node_type, 'to', node_type].edge_index = torch.stack([indices, indices], dim=0)

model = GNNModel(hidden_channels=64, out_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Atualiza os dicionários para o treino
x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
y_true = data['order'].y.to(device)
mask = data['order'].train_mask.to(device)

# 1. Calcule o peso das classes (Inverso da frequência)
# Se ~5 vezes mais pedidos no prazo, dar peso 5 para o atraso.
# weights = torch.tensor([1.0, 5.0]).to(device)
weights = torch.tensor([1.0, 2.5]).to(device)

def train():
    model.train()
    optimizer.zero_grad()
    out_dict = model(x_dict, edge_index_dict)
#    loss = F.cross_entropy(out_dict['order'][mask], y_true[mask])
    loss = F.cross_entropy(out_dict['order'][mask], y_true[mask], weight=weights)
    loss.backward()
    optimizer.step()
    return float(loss)

print("Iniciando Treinamento com Identidade de Nós...")
for epoch in range(1, 101):
    loss = train()
    if epoch % 10 == 0:
        print(f'Época: {epoch:03d}, Perda: {loss:.4f}')
        
#%% Avaliação e Métricas
# Gerar a Acurácia e a Matriz de Confusão.
def evaluate():
    model.eval()
    with torch.no_grad():
        # 1. Garante que os dados de entrada estão no device (GPU)
        x_dict_gpu = {k: v.to(device) for k, v in data.x_dict.items()}
        edge_index_dict_gpu = {k: v.to(device) for k, v in data.edge_index_dict.items()}
        
        # 2. Passada do modelo
        out_dict = model(x_dict_gpu, edge_index_dict_gpu)
        
        # 3. Prepara a máscara e o target no mesmo device
        mask = data['order'].test_mask.to(device)
        target = data['order'].y.to(device) # Garante que o Y também vá para a GPU
        
        # 4. Filtra os resultados de teste
        logits = out_dict['order'][mask]
        
        # 5. Converte para classes e move para CPU para o Scikit-Learn
        preds = logits.argmax(dim=-1).cpu().numpy()
        y_true = target[mask].cpu().numpy()
        
    return y_true, preds

#&&
# Execução e Visualização
try:
    y_true, y_pred = evaluate()

    acc = accuracy_score(y_true, y_pred)
    print(f"\n--- Resultados Finais (Base de Teste 2018) ---")
    print(f"Acurácia Geral: {acc:.2%}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_true, y_pred, target_names=['No Prazo', 'Atrasado'], zero_division=0))

    # Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['No Prazo', 'Atrasado'], 
                yticklabels=['No Prazo', 'Atrasado'])
    plt.xlabel('Predição (Modelo)')
    plt.ylabel('Realidade (Banco de Dados)')
    plt.title('Matriz de Confusão - Predição de Atrasos Bike Store')
    plt.show()

except Exception as e:
    print(f"Erro na avaliação: {e}")

