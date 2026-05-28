#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 16:49:54 2026

@author: angelosagnori
"""

#%% Instalando os pacotes
 
import pandas as pd
import numpy as np
import torch 
import torch_geometric
from torch_geometric.data import HeteroData
#import mysql.connector
from sqlalchemy import create_engine


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

#%%
# 3. Verificando Split Temporal

df_orders = df_dict['orders']
df_orders['order_date'] = pd.to_datetime(df_orders['order_date'])

# Verificando a distribuição para decidir o corte
print(df_orders['order_date'].dt.year.value_counts().sort_index())

# Exemplo de Split: Treino até 2017, Teste 2018
data_corte = '2018-01-01'

train_orders = df_orders[df_orders['order_date'] < data_corte].copy()
test_orders = df_orders[df_orders['order_date'] >= data_corte].copy()

print(f"Pedidos para Treino: {len(train_orders)}")
print(f"Pedidos para Teste: {len(test_orders)}")

#%%
# 4. Engenharia de Features e Definição do Grafo

# Definindo o atraso (Target) apenas para os pedidos
# E filtro 'shipped_date IS NOT NULL' 
train_orders = train_orders.dropna(subset=['shipped_date'])
train_orders['y'] = (train_orders['shipped_date'] > train_orders['required_date']).astype(int)

# --- 1. Preparação do Objeto HeteroData ---
data = HeteroData()

# Exemplo: Adicionando nós de Clientes (Features básicas: One-hot do estado/cidade)
# Aqui você usaria o pd.get_dummies() para transformar categorias em números
customer_features = pd.get_dummies(df_dict['customers'][['city', 'state']])
data['customer'].x = torch.tensor(customer_features.values, dtype=torch.float)

# Exemplo: Adicionando arestas Pedido -> Cliente
# Você precisa criar um tensor [2, Num_Arestas] com os índices
print(data)

# --- 2. Função Auxiliar para Mapeamento de IDs ---
# O PyG precisa que os IDs comecem em 0 para cada tipo de nó
def get_mapping(df, id_col):
    return {old_id: new_id for new_id, old_id in enumerate(df[id_col].unique())}

# Criando os mapeamentos baseados em TODAS as tabelas
maps = {
    'customer': get_mapping(df_dict['customers'], 'customer_id'),
    'order': get_mapping(df_dict['orders'], 'order_id'),
    'product': get_mapping(df_dict['products'], 'product_id'),
    'store': get_mapping(df_dict['stores'], 'store_id'),
    'staff': get_mapping(df_dict['staffs'], 'staff_id'),
    'brand': get_mapping(df_dict['brands'], 'brand_id'),
    'category': get_mapping(df_dict['categories'], 'category_id')
}

# --- 3. Definindo os Nós (Features) ---
# Dica: No MBA, o orientador quer ver "Feature Engineering"
# Para Clientes: One-hot do estado
cust_features = pd.get_dummies(df_dict['customers']['state']).astype(float)
data['customer'].x = torch.tensor(cust_features.values, dtype=torch.float)

# Para Produtos: Preço normalizado
price = df_dict['products']['list_price']
price_norm = (price - price.min()) / (price.max() - price.min())
data['product'].x = torch.tensor(price_norm.values.reshape(-1, 1), dtype=torch.float)

# Para Pedidos (Target): Definindo o Y (Atraso)
# Lembre-se: shipped_date > required_date = 1 (atrasado)
df_orders = df_dict['orders'].copy()
df_orders['y'] = (df_orders['shipped_date'] > df_orders['required_date']).astype(int)
data['order'].y = torch.tensor(df_orders['y'].values, dtype=torch.long)
# Feature de Pedido: mês do pedido
order_month = pd.to_datetime(df_orders['order_date']).dt.month
data['order'].x = torch.tensor(pd.get_dummies(order_month).values, dtype=torch.float)

# --- 4. Definindo as Arestas (Conexões) ---
def create_edges(df, src_col, dst_col, src_map, dst_map):
    src_idx = [src_map[i] for i in df[src_col]]
    dst_idx = [dst_map[i] for i in df[dst_col]]
    return torch.tensor([src_idx, dst_idx], dtype=torch.long)

# Pedido -> Cliente
data['order', 'bought_by', 'customer'].edge_index = create_edges(
    df_dict['orders'], 'order_id', 'customer_id', maps['order'], maps['customer'])

# Itens -> Pedido e Itens -> Produto
data['order', 'contains', 'product'].edge_index = create_edges(
    df_dict['order_items'], 'order_id', 'product_id', maps['order'], maps['product'])

# Produto -> Marca e Produto -> Categoria
data['product', 'belongs_to', 'brand'].edge_index = create_edges(
    df_dict['products'], 'product_id', 'brand_id', maps['product'], maps['brand'])

# --- 5. Aplicando o Split Temporal (Ponto chave do orientador) ---
order_dates = pd.to_datetime(df_dict['orders']['order_date'])
data['order'].train_mask = torch.tensor((order_dates < '2018-01-01').values, dtype=torch.bool)
data['order'].test_mask = torch.tensor((order_dates >= '2018-01-01').values, dtype=torch.bool)

print("Estrutura do Grafo Heterogêneo:")
print(data)

# Criando máscaras booleanas baseadas no ano (Corte em 2018)
order_dates = pd.to_datetime(df_dict['orders']['order_date'])
train_mask = order_dates < '2018-01-01'
test_mask = order_dates >= '2018-01-01'

data['order'].train_mask = torch.tensor(train_mask.values, dtype=torch.bool)
data['order'].test_mask = torch.tensor(test_mask.values, dtype=torch.bool)

print(f"Nós de treino: {data['order'].train_mask.sum()}")
print(f"Nós de teste: {data['order'].test_mask.sum()}")

