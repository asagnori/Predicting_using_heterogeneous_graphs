#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 16:19:12 2026

@author: angelosagnori
"""

#%% Instalando os pacotes

import sys
import pandas as pd
import seaborn as sns
#import mysql.connector
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

print(sys.executable)

#%%
# 1. Conexão com o Banco de Dados
config = {
    'user': 'root',
    'password': 'Vitoria%402526',
    'host': 'localhost:3306',
    'database': 'bike_store'
}

engine = create_engine(
    "mysql+pymysql://root:Vitoria%402526@localhost:3306/bike_store"
)

# teste simples
query = "SELECT * FROM orders"

df = pd.read_sql(query, engine)

print(df.head())
print(df.info())

#%%
# 2. Extração Simples (pd.read_sql)
df_orders = pd.read_sql("SELECT * FROM orders", engine)
df_items  = pd.read_sql("SELECT * FROM order_items", engine)
df_products = pd.read_sql("SELECT * FROM products", engine)
df_stocks = pd.read_sql("SELECT * FROM stocks", engine)
df_customers = pd.read_sql("SELECT * FROM customers", engine)
df_stores = pd.read_sql("SELECT * FROM stores", engine)
df_staffs = pd.read_sql("SELECT * FROM staffs", engine)
df_categories = pd.read_sql("SELECT * FROM categories", engine)
df_brands = pd.read_sql("SELECT * FROM brands", engine)

# --- ANÁLISE DE VOLUMETRIA (O seu COUNT*) ---
print(f"Total de Pedidos: {len(df_orders)}")
print(f"Total de Itens: {len(df_items)}")
print(f"Total de Produtos: {len(df_products)}")
print(f"Total de Estoque: {len(df_stocks)}")
print(f"Total de Customers: {len(df_customers)}")
print(f"Total de Stores: {len(df_stores)}")
print(f"Total de Vendedores: {len(df_staffs)}")
print(f"Total de Categorias: {len(df_categories)}")
print(f"Total de Marcas: {len(df_brands)}")

#%%
# --- CÁLCULO DO N TOTAL (População de Nós) ---
n_total = (len(df_customers) + len(df_orders) + len(df_items) + 
           len(df_products) + len(df_stores) + len(df_staffs) + 
           len(df_brands) + len(df_categories))

print(f"--- VOLUMETRIA DE NÓS (N) ---")
print(f"Total de Nós (N): {n_total}")
print("-" * 30)

#%%
# --- CÁLCULO DE ATRASO (Baseado nas datas do DataFrame) ---
# Convertendo para datetime caso não venha formatado
df_orders['required_date'] = pd.to_datetime(df_orders['required_date'])
df_orders['shipped_date'] = pd.to_datetime(df_orders['shipped_date'])

# Criando a coluna de Atraso: 1 para atrasado, 0 para no prazo
# Consideramos atraso se a data de envio for MAIOR que a data requerida
df_orders['atraso'] = (df_orders['shipped_date'] > df_orders['required_date']).astype(int)

# Filtrando apenas pedidos que já foram enviados (igual ao seu WHERE no SQL)
df_enviados = df_orders.dropna(subset=['shipped_date'])

total_enviados = len(df_enviados)
atrasados = len(df_enviados[df_enviados['shipped_date'] > df_enviados['required_date']])

# Atraso = Data de Envio > Data Requerida
total_pedidos = len(df_orders)
no_prazo = total_pedidos - atrasados
    
percentual_atraso = (atrasados / total_pedidos) * 100
percentual_correto = (atrasados / total_enviados) * 100

print(f"--- CÁLCULO DE ATRASO ---")
print(f"Total de Pedidos: {total_pedidos}")
print(f"Pedidos em Atraso: {atrasados}")
print(f"Pedidos no Prazo: {no_prazo}")
print(f"Percentual de Atraso: {percentual_atraso:.2f}%")
print(f"Percentual Correto sobre Enviados: {percentual_correto:.2f}%")

#%%
# --- CÁLCULO DO E TOTAL (Arestas/Relacionamentos) ---
# Seguindo a lógica do seu SQL de somar as Foreign Keys
e_total = (
    df_orders['customer_id'].count() + 
    df_orders['store_id'].count() + 
    df_orders['staff_id'].count() + 
    df_items['order_id'].count() + 
    df_items['product_id'].count() + 
    df_products['brand_id'].count() + 
    df_products['category_id'].count() + 
    len(df_stocks) # Cada linha de stock liga 1 loja a 1 produto
)

print(f"--- DENSIDADE DE ARESTAS (E) ---")
print(f"Total de Arestas (E): {e_total}")
print("-" * 30)

#%%
# --- VISUALIZAÇÃO DOS ATRASOS ---
plt.figure(figsize=(8, 5))
sns.countplot(x='atraso', data=df_orders, palette='RdYlGn_r')
plt.title('Distribuição de Atrasos (0=No Prazo, 1=Atrasado)')
plt.xticks([0, 1], ['No Prazo', 'Atrasado'])
plt.show()

#%%
# --- ANÁLISE DE CONECTIVIDADE (Exemplo: Itens por Pedido) ---
itens_por_pedido = df_items.groupby('order_id').size()
print(f"Média de itens por pedido: {itens_por_pedido.mean():.2f}")

plt.figure(figsize=(10, 5))
sns.histplot(itens_por_pedido, kde=True, color='blue')
plt.title('Distribuição de Itens por Pedido (Conectividade do Grafo)')
plt.xlabel('Quantidade de Itens')
plt.show()

