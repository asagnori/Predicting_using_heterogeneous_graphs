#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 15:20:32 2026

@author: angelosagnori
"""

#%% 1 Configuração 
#   1.1 Instalando os pacotes

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

import shap
 
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig, ModelMode, ModelTaskLevel, ModelReturnType

import xgboost as xgb
from xgboost import XGBClassifier

import sklearn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import sqlalchemy
from sqlalchemy import create_engine

from scipy.stats import ttest_rel
from importlib.metadata import version

import warnings
warnings.filterwarnings('ignore')

config= {
'user': '',
'password': '',
'host': 'localhost:3306',
'database': 'bike_store'
}

engine = create_engine(
    f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}/{config['database']}"
)

# Print Versions
print(f"PyTorch: {torch.__version__}")
print(f"PyG: {torch_geometric.__version__}")
print(f"XGBoost: {xgb.__version__}")
print(f"Scikit-Learn: {sklearn.__version__}")

#   1.2 Configuração de Hardware (Essencial para Mac M1/M2/M3) ---

# Configurações de ambiente
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Usando GPU Apple (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Usando GPU NVIDIA")
else:
    device = torch.device("cpu")
    print("Usando CPU")
    
#   1.3 Conexão com o Banco de Dados

config= {
'user': 'root',
'password': 'Vitoria%402526',
'host': 'localhost:3306',
'database': 'bike_store'
}

engine = create_engine(
    f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}/{config['database']}"
)

#%% 2 Obter e preparar os Dados
#   2.1 Extração das Tabelas

def load_data(engine):
    tables = ['customers', 'orders', 'order_items', 'products', 'staffs', 'stores', 'brands', 'categories']
    data_dict = {}
    for t in tables:
        data_dict[t] = pd.read_sql(f"SELECT * FROM {t}", engine)
        print(f"Tabela {t} carregada: {len(data_dict[t])} linhas.")
    return data_dict

df_dict = load_data(engine)

#   2.2 Preparação do Objeto HeteroData ---
data = HeteroData()

#   2.3 Mapeamento de IDs ---
def get_mapping(df, id_col):
    return {old_id: new_id for new_id, old_id in enumerate(df[id_col].unique())}

# Mapeamento completo de IDs para alinhar com a EDA teórica
maps = {
    'customer': get_mapping(df_dict['customers'], 'customer_id'),
    'order': get_mapping(df_dict['orders'], 'order_id'),
    'product': get_mapping(df_dict['products'], 'product_id'),
    'brand': get_mapping(df_dict['brands'], 'brand_id'),
    'category': get_mapping(df_dict['categories'], 'category_id'),
    'store': get_mapping(df_dict['stores'], 'store_id'),
    'staff': get_mapping(df_dict['staffs'], 'staff_id')
}

# %% 1.1 CARREGAMENTO E ORDENAÇÃO CRONOLÓGICA (Prevenção Estrita de Data Leakage)

def load_and_sort_data(engine):
    tables = ['customers', 'orders', 'order_items', 'products', 'staffs', 'stores', 'brands', 'categories']
    df_dict = {t: pd.read_sql(f"SELECT * FROM {t}", engine) for t in tables}
    
    # IMPORTANTE: Ordenar ordens cronologicamente para o Time Series Split
    df_dict['orders']['y'] = (df_dict['orders']['shipped_date'] > df_dict['orders']['required_date']).astype(int)
    df_dict['orders']['order_date'] = pd.to_datetime(df_dict['orders']['order_date'])
    df_dict['orders'] = df_dict['orders'].sort_values('order_date').reset_index(drop=True)
    return df_dict

df_dict = load_and_sort_data(engine)

#   2.4 Criando o objeto principal
data = HeteroData()

# Mapeamentos de IDs
def get_mapping(df, id_col):
    return {old_id: new_id for new_id, old_id in enumerate(df[id_col].unique())}

maps = {
    'customer': get_mapping(df_dict['customers'], 'customer_id'),
    'order': {old_id: new_id for new_id, old_id in enumerate(df_dict['orders']['order_id'])}, # Alinhado com ordenação
    'product': get_mapping(df_dict['products'], 'product_id'),
    'brand': get_mapping(df_dict['brands'], 'brand_id'),
    'category': get_mapping(df_dict['categories'], 'category_id'),
    'store': get_mapping(df_dict['stores'], 'store_id'),
    'staff': get_mapping(df_dict['staffs'], 'staff_id')
}

#   2.5 Features
# %% Construçao do Grafo e Features dos Nós (Node Features)

cust_features = pd.get_dummies(df_dict['customers']['state']).astype(float)
data['customer'].x = torch.tensor(cust_features.values, dtype=torch.float)

price = df_dict['products']['list_price']
price_norm = (price - price.min()) / (price.max() - price.min())
data['product'].x = torch.tensor(price_norm.values.reshape(-1, 1), dtype=torch.float)

data['brand'].x = torch.eye(len(df_dict['brands']), dtype=torch.float)
data['category'].x = torch.eye(len(df_dict['categories']), dtype=torch.float)
data['store'].x = torch.eye(len(df_dict['stores']), dtype=torch.float)
data['staff'].x = torch.eye(len(df_dict['staffs']), dtype=torch.float)

order_month = df_dict['orders']['order_date'].dt.month
data['order'].x = torch.tensor(pd.get_dummies(order_month).values, dtype=torch.float)
data['order'].y = torch.tensor(df_dict['orders']['y'].values, dtype=torch.long)

# Arestas
def create_edges(df, src_col, dst_col, src_map, dst_map):
    return torch.tensor([[src_map[src], dst_map[dst]] for src, dst in zip(df[src_col], df[dst_col])], dtype=torch.long).t()

data['customer', 'to', 'order'].edge_index = create_edges(df_dict['orders'], 'customer_id', 'order_id', maps['customer'], maps['order'])
data['product', 'to', 'order'].edge_index = create_edges(df_dict['order_items'], 'product_id', 'order_id', maps['product'], maps['order'])
data['brand', 'to', 'product'].edge_index = create_edges(df_dict['products'], 'brand_id', 'product_id', maps['brand'], maps['product'])
data['category', 'to', 'product'].edge_index = create_edges(df_dict['products'], 'category_id', 'product_id', maps['category'], maps['product'])
data['store', 'to', 'order'].edge_index = create_edges(df_dict['orders'], 'store_id', 'order_id', maps['store'], maps['order'])
data['staff', 'to', 'order'].edge_index = create_edges(df_dict['orders'], 'staff_id', 'order_id', maps['staff'], maps['order'])

# Inserção de Auto-loops para estabilização de representação dos nós
for node_type in data.node_types:
    num_nodes = data[node_type].x.shape[0]
    indices = torch.arange(num_nodes, dtype=torch.long)
    data[node_type, 'to', node_type].edge_index = torch.stack([indices, indices], dim=0)

print("--- Grafo Heterogêneo Consolidado ---")
print(data)

# %% 3. Modelo GNN Dinamico (GraphSAGE) para Grafos Heterogêneos Corrigido para Nós Fonte/Destino)

class GNNModel(torch.nn.Module):
    def __init__(self, edge_types, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = HeteroConv({et: SAGEConv((-1, -1), hidden_channels) for et in edge_types}, aggr='sum')
        self.conv2 = HeteroConv({et: SAGEConv((-1, -1), out_channels) for et in edge_types}, aggr='sum')

    # "saltos" (hop) de distância no grafo logístico:
    def forward(self, x_dict, edge_index_dict):
        # 1ª Convolução
        out1 = self.conv1(x_dict, edge_index_dict)
        
        # Se o nó recebeu mensagem, aplica ReLU. Se for nó de origem pura (ex: customer), mantém o x original
        x_dict_l1 = {
            k: F.relu(out1[k]) if k in out1 else x_dict[k]
            for k in x_dict.keys()
        }
        
        # 2ª Convolução
        out2 = self.conv2(x_dict_l1, edge_index_dict)
        return out2

# %% 4. Pipeline de validaçao cruzada temporal (Time Series Split) para avaliação comparativa entre XGBoost e GNN

tscv = TimeSeriesSplit(n_splits=4)

# 4.1 Preparação da Base Tabular para o XGBoost (com dados categóricos e required_month)
df_orders = df_dict['orders'].copy()
df_orders['order_month'] = pd.to_datetime(df_orders['order_date']).dt.month.astype('category')
df_orders['required_month'] = pd.to_datetime(df_orders['required_date']).dt.month.astype('category')

X_xgb = pd.DataFrame({
    'customer_id': df_orders['customer_id'].astype('category'),
    'store_id': df_orders['store_id'].astype('category'),
    'staff_id': df_orders['staff_id'].astype('category'),
    'order_month': df_orders['order_month'],
    'required_month': df_orders['required_month']
})
y_xgb = df_orders['y']

x_dict_gpu = {k: v.to(device) for k, v in data.x_dict.items()}
edge_index_dict_gpu = {k: v.to(device) for k, v in data.edge_index_dict.items()}

# Estruturas para armazenar métricas acumuladas por fold
folds_metrics = []
folds_gnn_f1_atraso = []
folds_xgb_f1_atraso = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(df_orders)):
    print(f"\n>>> Processando Fold Temporal {fold + 1}/4...")
    
    # Máscaras de Treino e Validação do Fold
    train_idx_tensor = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx_tensor = torch.tensor(val_idx, dtype=torch.long, device=device)
    
    train_mask = torch.zeros(len(df_orders), dtype=torch.bool, device=device)
    train_mask[train_idx_tensor] = True
    
    val_mask = torch.zeros(len(df_orders), dtype=torch.bool, device=device)
    val_mask[val_idx_tensor] = True
    
    y_train_fold = df_orders.loc[train_idx, 'y'].values
    y_val_fold = df_orders.loc[val_idx, 'y'].values
    
    # Reponderação dinâmica do desbalanceamento por fold
    weight_pos = np.sum(y_train_fold == 0) / np.sum(y_train_fold == 1)
    
    # --- Treinamento GraphSAGE GNN ---
    gnn_weights = torch.tensor([1.0, weight_pos], dtype=torch.float).to(device)
    model = GNNModel(data.edge_types, hidden_channels=64, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(1, 51):
        model.train()
        optimizer.zero_grad()
        out = model(x_dict_gpu, edge_index_dict_gpu)
        loss = F.cross_entropy(out['order'][train_mask], data['order'].y.to(device)[train_mask], weight=gnn_weights)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        # Correção do erro: conversão direta para lista Python via .tolist()
        preds_gnn = model(x_dict_gpu, edge_index_dict_gpu)['order'][val_mask].argmax(dim=-1).cpu().tolist()
        
    # --- Treinamento XGBoost Baseline ---
    xgb_model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        objective='binary:logistic', eval_metric='logloss',
        scale_pos_weight=weight_pos, enable_categorical=True, random_state=42
    )
    xgb_model.fit(X_xgb.iloc[train_idx], y_xgb.iloc[train_idx])
    preds_xgb = xgb_model.predict(X_xgb.iloc[val_idx])
    
    # Extração de Métricas Individuais (Foco na Classe 1: Atrasado)
    f1_gnn = f1_score(y_val_fold, preds_gnn, zero_division=0)
    f1_xgb = f1_score(y_val_fold, preds_xgb, zero_division=0)
    
    folds_gnn_f1_atraso.append(f1_gnn)
    folds_xgb_f1_atraso.append(f1_xgb)
    
    folds_metrics.append({
        'acc_xgb': accuracy_score(y_val_fold, preds_xgb),
        'prc_xgb': precision_score(y_val_fold, preds_xgb, zero_division=0),
        'rec_xgb': recall_score(y_val_fold, preds_xgb, zero_division=0),
        'f1_xgb':  f1_xgb,
        'acc_gnn': accuracy_score(y_val_fold, preds_gnn),
        'prc_gnn': precision_score(y_val_fold, preds_gnn, zero_division=0),
        'rec_gnn': recall_score(y_val_fold, preds_gnn, zero_division=0),
        'f1_gnn':  f1_gnn,
        'prevalence': np.mean(y_val_fold),
        'instances': len(y_val_fold)
    })

df_res = pd.DataFrame(folds_metrics)

# %% 5. Exibição Formal de Resultados (Formato Sumário Estatístico na Tela)

print("\n" + "="*80)
print("MUTUAL COMPARATIVE SUMMARY: BASELINE vs HETEROGENEOUS GNN (TIME SERIES CV)")
print("="*80)
print(f"{'METRIC (CLASS: DELAYED)':<30}  | {'XGBOOST BASELINE':<22} | {'GRAPH-SAGE GNN':<22}")
print("-"*80)
print(f"{'Global Accuracy (Acurácia)':<30}  | {df_res['acc_xgb'].mean():.2%} ± {df_res['acc_xgb'].std():.2%}         |  {df_res['acc_gnn'].mean():.2%} ± {df_res['acc_gnn'].std():.2%}")
print(f"{'Class Precision (Precisão)':<30}  | {df_res['prc_xgb'].mean():.4f} ± {df_res['prc_xgb'].std():.4f}        |  {df_res['prc_gnn'].mean():.4f} ± {df_res['prc_gnn'].std():.4f}")
print(f"{'Class Recall (Revocação)':<30}  | {df_res['rec_xgb'].mean():.4f} ± {df_res['rec_xgb'].std():.4f}        |  {df_res['rec_gnn'].mean():.4f} ± {df_res['rec_gnn'].std():.4f}")
print(f"{'F1-Score (Medida F)':<30}  | {df_res['f1_xgb'].mean():.4f} ± {df_res['f1_xgb'].std():.4f}        |  {df_res['f1_gnn'].mean():.4f} ± {df_res['f1_gnn'].std():.4f}")
print("-"*80)
print(f"{'Target Base Prevalence (Média)':<30}  | {df_res['prevalence'].mean():<22.2%} |  {df_res['prevalence'].mean():<22.2%}")
print(f"{'Total Test Instances (Por Fold)':<30} | {int(df_res['instances'].mean()):<22d} |  {int(df_res['instances'].mean()):<22d}")
print("="*80)

# %% 6. Teste Estatístico Formal das Hipóteses H0/H1

t_stat, p_value = ttest_rel(folds_gnn_f1_atraso, folds_xgb_f1_atraso)
print("\n>>> TESTE DE HIPÓTESE FORMAL (Paired t-test nos Folds Temporais):")
print(f"Estatística t: {t_stat:.4f} | p-value: {p_value:.5f}")
if p_value < 0.05:
    print("Resultado: Rejeita-se H0! A superioridade do modelo de Grafos Heterogêneos é ESTATISTICAMENTE SIGNIFICATIVA (p < 0.05).")
else:
    print("Resultado: Não se rejeita H0. A diferença observada não possui significância estatística formal no número de folds avaliado.")
print("="*80)

# %% 7. Modulos de Explicabilidade (SHAP e GNNExplainer) 
# Código para extração de valores SHAP (XGBoost)
# Código para execução do GNNExplainer (HeteroGNN)

print("\n" + "="*80)
print("INICIANDO ETAPA DE EXPLICABILIDADE DOS MODELOS")
print("="*80)


# 7.1. SHAP (SHapley Additive exPlanations) para XGBoost
print("\n>>> [1/2] Gerando explicações SHAP para o modelo XGBoost...")

explainer_xgb = shap.TreeExplainer(xgb_model)
X_val_sample = X_xgb.iloc[val_idx]
shap_values_xgb = explainer_xgb(X_val_sample)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_xgb, X_val_sample, show=False)
plt.title("XGBoost - Importância das Features via SHAP Values", fontsize=12)
plt.tight_layout()
plt.show()

# 7.2. GNNExplainer para GraphSAGE (PyTorch Geometric)
print("\n>>> [2/2] Gerando explicações via GNNExplainer para a GraphSAGE...")

# Wrapper para extrair apenas as previsões do nó 'order' (para grafos heterogêneos)
class HeteroOrderWrapper(torch.nn.Module):
    def __init__(self, gnn_model):
        super().__init__()
        self.gnn_model = gnn_model
        
    def forward(self, x_dict, edge_index_dict, *args, **kwargs):
        out_dict = self.gnn_model(x_dict, edge_index_dict)
        return out_dict['order']

wrapped_model = HeteroOrderWrapper(model)

# Configuração focada em explicabilidade dos atributos dos nós (edge_mask_type=None)
explainer_gnn = Explainer(
    model=wrapped_model,
    algorithm=GNNExplainer(epochs=100),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type=None,  # Desativa máscara de arestas não alcançadas no subgrafo
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='probs'
    )
)

# Seleciona um pedido específico da base de validação para explicar (ex: o 1º do fold)
target_order_idx = int(val_idx[0])

# Executa a busca pelos atributos de nós mais influentes na previsão
explanation = explainer_gnn(
    x=x_dict_gpu,
    edge_index=edge_index_dict_gpu,
    index=target_order_idx
)

print(f"GNNExplainer concluído com sucesso para o Pedido Índice: {target_order_idx}")
if hasattr(explanation, 'node_mask_dict'):
    print(f"- Máscaras de importância de atributos geradas para as entidades:")
    for node_type, mask in explanation.node_mask_dict.items():
        print(f"  * Entidade '{node_type}': {mask.shape}")
print("="*80)
