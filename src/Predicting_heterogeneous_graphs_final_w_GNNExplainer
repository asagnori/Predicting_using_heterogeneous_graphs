#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 13:55:19 2026

@author: angelosagnori
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

import shap
import torch
import torch.nn.functional as F

import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.explain import Explainer, GNNExplainer

import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    average_precision_score, confusion_matrix, precision_recall_curve)

from scipy.stats import ttest_rel
from sqlalchemy import create_engine

# -----------------------------------------------------------------------------
# 0. Configurações de ambiente e Sementes
# -----------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

torch.manual_seed(42)
np.random.seed(42)

# Conexão com o Banco de Dados
config = {
    'user': '',
    'password': '',
    'host': 'localhost:3306',
    'database': 'bike_store'
}

engine = create_engine(
    f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}/{config['database']}"
)

# -----------------------------------------------------------------------------
# 1. Obter, Carregar e Ordenar os dados cronologicamente
# -----------------------------------------------------------------------------
def load_and_sort_data(engine):
    tables = ['customers', 'orders', 'order_items', 'products', 'staffs', 'stores', 'brands', 'categories']
    df_dict = {t: pd.read_sql(f"SELECT * FROM {t}", engine) for t in tables}
    
    df_dict['orders']['y'] = (df_dict['orders']['shipped_date'] > df_dict['orders']['required_date']).astype(int)
    df_dict['orders']['order_date'] = pd.to_datetime(df_dict['orders']['order_date'])
    df_dict['orders'] = df_dict['orders'].sort_values('order_date').reset_index(drop=True)
    return df_dict

df_dict = load_and_sort_data(engine)

# Mapeamentos de IDs para a GNN (Alinhado com a EDA teórica)
def get_mapping(df, id_col):
    return {old_id: new_id for new_id, old_id in enumerate(df[id_col].unique())}

maps = {
    'customer': get_mapping(df_dict['customers'], 'customer_id'),
    'order': {old_id: new_id for new_id, old_id in enumerate(df_dict['orders']['order_id'])},
    'product': get_mapping(df_dict['products'], 'product_id'),
    'brand': get_mapping(df_dict['brands'], 'brand_id'),
    'category': get_mapping(df_dict['categories'], 'category_id'),
    'store': get_mapping(df_dict['stores'], 'store_id'),
    'staff': get_mapping(df_dict['staffs'], 'staff_id')
}

# -----------------------------------------------------------------------------
# 2. Construção do Grafo Heterogêneo (GNN) e Feature de Nós (PyTorch Geometric)
# -----------------------------------------------------------------------------
data = HeteroData()

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

# Self-loops em todas as entidades
for node_type in data.node_types:
    num_nodes = data[node_type].x.shape[0]
    indices = torch.arange(num_nodes, dtype=torch.long)
    data[node_type, 'to', node_type].edge_index = torch.stack([indices, indices], dim=0)

print("--- Grafo Heterogêneo Consolidado ---")
print(data)

# Exibição do Schema do Grafo Heterogêneo
G_schema = nx.DiGraph()
G_schema.add_nodes_from(['customer', 'store', 'staff', 'product', 'brand', 'category', 'order'])
G_schema.add_edges_from([
    ('customer', 'order'), ('store', 'order'), ('staff', 'order'),
    ('product', 'order'), ('brand', 'product'), ('category', 'product')
])
pos_schema = {
    'brand': (0.0, 1.0), 'category': (0.0, -0.2), 'product': (1.0, 0.4),
    'store': (1.0, 1.8), 'customer': (1.7, 2.5), 'staff': (2.2, 0.4), 'order': (2.4, 1.5)
}
fig, ax = plt.subplots(figsize=(10, 7))
nx.draw_networkx_nodes(G_schema, pos_schema, node_size=3200, node_color='#1f77b4', ax=ax)
nx.draw_networkx_labels(G_schema, pos_schema, font_size=11, font_color='white', font_weight='bold', ax=ax)
nx.draw_networkx_edges(G_schema, pos_schema, arrowstyle='->', arrowsize=22, edge_color='#333333', width=1.5, node_size=3200, ax=ax)
for node, (x_p, y_p) in pos_schema.items():
    ax.annotate(
        'to', xy=(x_p, y_p + 0.18), xytext=(x_p, y_p + 0.38),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-1.3", lw=1.2, color='#222222'),
        ha='center', va='center', fontsize=10, fontweight='bold'
    )
ax.set_title("Schema do Grafo Heterogêneo", fontsize=14, pad=20, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 3. Definição da arquitetura GRAPHSAGE - Modelo Dinâmico (Corrigido para Nós Fonte/Destino)
# -----------------------------------------------------------------------------
class GNNModel(torch.nn.Module):
    def __init__(self, edge_types, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = HeteroConv({et: SAGEConv((-1, -1), hidden_channels) for et in edge_types}, aggr='sum')
        self.conv2 = HeteroConv({et: SAGEConv((-1, -1), out_channels) for et in edge_types}, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        out1 = self.conv1(x_dict, edge_index_dict)
        x_dict_l1 = {k: F.relu(out1[k]) if k in out1 else x_dict[k] for k in x_dict.keys()}
        out2 = self.conv2(x_dict_l1, edge_index_dict)
        return out2

# -----------------------------------------------------------------------------
# 4. Validação cruzada temporal (Time Series Split - K= 4)
# -----------------------------------------------------------------------------
tscv = TimeSeriesSplit(n_splits=4)
df_orders = df_dict['orders'].copy()
df_orders['order_month'] = pd.to_datetime(df_orders['order_date']).dt.month.astype('category')
df_orders['required_month'] = pd.to_datetime(df_orders['required_date']).dt.month.astype('category')

# Base Tabular sem customer_id bruto (substituído por store_id, staff_id e datas)
X_xgb_base = pd.DataFrame({
    'store_id': df_orders['store_id'].astype('category'),
    'staff_id': df_orders['staff_id'].astype('category'),
    'order_month': df_orders['order_month'],
    'required_month': df_orders['required_month']
})
y_xgb = df_orders['y']

x_dict_gpu = {k: v.to(device) for k, v in data.x_dict.items()}
edge_index_dict_gpu = {k: v.to(device) for k, v in data.edge_index_dict.items()}

folds_metrics = []
folds_gnn_f1 = []
folds_xgb_f1 = []
folds_dummy_f1 = []

# Acumuladores de probabilidades para a Curva PR-AUC
all_y_val = []
all_probs_dummy = []
all_probs_xgb = []
all_probs_gnn = []

# Acumuladores das classes previstas para as matrizes de confusão
all_preds_xgb = []
all_preds_gnn = []

# Modelo e dados do último fold para explicabilidade
gnn_model_explain = None
gnn_eval_edges_explain = None
gnn_x_dict_explain = None
gnn_val_idx_explain = None

for fold, (train_idx, val_idx) in enumerate(tscv.split(df_orders)):
    print(f"\n>>> Processando Fold Temporal {fold + 1}/4...")
    
    y_train_fold = y_xgb.iloc[train_idx].values
    y_val_fold = y_xgb.iloc[val_idx].values
    weight_pos = np.sum(y_train_fold == 0) / np.sum(y_train_fold == 1)
    
    # A. Classificar Dummy (BASELINE CEGO)
    dummy_model = DummyClassifier(strategy="stratified", random_state=42)
    dummy_model.fit(X_xgb_base.iloc[train_idx], y_train_fold)
    preds_dummy = dummy_model.predict(X_xgb_base.iloc[val_idx])
    probs_dummy = dummy_model.predict_proba(X_xgb_base.iloc[val_idx])[:, 1]
    
    # B. Modelo XGBOOST Baseline (Sem IDs brutos)
    xgb_model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        objective='binary:logistic', eval_metric='logloss',
        scale_pos_weight=weight_pos, enable_categorical=True, random_state=42
    )
    xgb_model.fit(X_xgb_base.iloc[train_idx], y_train_fold)
    preds_xgb = xgb_model.predict(X_xgb_base.iloc[val_idx])
    probs_xgb = xgb_model.predict_proba(X_xgb_base.iloc[val_idx])[:, 1]
    
    # C. GRAPHSAGE GNN (com isolamento de arestas posteriores)
    train_order_set = set(train_idx)
    val_order_set = set(val_idx)
    eval_order_set = train_order_set.union(val_order_set)
    
    # Filtragem estrita das arestas de treino (Elimina posterior edges)
    train_edge_index = {}
    eval_edge_index = {}
    
    for etype, eindex in edge_index_dict_gpu.items():
        src_type, rel, dst_type = etype
        
        # Filtro de treino: conexões apenas com pedidos do conjunto de treino
        if dst_type == 'order':
            mask_tr = torch.tensor([idx.item() in train_order_set for idx in eindex[1]], device=device)
            mask_ev = torch.tensor([idx.item() in eval_order_set for idx in eindex[1]], device=device)
        elif src_type == 'order':
            mask_tr = torch.tensor([idx.item() in train_order_set for idx in eindex[0]], device=device)
            mask_ev = torch.tensor([idx.item() in eval_order_set for idx in eindex[0]], device=device)
        else:
            mask_tr = torch.ones(eindex.shape[1], dtype=torch.bool, device=device)
            mask_ev = torch.ones(eindex.shape[1], dtype=torch.bool, device=device)
            
        train_edge_index[etype] = eindex[:, mask_tr]
        eval_edge_index[etype] = eindex[:, mask_ev]

    # Conversão explícita para evitar erro 'numpy.int64'
    train_idx_tensor = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx_tensor = torch.tensor(val_idx, dtype=torch.long, device=device)

    train_mask = torch.zeros(len(df_orders), dtype=torch.bool, device=device)
    train_mask[train_idx_tensor] = True
    
    val_mask = torch.zeros(len(df_orders), dtype=torch.bool, device=device)
    val_mask[val_idx_tensor] = True   
    
    gnn_weights = torch.tensor([1.0, weight_pos], dtype=torch.float).to(device)
    model = GNNModel(data.edge_types, hidden_channels=64, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Treinamento usando apenas as arestas de treino
    for epoch in range(1, 51):
        model.train()
        optimizer.zero_grad()
        out = model(x_dict_gpu, train_edge_index)
        loss = F.cross_entropy(out['order'][train_mask], data['order'].y.to(device)[train_mask], weight=gnn_weights)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        out_eval = model(x_dict_gpu, eval_edge_index)['order'][val_mask]
        
    # Guardar o modelo treinado e o contexto do fold
    gnn_model_explain = model
    gnn_eval_edges_explain = eval_edge_index
    gnn_x_dict_explain = x_dict_gpu
    gnn_val_idx_explain = val_idx.copy()
        
    # Uso de .detach().cpu().tolist() para evitar incompatibilidade do .numpy()
    probs_gnn = F.softmax(out_eval, dim=-1)[:, 1].detach().cpu().tolist()
    preds_gnn = out_eval.argmax(dim=-1).detach().cpu().tolist()
    
    # Acumular previsões de cada fold
    all_preds_xgb.extend(np.asarray(preds_xgb).astype(int).tolist())
    all_preds_gnn.extend(preds_gnn)

    # Métricas
    f1_gnn = f1_score(y_val_fold, preds_gnn, zero_division=0)
    f1_xgb = f1_score(y_val_fold, preds_xgb, zero_division=0)
    f1_dummy = f1_score(y_val_fold, preds_dummy, zero_division=0)
    
    folds_gnn_f1.append(f1_gnn)
    folds_xgb_f1.append(f1_xgb)
    folds_dummy_f1.append(f1_dummy)
    
    all_y_val.extend(y_val_fold)
    all_probs_dummy.extend(probs_dummy)
    all_probs_xgb.extend(probs_xgb)
    all_probs_gnn.extend(probs_gnn)
    
    # ------------------------------------------------------------------------- 
    folds_metrics.append({
        'acc_dummy': accuracy_score(y_val_fold, preds_dummy),
        'f1_dummy':  f1_dummy,
        'prauc_dummy': average_precision_score(y_val_fold, probs_dummy),
        
        'acc_xgb': accuracy_score(y_val_fold, preds_xgb),
        'prc_xgb': precision_score(y_val_fold, preds_xgb, zero_division=0),
        'rec_xgb': recall_score(y_val_fold, preds_xgb, zero_division=0),
        'f1_xgb':  f1_xgb,
        'prauc_xgb': average_precision_score(y_val_fold, probs_xgb),
        
        'acc_gnn': accuracy_score(y_val_fold, preds_gnn),
        'prc_gnn': precision_score(y_val_fold, preds_gnn, zero_division=0),
        'rec_gnn': recall_score(y_val_fold, preds_gnn, zero_division=0),
        'f1_gnn':  f1_gnn,
        'prauc_gnn': average_precision_score(y_val_fold, probs_gnn),
    })

df_res = pd.DataFrame(folds_metrics)

# ----------------------------------------------------------------------
# Matriz de Confusão com Previsões Agregadas dos 4 Folds
# ----------------------------------------------------------------------
y_true = np.asarray(all_y_val, dtype=int)
y_pred_xgb = np.asarray(all_preds_xgb, dtype=int)
y_pred_gnn = np.asarray(all_preds_gnn, dtype=int)

# Matrizes de confusão: linhas = classe real; colunas = classe prevista
cm_xgb = confusion_matrix(y_true, y_pred_xgb, labels=[0, 1])
cm_gnn = confusion_matrix(y_true, y_pred_gnn, labels=[0, 1])

# Normalização por linha para mostrar também os percentuais
cm_xgb_norm = confusion_matrix(
    y_true, y_pred_xgb, labels=[0, 1], normalize='true'
)
cm_gnn_norm = confusion_matrix(
    y_true, y_pred_gnn, labels=[0, 1], normalize='true'
)

# Anotações: quantidade absoluta + percentual por classe real
def make_annotations(cm, cm_norm):
    return np.array([
        [f"{cm[i, j]}\n({cm_norm[i, j]:.1%})" for j in range(2)]
        for i in range(2)
    ])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

labels = ['No prazo (0)', 'Atrasado (1)']

for ax, cm_norm, annotations, title in [
    (axes[0], cm_xgb_norm, make_annotations(cm_xgb, cm_xgb_norm),
     'XGBoost'),
    (axes[1], cm_gnn_norm, make_annotations(cm_gnn, cm_gnn_norm),
     'GraphSAGE')
]:
    sns.heatmap(
        cm_norm,
        annot=annotations,
        fmt='',
        cmap='Blues',
        vmin=0,
        vmax=1,
        cbar=False,
        linewidths=1,
        linecolor='white',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={'size': 12}
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Classe prevista')
    ax.set_ylabel('Classe real')

plt.suptitle(
    'Matrizes de Confusão — Validação Temporal (K=4)',
    fontsize=15,
    fontweight='bold'
)

plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png',
            dpi=300, bbox_inches='tight')
plt.show()

print("Matriz XGBoost:\n", cm_xgb)
print("\nMatriz GraphSAGE:\n", cm_gnn)


# -----------------------------------------------------------------------------
# 5. Exibição formal dos resultados (Tabela completa com DUMMY e PR-AUC)
# -----------------------------------------------------------------------------

print("\n" + "="*95)
print("Mutual Comparative Summary: Dummy vs Baseline XGBOOST vs Heterogeneous GNN (Time Series CV)")
print("="*95)
print(f"{'Metric (CLASS: Delayed)':<25} | {'DUMMY Classifier':<20} | {'XGBOOST Baseline':<20}  | {'GRAPHSAGE GNN':<20}")
print("-"*95)
print(f"{'Global Accuracy':<25} | {df_res['acc_dummy'].mean():.2%} ± {df_res['acc_dummy'].std():.2%}       | {df_res['acc_xgb'].mean():.2%} ± {df_res['acc_xgb'].std():.2%}        | {df_res['acc_gnn'].mean():.2%} ± {df_res['acc_gnn'].std():.2%}")
print(f"{'Class Precision':<25} | -                    | {df_res['prc_xgb'].mean():.4f} ± {df_res['prc_xgb'].std():.4f}       | {df_res['prc_gnn'].mean():.4f} ± {df_res['prc_gnn'].std():.4f}")
print(f"{'Class Recall':<25} | -                    | {df_res['rec_xgb'].mean():.4f} ± {df_res['rec_xgb'].std():.4f}       | {df_res['rec_gnn'].mean():.4f} ± {df_res['rec_gnn'].std():.4f}")
print(f"{'F1-Score':<25} | {df_res['f1_dummy'].mean():.4f} ± {df_res['f1_dummy'].std():.4f}      | {df_res['f1_xgb'].mean():.4f} ± {df_res['f1_xgb'].std():.4f}       | {df_res['f1_gnn'].mean():.4f} ± {df_res['f1_gnn'].std():.4f}")
print(f"{'PR-AUC (Precision-Recall)':<25} | {df_res['prauc_dummy'].mean():.4f} ± {df_res['prauc_dummy'].std():.4f}      | {df_res['prauc_xgb'].mean():.4f} ± {df_res['prauc_xgb'].std():.4f}       | {df_res['prauc_gnn'].mean():.4f} ± {df_res['prauc_gnn'].std():.4f}")
print("="*95)

# -----------------------------------------------------------------------------
# 6. Testes Estatísticos formal das hipóteses H0/H1 Pareados (Paired T-Test)
# -----------------------------------------------------------------------------

t_stat_gnn_xgb, p_val_gnn_xgb = ttest_rel(folds_gnn_f1, folds_xgb_f1)
t_stat_xgb_dum, p_val_xgb_dum = ttest_rel(folds_xgb_f1, folds_dummy_f1)

print("\n>>> Testes de hipóteses pareados (Paired t-test nos Folds Temporais):")
print(f"1. XGBoost vs Dummy Classifier : t = {t_stat_xgb_dum:.4f} | p-value = {p_val_xgb_dum:.5f}")
print(f"2. GraphSAGE vs XGBoost Baseline: t = {t_stat_gnn_xgb:.4f} | p-value = {p_val_gnn_xgb:.5f}")
print("="*95)

if p_val_gnn_xgb < 0.05:
    print("Resultado: Rejeita-se H0! A superioridade do modelo de Grafos Heterogêneos é ESTATISTICAMENTE SIGNIFICATIVA (p < 0.05).")
else:
    print("Resultado: Não se rejeita H0. A diferença observada não possui significância estatística formal no número de folds avaliado.")
print("="*95)

# -----------------------------------------------------------------------------
# 7. Explicabilidade SHAP Global e Local (XGBOOST sem ruído de Id's))
# -----------------------------------------------------------------------------

# Exibe Modelo treinado com validação temporal (último fold)
print("\n>>> Gerando explicações SHAP para o modelo XGBoost...")

# 1. Dados de validação do último fold
X_val_sample = X_xgb_base.iloc[val_idx].copy()

# 2. Explainer para modelo baseado em árvores
explainer_xgb = shap.TreeExplainer(xgb_model)

# 3. Calcular SHAP Values
shap_values_xgb = explainer_xgb(X_val_sample)


# Gráfico 1 - BEESWARM: Importância e Direção do Impacto

plt.figure(figsize=(10, 6))

shap.plots.beeswarm(
    shap_values_xgb,
    max_display=15,
    show=False
)

plt.title(
    "XGBoost - Importância das Features Reais via SHAP Values",
    fontsize=12
)

plt.tight_layout()
plt.show()


# Gráfico 2 - BAR: Importância Global

plt.figure(figsize=(10, 6))

shap.plots.bar(
    shap_values_xgb,
    max_display=15,
    show=False
)

plt.title(
    "XGBoost - Importância Global das Features (SHAP)",
    fontsize=12
)

plt.tight_layout()
plt.show()


# Gráfico 3 - WATERFALL: Explicação Individual

# Amostra: pedido 1473, mesmo pedido analisado pelo GNNExplainer
pedido_id = 1473

# Localizar o pedido dentro do conjunto de validação
orders_val = df_dict['orders'].iloc[val_idx].reset_index(drop=True)

pedido_positions = np.where(
    orders_val['order_id'].values == pedido_id
)[0]

if len(pedido_positions) > 0:

    pos = pedido_positions[0]

    plt.figure(figsize=(10, 6))

    shap.plots.waterfall(
        shap_values_xgb[pos],
        max_display=15,
        show=False
    )

    plt.title(
        f"XGBoost - Explicação SHAP do Pedido {pedido_id}",
        fontsize=12
    )

    plt.tight_layout()
    plt.show()

else:
    print(
        f"Pedido {pedido_id} não encontrado "
        "na validação do último fold."
    )


print("\n>>> Análise SHAP concluída.")


# -----------------------------------------------------------------------------
# 8. Representação Gráfica da PR-AUC e Comparativo de Métricas
# -----------------------------------------------------------------------------
y_val_arr = np.asarray(all_y_val).ravel()
probs_xgb_arr = np.asarray(all_probs_xgb).ravel()
probs_gnn_arr = np.asarray(all_probs_gnn).ravel()

if len(y_val_arr) > 0:
    p_xgb, r_xgb, _ = precision_recall_curve(y_val_arr, probs_xgb_arr)
    p_gnn, r_gnn, _ = precision_recall_curve(y_val_arr, probs_gnn_arr)

    pos_ratio = np.mean(y_val_arr)

    plt.figure(figsize=(8, 5))
    plt.plot([0, 1], [pos_ratio, pos_ratio], '--', color='#7f7f7f',
             label=f'Dummy (PR-AUC = {df_res["prauc_dummy"].mean():.4f})')
    plt.plot(r_xgb, p_xgb, color='#d62728', lw=2,
             label=f'XGBoost (PR-AUC = {df_res["prauc_xgb"].mean():.4f})')
    plt.plot(r_gnn, p_gnn, color='#2ca02c', lw=2,
             label=f'GraphSAGE (PR-AUC = {df_res["prauc_gnn"].mean():.4f})')

    plt.title('Curvas Precision-Recall — Validação Temporal')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()

else:
    print("Curva PR não gerada: previsões dos folds não foram acumuladas.")
    
# -----------------------------------------------------------------------------
# 9. Explicabilidade com GNN Explainer
# -----------------------------------------------------------------------------

# GNN Explainer Agregado
# Para explicar todos os pedidos da validação, use:
# N_ORDERS = None

N_ORDERS = 30      # Número de pedidos a explicar
RANDOM_STATE = 42
TOP_N = 2           # Número de relações consideradas no Top N


# Validar Objetos Necessários
required_objects = [
    'explainer',
    'gnn_x_dict_explain',
    'edge_index_explain',
    'gnn_val_idx_explain',
    'maps',
    'orders'
]

# 1. Recuperar Modelo e Grafo do Último Fold 
if gnn_model_explain is None:
    raise ValueError("Modelo GraphSAGE não encontrado.")

if gnn_eval_edges_explain is None:
    raise ValueError("Arestas de avaliação não encontradas.")

if gnn_x_dict_explain is None or gnn_val_idx_explain is None:
    raise ValueError("Dados de explicabilidade não encontrados.")

gnn_model_explain.eval()

# Alias para manter a nomenclatura usada no código do explainer
edge_index_explain = gnn_eval_edges_explain


# 2. Definir as relações de cada camada

# Primeira camada: propaga informações para os diferentes
# tipos de nós do grafo.
layer1_relations = [
    etype for etype in edge_index_explain.keys()
    if etype[2] in ['customer', 'product', 'store', 'staff', 'order']
]

# Segunda camada: propaga informações para os pedidos.
layer2_relations = [
    etype for etype in edge_index_explain.keys()
    if etype[2] == 'order'
]


# 3. Wrapper para explicar a previsão do no order

class OrderExplainerModel(torch.nn.Module):

    def __init__(self, trained_model):
        super().__init__()
        self.model = trained_model

    def forward(self, x_dict, edge_index_dict):

        edge_layer1 = {
            etype: edge_index
            for etype, edge_index in edge_index_dict.items()
            if etype in layer1_relations
        }

        out1 = self.model.conv1(x_dict, edge_layer1)

        x_dict_l1 = {
            k: F.relu(out1[k]) if k in out1 else x_dict[k]
            for k in x_dict.keys()
        }

        edge_layer2 = {
            etype: edge_index
            for etype, edge_index in edge_index_dict.items()
            if etype in layer2_relations
        }

        out2 = self.model.conv2(x_dict_l1, edge_layer2)

        return out2['order']


model_explain = OrderExplainerModel(
    gnn_model_explain
).to(device)

model_explain.eval()


# 4. Configurar o GNNExplainer

explainer = Explainer(
    model=model_explain,
    algorithm=GNNExplainer(
        epochs=200,
        lr=0.01
    ),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='raw'
    )
)

print("GNNExplainer configurado com sucesso.")
print("Modelo: GraphSAGE do último fold temporal.")
print("Pedidos disponíveis:", len(gnn_val_idx_explain))

# GNNExplainer Global -  GraphSage Heterogeneo

# 1. Validar variáveis
required_vars = [
    'gnn_model_explain',
    'gnn_eval_edges_explain',
    'gnn_x_dict_explain',
    'gnn_val_idx_explain',
    'maps',
    'df_dict',
    'device'
]

missing = [v for v in required_vars if v not in globals()]

if missing:
    raise NameError(f"Variáveis não encontradas: {missing}")

if gnn_model_explain is None:
    raise ValueError("Modelo do último fold não encontrado.")


# 2. Reconstruir Rótulos 
orders_df = df_dict['orders'].copy()

orders_df['required_date'] = pd.to_datetime(
    orders_df['required_date']
)

orders_df['shipped_date'] = pd.to_datetime(
    orders_df['shipped_date']
)

orders_df['actual_class'] = (
    orders_df['shipped_date'] > orders_df['required_date']
).astype(int)

orders_labels = orders_df.set_index('order_id')['actual_class']

y_np = np.empty(len(maps['order']), dtype=int)

for order_id, node_idx in maps['order'].items():
    y_np[node_idx] = int(orders_labels.loc[order_id])

 
# 3. Selecionar Pedidos da Validação
val_indices = np.asarray(gnn_val_idx_explain)

val_class_0 = val_indices[y_np[val_indices] == 0]
val_class_1 = val_indices[y_np[val_indices] == 1]

N_PER_CLASS = 25
rng = np.random.default_rng(42)

sample_0 = rng.choice(
    val_class_0,
    size=min(N_PER_CLASS, len(val_class_0)),
    replace=False
)

sample_1 = rng.choice(
    val_class_1,
    size=min(N_PER_CLASS, len(val_class_1)),
    replace=False
)

orders_to_explain = np.concatenate([sample_0, sample_1])
rng.shuffle(orders_to_explain)

print(f"Pedidos selecionados: {len(orders_to_explain)}")
print(f"Não atrasados: {len(sample_0)}")
print(f"Atrasados: {len(sample_1)}")

 
# 4. Filtrar Relações Relevantes para a Predição de Ordens
edge_index_full = gnn_eval_edges_explain

# Primeira camada:
# obter embeddings de customer, product, store, staff e order
layer1_relations = [
    et for et in edge_index_full.keys()
    if et[2] in [
        'customer',
        'product',
        'store',
        'staff',
        'order'
    ]
]

# Segunda camada: somente relações que chegam ao nó order
layer2_relations = [
    et for et in edge_index_full.keys()
    if et[2] == 'order'
]

# União das relações utilizadas pelo caminho até order
relevant_relations = list(dict.fromkeys(
    layer1_relations + layer2_relations
))

edge_index_explain = {
    et: edge_index_full[et]
    for et in relevant_relations
}

print("\nRelações utilizadas pelo explainer:")
for et in edge_index_explain:
    print(et)

# 5. Wrapper para duas camadas heterogêneas
class OrderExplainerModel(torch.nn.Module):

    def __init__(self, trained_model):
        super().__init__()
        self.model = trained_model

    def forward(self, x_dict, edge_index_dict):

        # Primeira camada
        edges_l1 = {
            et: edge_index_dict[et]
            for et in layer1_relations
            if et in edge_index_dict
        }

        out1 = self.model.conv1(x_dict, edges_l1)

        x_dict_l1 = {
            k: F.relu(out1[k]) if k in out1 else x_dict[k]
            for k in x_dict.keys()
        }

        # Segunda camada
        edges_l2 = {
            et: edge_index_dict[et]
            for et in layer2_relations
            if et in edge_index_dict
        }

        out2 = self.model.conv2(x_dict_l1, edges_l2)

        return out2['order']

model_explain = OrderExplainerModel(
    gnn_model_explain
).to(device)

model_explain.eval()

# 6. Configurar o GNNExplainer
explainer = Explainer(
    model=model_explain,
    algorithm=GNNExplainer(
        epochs=200,
        lr=0.01
    ),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='raw'
    )
)

# 7. Identificar relações diretas e indiretas
product_order_rel = next(
    et for et in edge_index_explain
    if et[0] == 'product' and et[2] == 'order'
)

product_order_edges = edge_index_explain[product_order_rel]

direct_relations = [
    et for et in edge_index_explain
    if et[2] == 'order' and et[0] != 'order'
]

indirect_relations = [
    et for et in edge_index_explain
    if et[2] == 'product'
    and et[0] in ['brand', 'category']
]

# 8. Executar explicações locais
records = []

for i, order_idx in enumerate(orders_to_explain):

    explanation = explainer(
        gnn_x_dict_explain,
        edge_index_explain,
        index=int(order_idx)
    )

    edge_masks = explanation.edge_mask_dict
    actual_class = int(y_np[order_idx])

    # Produtos associados ao pedido
    product_mask = (
        product_order_edges[1] == int(order_idx)
    )

    product_indices = product_order_edges[
        0, product_mask
    ].unique()

    # Relações diretas
    for et in direct_relations:

        edge_index = edge_index_explain[et]
        mask = edge_masks.get(et)

        if mask is None:
            continue

        selected = edge_index[1] == int(order_idx)

        if selected.any():

            values = np.asarray(mask[selected].detach().cpu().tolist())

            records.append({
                'order_idx': int(order_idx),
                'actual_class': actual_class,
                'relation': f'{et[0]} -> {et[2]}',
                'relation_type': 'Direct',
                'mean_importance': float(values.mean()),
                'max_importance': float(values.max()),
                'n_edges': len(values)
            })

    # Relações indiretas
    for et in indirect_relations:

        edge_index = edge_index_explain[et]
        mask = edge_masks.get(et)

        if mask is None or len(product_indices) == 0:
            continue

        # Executar a comparação no CPU para evitar incompatibilidade com MPS
        selected = torch.isin(
            edge_index[1].cpu(),
            product_indices.cpu()
        ).to(edge_index.device)

        if selected.any():

            values = np.asarray(mask[selected].detach().cpu().tolist())

            records.append({
                'order_idx': int(order_idx),
                'actual_class': actual_class,
                'relation': f'{et[0]} -> {et[2]}',
                'relation_type': 'Indirect',
                'mean_importance': float(values.mean()),
                'max_importance': float(values.max()),
                'n_edges': len(values)
            })

    if (i + 1) % 5 == 0:
        print(f"Processados: {i + 1}/{len(orders_to_explain)}")

# 9. Consolidar explicações
df_global = pd.DataFrame(records)

if df_global.empty:
    raise ValueError("Nenhuma máscara foi coletada.")

df_global_summary = (
    df_global
    .groupby(['relation', 'relation_type'])
    .agg(
        mean_importance=('mean_importance', 'mean'),
        median_importance=('mean_importance', 'median'),
        std_importance=('mean_importance', 'std'),
        mean_max_importance=('max_importance', 'mean'),
        orders_explained=('order_idx', 'nunique')
    )
    .reset_index()
    .sort_values('mean_importance', ascending=False)
)

print("\nIMPORTÂNCIA GLOBAL")
print(df_global_summary.to_string(index=False))

# 10. Gráfico Global
plot_data = df_global_summary.sort_values(
    'mean_importance',
    ascending=True
)

plt.figure(figsize=(10, 6))
plt.barh(
    plot_data['relation'],
    plot_data['mean_importance']
)

plt.xlabel('Importância média da máscara')
plt.ylabel('Relação do grafo')
plt.title('GNNExplainer - Importância Global das Relações')
plt.tight_layout()
plt.show()

 
# 11. Comparação por Classe
df_class_summary = (
    df_global
    .groupby(['actual_class', 'relation'])
    .agg(
        mean_importance=('mean_importance', 'mean'),
        orders_explained=('order_idx', 'nunique')
    )
    .reset_index()
)

df_class_pivot = df_class_summary.pivot(
    index='relation',
    columns='actual_class',
    values='mean_importance'
).fillna(0)

df_class_pivot.columns = [
    'Não atrasado' if c == 0 else 'Atrasado'
    for c in df_class_pivot.columns
]

print("\nIMPORTÂNCIA POR CLASSE")
print(df_class_pivot.to_string())

df_class_pivot.plot(
    kind='barh',
    figsize=(10, 6)
)

plt.xlabel('Importância média da máscara')
plt.ylabel('Relação do grafo')
plt.title('GNNExplainer - Comparação por Classe Real')
plt.tight_layout()
plt.show()

 
# 12. Exportar Resultados
df_global.to_csv(
    'gnnexplainer_global_detalhado.csv',
    index=False
)

df_global_summary.to_csv(
    'gnnexplainer_global_resumo.csv',
    index=False
)

df_class_pivot.to_csv(
    'gnnexplainer_global_por_classe.csv'
)

print("\nGNNEXPLAINER GLOBAL CONCLUÍDO")
print(f"Pedidos explicados: {len(orders_to_explain)}")
print(f"Registros coletados: {len(df_global)}")
print("CSV exportados.")