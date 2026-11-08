"""Temporal training and evaluation orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit

from src.config.settings import DEVICE, HIDDEN_CHANNELS, LEARNING_RATE, OUT_CHANNELS
from src.models.heterognn import GNNModel
from src.models.xgboost_baseline import (
    make_xgboost_features,
    train_xgboost_baseline,
)


def _filter_edges(edge_index_dict, train_indices, evaluation_indices, device):
    train_set = set(int(index) for index in train_indices)
    evaluation_set = set(int(index) for index in evaluation_indices)
    train_edges = {}
    evaluation_edges = {}
    for edge_type, edge_index in edge_index_dict.items():
        source_type, _, target_type = edge_type
        if target_type == "order":
            train_mask = torch.tensor(
                [int(index) in train_set for index in edge_index[1]],
                device=device,
            )
            evaluation_mask = torch.tensor(
                [int(index) in evaluation_set for index in edge_index[1]],
                device=device,
            )
        elif source_type == "order":
            train_mask = torch.tensor(
                [int(index) in train_set for index in edge_index[0]],
                device=device,
            )
            evaluation_mask = torch.tensor(
                [int(index) in evaluation_set for index in edge_index[0]],
                device=device,
            )
        else:
            train_mask = evaluation_mask = torch.ones(
                edge_index.shape[1], dtype=torch.bool, device=device
            )
        train_edges[edge_type] = edge_index[:, train_mask]
        evaluation_edges[edge_type] = edge_index[:, evaluation_mask]
    return train_edges, evaluation_edges


def train_gnn_fold(data, train_indices, validation_indices, epochs=50):
    model = GNNModel(data.edge_types, HIDDEN_CHANNELS, OUT_CHANNELS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    x_dict = {key: value.to(DEVICE) for key, value in data.x_dict.items()}
    all_edges = {
        key: value.to(DEVICE)
        for key, value in data.edge_index_dict.items()
    }
    train_edges, evaluation_edges = _filter_edges(
        all_edges,
        train_indices,
        np.concatenate([train_indices, validation_indices]),
        DEVICE,
    )
    train_mask = torch.zeros(data["order"].num_nodes, dtype=torch.bool, device=DEVICE)
    train_mask[torch.as_tensor(train_indices, device=DEVICE)] = True
    validation_mask = torch.zeros_like(train_mask)
    validation_mask[torch.as_tensor(validation_indices, device=DEVICE)] = True
    target = data["order"].y.to(DEVICE)
    positive = int((target[train_mask] == 1).sum())
    negative = int((target[train_mask] == 0).sum())
    if positive == 0:
        raise ValueError("A training fold does not contain positive examples.")
    weights = torch.tensor([1.0, negative / positive], device=DEVICE)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(x_dict, train_edges)["order"]
        loss = F.cross_entropy(output[train_mask], target[train_mask], weight=weights)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        output = model(x_dict, evaluation_edges)["order"][validation_mask]
    return model, output, x_dict, evaluation_edges


def run_temporal_experiment(data, df_orders, n_splits=4, gnn_epochs=50):
    """Train and compare Dummy, XGBoost, and GraphSAGE on temporal folds."""
    split = TimeSeriesSplit(n_splits=n_splits)
    results = []
    for fold, (train_indices, validation_indices) in enumerate(
        split.split(df_orders),
        start=1,
    ):
        xgb_features = make_xgboost_features(df_orders)
        target = df_orders["y"]
        dummy_model = DummyClassifier(
            strategy="stratified",
            random_state=42,
        )
        dummy_model.fit(
            xgb_features.iloc[train_indices],
            target.iloc[train_indices],
        )
        y_validation = target.iloc[validation_indices]
        dummy_predictions = dummy_model.predict(
            xgb_features.iloc[validation_indices]
        )
        dummy_probabilities = dummy_model.predict_proba(
            xgb_features.iloc[validation_indices]
        )[:, 1]

        xgb_model, y_validation, xgb_probabilities = train_xgboost_baseline(
            df_orders, train_indices, validation_indices
        )
        gnn_model, gnn_output, x_dict, edge_index = train_gnn_fold(
            data, train_indices, validation_indices, epochs=gnn_epochs
        )
        results.append({
            "fold": fold,
            "train_indices": train_indices,
            "validation_indices": validation_indices,
            "y_true": y_validation.to_numpy(),
            "dummy_model": dummy_model,
            "dummy_predictions": dummy_predictions,
            "dummy_probabilities": dummy_probabilities,
            "xgb_model": xgb_model,
            "xgb_probabilities": xgb_probabilities,
            "gnn_model": gnn_model,
            "gnn_logits": gnn_output.detach().cpu(),
            "x_dict": x_dict,
            "edge_index": edge_index,
        })
        gnn_probabilities = torch.softmax(
            results[-1]["gnn_logits"],
            dim=-1,
        )[:, 1].numpy()
        gnn_predictions = results[-1]["gnn_logits"].argmax(dim=-1).numpy()
        results[-1]["gnn_probabilities"] = gnn_probabilities
        results[-1]["gnn_predictions"] = gnn_predictions
        results[-1]["metrics"] = {
            "dummy": _classification_metrics(
                y_validation, dummy_predictions, dummy_probabilities
            ),
            "xgb": _classification_metrics(
                y_validation,
                (xgb_probabilities >= 0.5).astype(int),
                xgb_probabilities,
            ),
            "gnn": _classification_metrics(
                y_validation, gnn_predictions, gnn_probabilities
            ),
        }
    return results


def _classification_metrics(y_true, predictions, probabilities):
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "pr_auc": average_precision_score(y_true, probabilities),
    }


def temporal_results_table(results) -> pd.DataFrame:
    """Build the fold-by-fold comparison table, including PR-AUC."""
    rows = []
    for result in results:
        for model_name, metrics in result["metrics"].items():
            rows.append({
                "fold": result["fold"],
                "model": model_name,
                **metrics,
            })
    return pd.DataFrame(rows).sort_values(["fold", "model"]).reset_index(
        drop=True
    )
