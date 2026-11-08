"""GNNExplainer integration for order-node predictions."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer


class OrderExplainerModel(torch.nn.Module):
    """Expose only the order logits of a trained two-layer GraphSAGE model."""

    def __init__(self, trained_model):
        super().__init__()
        self.model = trained_model
        self.layer1_relations = trained_model.edge_types
        self.layer2_relations = tuple(
            relation
            for relation in trained_model.edge_types
            if relation[2] == "order"
        )

    def forward(self, x_dict, edge_index_dict):
        first_edges = {
            relation: edges
            for relation, edges in edge_index_dict.items()
            if relation in self.layer1_relations
        }
        first_output = self.model.conv1(x_dict, first_edges)
        first_output = {
            node_type: (
                F.relu(first_output[node_type])
                if node_type in first_output
                else x_dict[node_type]
            )
            for node_type in x_dict
        }
        second_edges = {
            relation: edges
            for relation, edges in edge_index_dict.items()
            if relation in self.layer2_relations
        }
        return self.model.conv2(first_output, second_edges)["order"]


def create_order_explainer(model, epochs=200, learning_rate=0.01):
    """Create a PyG explainer configured for order attribute explanations."""
    wrapped_model = OrderExplainerModel(model).eval()
    return Explainer(
        model=wrapped_model,
        algorithm=GNNExplainer(epochs=epochs, lr=learning_rate),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type=None,
        model_config={
            "mode": "multiclass_classification",
            "task_level": "node",
            "return_type": "raw",
        },
    )


def explain_orders(
    explainer,
    x_dict: Mapping,
    edge_index_dict: Mapping,
    order_indices,
) -> list[dict]:
    """Collect node-attribute summaries for selected order nodes."""
    records = []
    for order_index in order_indices:
        explanation = explainer(
            x_dict,
            edge_index_dict,
            index=int(order_index),
        )
        for node_type, mask in explanation.node_mask_dict.items():
            if mask.numel() == 0:
                continue
            values = mask.detach().cpu()
            records.append({
                "order_idx": int(order_index),
                "node_type": node_type,
                "mean_importance": float(values.mean()),
                "max_importance": float(values.max()),
                "n_features": int(values.numel()),
            })
    return records
