"""SHAP explanations for the modular XGBoost baseline."""

from __future__ import annotations

import pandas as pd
import shap

from src.models.xgboost_baseline import make_xgboost_features


def explain_xgboost(model, df_orders: pd.DataFrame):
    """Return a SHAP explanation using only business features."""
    features = make_xgboost_features(df_orders)
    explainer = shap.TreeExplainer(model)
    return explainer(features), features
